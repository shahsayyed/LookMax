#!/usr/bin/env python3
"""
verify_batch.py -- Batch Prediction variant of verify_dataset_vertex.py's VLM
audit. Same rubric, same --manifest input format, same output report shape --
just submitted as Vertex AI Batch Prediction jobs (GCS-staged JSONL in, JSONL
out) instead of N synchronous calls. Cheaper and immune to the online
per-minute rate limit, at the cost of each job's own queue/startup latency
(usually several minutes) -- not a fit for small interactive checks, but the
right tool once we're auditing hundreds+ of images at once.

Large runs are split into --chunk-size-sized batch jobs submitted one after
another (not one giant job) -- keeps each request JSONL a manageable size
(images are base64-inlined, so a single 28,000-image job could run into the
tens of GB), gives incremental progress instead of an all-or-nothing wait,
and means one bad chunk doesn't lose already-completed work. Each chunk's
raw results are cached to disk as soon as it finishes, so a killed/interrupted
run resumes from the last completed chunk instead of re-submitting jobs that
already succeeded (and already cost money).

Requires the service account to have Storage access to write/read GCS_BUCKET
(granted via `gcloud projects add-iam-policy-binding ... --role=roles/storage.admin`).

Usage:
  python3 verify_batch.py --manifest <manifest.json> --images-dir <dir> --label <name>
  python3 verify_batch.py --filter "clean-shaven" --sample-size 200 --label clean_shaven_big
  python3 verify_batch.py --sample-size 28000 --label full_audit --chunk-size 500
"""
import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
IMAGES_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated" / "images"
QA_OUT_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "qa_processed"
CREDENTIALS_FILE = REPO_ROOT / "lookmax-generation-513e3f9ab69e.json"
CHUNKS_CACHE_DIR = SCRIPT_DIR / ".batch_chunks_cache"

GCS_BUCKET = "lookmax-generation-batch-staging"
MODEL = "gemini-3.8-flash"
LOCATION = "global"

sys.path.insert(0, str(SCRIPT_DIR))
from verify_dataset_vertex import VERIFY_SYSTEM_PROMPT, load_prompt_index, sample_images, parse_verdict_json  # noqa: E402


def get_client_and_project():
    with open(CREDENTIALS_FILE) as f:
        project_id = json.load(f)["project_id"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE.resolve())
    from google import genai
    from google.genai import types
    client = genai.Client(vertexai=True, project=project_id, location=LOCATION,
                           http_options=types.HttpOptions(timeout=45000))
    return client, project_id


def build_request_line(key: str, image_path: Path, prompt_text: str) -> dict:
    img_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return {
        "key": key,
        "request": {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": f"ORIGINAL GENERATION PROMPT:\n{prompt_text}\n\nAnalyze this image and return the required JSON object:"},
                ],
            }],
            "system_instruction": {"parts": [{"text": VERIFY_SYSTEM_PROMPT}]},
            "generation_config": {"temperature": 0.1, "response_mime_type": "application/json"},
        },
    }


def submit_chunk(chunk_items, client, chunk_label):
    """Builds + uploads the request JSONL and creates the batch job. Returns
    the job handle immediately (does NOT wait) -- callers submit all chunks
    this way first, then poll them concurrently, so per-job queue/startup
    overhead (several minutes, mostly fixed regardless of chunk size) is
    paid once in parallel instead of once per chunk serially."""
    from google.genai import types

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gcs_prefix = f"gs://{GCS_BUCKET}/batch_verify/{chunk_label}_{ts}"
    input_path = SCRIPT_DIR / f".batch_input_{chunk_label}_{ts}.jsonl"
    with open(input_path, "w") as f:
        for fn, img_path, prompt_text in chunk_items:
            f.write(json.dumps(build_request_line(fn, img_path, prompt_text)) + "\n")

    gcs_input = f"{gcs_prefix}/input.jsonl"
    subprocess.run(["gcloud", "storage", "cp", str(input_path), gcs_input],
                    check=True, capture_output=True)
    input_path.unlink(missing_ok=True)

    job = client.batches.create(
        model=MODEL,
        src=gcs_input,
        config=types.CreateBatchJobConfig(dest=f"{gcs_prefix}/output/"),
    )
    print(f"    [{chunk_label}] job {job.name.split('/')[-1]} submitted ({len(chunk_items)} images)", flush=True)
    return job


def collect_chunk_results(job, project_id, chunk_label):
    """Downloads + parses a SUCCEEDED job's output. Raises if it didn't succeed."""
    if "SUCCEEDED" not in str(job.state):
        raise RuntimeError(f"[{chunk_label}] job did not succeed: state={job.state}, error={getattr(job, 'error', None)}")

    dest_uri = job.dest.gcs_uri if job.dest else None
    local_out_dir = SCRIPT_DIR / f".batch_output_{chunk_label}"
    local_out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gcloud", "storage", "cp", "-r", dest_uri.rstrip("/") + "/*", str(local_out_dir),
                     "--project", project_id], check=True, capture_output=True)

    results = []
    for pf in local_out_dir.rglob("*.jsonl"):
        for line in pf.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("key")
            try:
                text = rec["response"]["candidates"][0]["content"]["parts"][0]["text"]
                verdict = parse_verdict_json(text) or {"verdict": "error", "issues": ["unparseable batch response"]}
            except Exception as e:
                verdict = {"verdict": "error", "issues": [f"batch response parse failure: {e}"]}
            verdict["filename"] = key
            results.append(verdict)

    shutil.rmtree(local_out_dir, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Batch-Prediction VLM audit of synthetic dataset images.")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--images-dir", type=str, default=None)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--label", type=str, default="batch")
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--chunk-size", type=int, default=500,
                         help="Max images per individual batch job (default 500). Large runs are split into "
                              "this many images per job.")
    parser.add_argument("--max-concurrent", type=int, default=6,
                         help="Max batch jobs in flight at once (default 6). Chunks are submitted in a sliding "
                              "window -- each job's queue/startup overhead (several minutes, mostly fixed "
                              "regardless of chunk size) is paid once in parallel instead of once per chunk "
                              "serially, without submitting hundreds of jobs simultaneously.")
    args = parser.parse_args()

    if args.manifest:
        if not args.images_dir:
            sys.exit("--images-dir is required with --manifest")
        images_dir = Path(args.images_dir)
        manifest = json.loads(Path(args.manifest).read_text())
        items = [(fn, images_dir / fn, rec["prompt"]) for fn, rec in manifest.items() if (images_dir / fn).exists()]
    else:
        images_dir = IMAGES_DIR
        prompt_index = load_prompt_index()
        candidates = None
        if args.filter:
            import re
            pat = re.compile(args.filter, re.IGNORECASE)
            candidates = {fn for fn, rec in prompt_index.items() if pat.search(rec["prompt"])}
        sample = sample_images(args.sample_size, args.seed, candidates, images_dir)
        items = [(p.name, p, prompt_index[p.name]["prompt"]) for p in sample if p.name in prompt_index]

    if not items:
        sys.exit("No images to verify.")

    chunks = [items[i:i + args.chunk_size] for i in range(0, len(items), args.chunk_size)]
    print(f"Verifying {len(items)} images via Batch Prediction ({MODEL}), "
          f"split into {len(chunks)} job(s) of up to {args.chunk_size} each")

    client, project_id = get_client_and_project()
    CHUNKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    pending_idxs = []  # chunk indices not yet cached, still to submit
    for i in range(len(chunks)):
        cache_path = CHUNKS_CACHE_DIR / f"{args.label}_chunk{i:04d}.json"
        if cache_path.exists():
            chunk_results = json.loads(cache_path.read_text())
            print(f"[chunk {i+1}/{len(chunks)}] already done (cached), {len(chunk_results)} results -- skipping")
            all_results.extend(chunk_results)
        else:
            pending_idxs.append(i)

    in_flight = {}  # chunk_index -> job handle
    next_idx_pos = 0  # position into pending_idxs

    def top_up():
        nonlocal next_idx_pos
        while len(in_flight) < args.max_concurrent and next_idx_pos < len(pending_idxs):
            i = pending_idxs[next_idx_pos]
            next_idx_pos += 1
            job = submit_chunk(chunks[i], client, f"{args.label}_c{i:04d}")
            in_flight[i] = job

    def poll_with_retry(job_name, max_retries=5):
        """A multi-hour run will see transient network blips (DNS, TLS handshake
        timeouts, etc.) on the polling calls -- retry a few times with backoff
        rather than letting one hiccup kill hours of already-in-flight work."""
        for attempt in range(max_retries):
            try:
                return client.batches.get(name=job_name)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = (attempt + 1) * 15
                print(f"    [poll retry {attempt+1}/{max_retries} after {type(e).__name__}, waiting {wait}s]", flush=True)
                time.sleep(wait)

    top_up()
    done_count = len(chunks) - len(pending_idxs)
    while in_flight:
        time.sleep(args.poll_interval)
        for i in list(in_flight.keys()):
            job = poll_with_retry(in_flight[i].name)
            if not str(job.state).endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED")):
                in_flight[i] = job
                continue
            del in_flight[i]
            chunk_results = collect_chunk_results(job, project_id, f"{args.label}_c{i:04d}")
            cache_path = CHUNKS_CACHE_DIR / f"{args.label}_chunk{i:04d}.json"
            cache_path.write_text(json.dumps(chunk_results, indent=2))
            all_results.extend(chunk_results)
            done_count += 1
            print(f"[chunk {i+1}/{len(chunks)}] done, {len(chunk_results)} results "
                  f"({done_count}/{len(chunks)} chunks complete, {len(in_flight)} in flight)", flush=True)
            top_up()

    results = all_results
    valid = sum(1 for r in results if r.get("verdict") == "valid")
    invalid = sum(1 for r in results if r.get("verdict") == "invalid")
    errors = sum(1 for r in results if r.get("verdict") == "error")
    total = len(results)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Sample size : {total}")
    if total:
        print(f"Valid       : {valid} ({valid/total*100:.1f}%)")
        print(f"Invalid     : {invalid} ({invalid/total*100:.1f}%)")
        print(f"Errors      : {errors} ({errors/total*100:.1f}%)")

    QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = QA_OUT_DIR / f"vlm_batch_report_{args.label}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({"model": MODEL, "sample_size": total, "generated_at": ts, "chunk_size": args.chunk_size,
                    "summary": {"valid": valid, "invalid": invalid, "errors": errors},
                    "results": results}, f, indent=2)
    print(f"\nFull report written to: {out_path.relative_to(REPO_ROOT)}")

    # Only clear the resume cache for this label once everything succeeded and was written out.
    for f in CHUNKS_CACHE_DIR.glob(f"{args.label}_chunk*.json"):
        f.unlink()


if __name__ == "__main__":
    main()
