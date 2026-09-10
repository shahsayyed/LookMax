#!/usr/bin/env python3
"""
verify_dataset_vertex.py -- Random-sample VLM audit of generated synthetic images
against the exact prompt each was generated from, using Gemini via Vertex AI
(same service-account auth as ML/stylist_llm/test_vertex_connection.py).

For each sampled image, sends the image + its original generation prompt
(deterministically reconstructed via full_run.py's build_full_task_list(),
since most remote hosts' generation_log.jsonl never made it back to the Mac
before their instances were destroyed -- reconstruction is exact: verified
byte-for-byte against a real logged prompt) to Gemini and asks it to judge:
  1. Prompt alignment  -- does the image actually show what was requested
     (garments, colors, hair, background, framing)?
  2. Anatomy / photorealism -- deformed hands, extra/missing limbs, multiple
     people, cropped head/feet, watermarks/text -- the exact failure modes
     the negative_prompt was written to suppress.

Writes a JSON report to ML/data/vision_synthetic/qa_processed/ and prints a
pass-rate summary. Does not modify or delete any image -- audit only.

Usage:
  python3 verify_dataset_vertex.py --sample-size 30
  python3 verify_dataset_vertex.py --sample-size 50 --seed 7
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
RAW_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "raw_generated"
IMAGES_DIR = RAW_DIR / "images"
QA_OUT_DIR = REPO_ROOT / "ML" / "data" / "vision_synthetic" / "qa_processed"
CREDENTIALS_FILE = REPO_ROOT / "lookmax-generation-513e3f9ab69e.json"

sys.path.insert(0, str(SCRIPT_DIR))

MODEL = "gemini-3.8-flash"
LOCATION = "global"

VERIFY_SYSTEM_PROMPT = """\
You are a QA auditor for a synthetic image dataset used to train a computer-
vision model on human body type, face shape, and clothing attributes. You
will be shown one generated image and the exact text prompt it was generated
from. This dataset intentionally includes flawed/unflattering appearances and
grooming (that is the point of several tiers) -- your job is NOT to judge
attractiveness or catch cosmetic AI-rendering noise. It is to catch images
where the LABEL the image will be trained under does not match what's
actually shown, because that teaches the model the wrong thing.

Judge exactly these four things:

1. body_type_match: Does the person's build/weight match what the prompt
   specified (e.g. "heavy set", "slender", "average build")? Fail only on a
   clear category mismatch (e.g. prompt says heavy set, image shows slender),
   not on subtle proportion variation.
2. face_match: Does face shape, apparent age, and ethnicity roughly match
   what the prompt specified? Fail only on a clear mismatch (e.g. requested
   age 55 but image looks 20s; requested ethnicity clearly not depicted).
3. attribute_match: Are the SPECIFIC garment/hair/grooming attributes named
   in the prompt actually visible -- correct garment TYPE, requested COLOR,
   requested PATTERN, hair description, and (if this is a flaw tier) the
   named flaw itself (e.g. "patchy beard", "clashing makeup colors",
   "cropped/stained clothing")? A wrong garment type or wrong color is a
   fail. A garment that is merely a slightly different cut/style of the
   same requested type, in the requested color, is NOT a fail.
4. severe_anatomy_defect: Only flag this for defects severe enough to make
   the image unusable for training -- extra or missing limbs, more than one
   person, a cropped head or feet in a full-body shot, a visible watermark/
   text/logo, or hands so deformed the finger count is wrong. Do NOT flag:
   minor hand/finger stiffness, mild skin smoothing, slightly-off camera
   angle or framing, minor fabric wrinkling not specified in the prompt, or
   any other small AI-rendering artifact -- these are expected and are not
   dataset defects.

verdict is "invalid" if body_type_match is false, OR face_match is false, OR
attribute_match is false, OR severe_anatomy_defect is true. Otherwise "valid".
Do not use a "borderline" verdict -- decide.

Return ONLY a JSON object, no markdown fences, no commentary:
{
  "body_type_match": true|false,
  "face_match": true|false,
  "attribute_match": true|false,
  "severe_anatomy_defect": true|false,
  "verdict": "valid" | "invalid",
  "issues": ["short phrase per REAL problem found (per the 4 checks above only), empty list if none"]
}
"""


def load_prompt_index() -> dict:
    """filename -> {"prompt", "category", "tier"}, deterministically reconstructed
    for all 28,000 possible filenames (see module docstring)."""
    from full_run import build_full_task_list
    tasks = build_full_task_list()
    return {
        t["filename"]: {
            "prompt": t["task"]["prompt"],
            "category": t["category"],
            "tier": t["tier"],
        }
        for t in tasks
    }


def sample_images(n: int, seed: int, candidates: set[str] | None = None, images_dir: Path = IMAGES_DIR) -> list[Path]:
    """candidates, if given, restricts to this set of filenames (e.g. only
    filenames whose reconstructed prompt matches a --filter regex)."""
    real = [
        f for f in images_dir.iterdir()
        if f.is_file() and f.stat().st_size > 0 and f.name.endswith(".png")
        and (candidates is None or f.name in candidates)
    ]
    rng = random.Random(seed)
    return rng.sample(real, min(n, len(real)))


def get_client():
    if not CREDENTIALS_FILE.exists():
        sys.exit(f"Missing service account key: {CREDENTIALS_FILE}")
    with open(CREDENTIALS_FILE) as f:
        project_id = json.load(f)["project_id"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE.resolve())

    from google import genai
    from google.genai import types
    return genai.Client(
        vertexai=True, project=project_id, location=LOCATION,
        http_options=types.HttpOptions(timeout=45000),
    )


def parse_verdict_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        return None


def verify_one(client, image_path: Path, prompt_text: str, max_retries: int = 4) -> dict:
    from google.genai import types

    with open(image_path, "rb") as f:
        img_bytes = f.read()

    contents = [
        types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        f"ORIGINAL GENERATION PROMPT:\n{prompt_text}\n\nAnalyze this image and return the required JSON object:",
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=VERIFY_SYSTEM_PROMPT,
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            parsed = parse_verdict_json(response.text or "")
            if parsed is not None:
                return parsed
        except Exception as e:
            err = str(e)
            if ("429" in err or "RESOURCE_EXHAUSTED" in err) and attempt < max_retries - 1:
                time.sleep((attempt + 1) * 5)
                continue
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return {"verdict": "error", "issues": [f"Gemini call failed: {err[:150]}"]}
    return {"verdict": "error", "issues": ["failed to get a parseable JSON verdict after retries"]}


def main():
    parser = argparse.ArgumentParser(description="Random-sample VLM audit of synthetic dataset images.")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--filter", type=str, default=None,
                         help="Case-insensitive regex matched against each image's reconstructed prompt text; "
                              "restricts sampling to only matching images (targeted category audit).")
    parser.add_argument("--label", type=str, default=None, help="Name for this category, used only in output filename.")
    parser.add_argument("--manifest", type=str, default=None,
                         help="Path to a prompt_experiment.py manifest.json instead of the production dataset -- "
                              "verifies every image it lists (no sampling) against its own recorded prompt.")
    parser.add_argument("--images-dir", type=str, default=None,
                         help="Directory the manifest's images live in (required with --manifest).")
    args = parser.parse_args()

    print("=" * 80)
    print(f"LookMax Synthetic Dataset -- VLM Sample Audit ({MODEL} via Vertex AI)")
    print("=" * 80)

    candidates = None
    if args.manifest:
        if not args.images_dir:
            sys.exit("--images-dir is required with --manifest")
        images_dir = Path(args.images_dir)
        with open(args.manifest) as f:
            manifest = json.load(f)
        prompt_index = {fn: {"prompt": rec["prompt"], "category": rec.get("category"), "tier": rec.get("tier")}
                        for fn, rec in manifest.items()}
        sample = [images_dir / fn for fn in prompt_index if (images_dir / fn).exists()]
        print(f"Manifest '{args.manifest}': verifying all {len(sample)} images (no sampling)\n")
    else:
        images_dir = IMAGES_DIR
        prompt_index = load_prompt_index()
        print(f"Reconstructed {len(prompt_index):,} deterministic prompts via build_full_task_list()")

        if args.filter:
            import re
            pat = re.compile(args.filter, re.IGNORECASE)
            candidates = {fn for fn, rec in prompt_index.items() if pat.search(rec["prompt"])}
            print(f"Filter '{args.filter}' matches {len(candidates):,} of 28,000 possible tasks")

        sample = sample_images(args.sample_size, args.seed, candidates, images_dir)
        print(f"Sampled {len(sample)} images (seed={args.seed})\n")

    client = get_client()

    results = []
    for i, img_path in enumerate(sample, 1):
        rec = prompt_index.get(img_path.name)
        if rec is None:
            print(f"[{i}/{len(sample)}] {img_path.name}: SKIP (no matching generation_log entry)")
            continue

        verdict = verify_one(client, img_path, rec["prompt"])
        verdict["filename"] = img_path.name
        verdict["category"] = rec.get("category")
        verdict["tier"] = rec.get("tier")
        results.append(verdict)

        tag = verdict.get("verdict", "?")
        issues = ", ".join(verdict.get("issues", [])) or "-"
        print(f"[{i}/{len(sample)}] {img_path.name}: {tag}  ({issues})")

    valid = sum(1 for r in results if r.get("verdict") == "valid")
    invalid = sum(1 for r in results if r.get("verdict") == "invalid")
    errors = sum(1 for r in results if r.get("verdict") == "error")
    body_fails = sum(1 for r in results if r.get("body_type_match") is False)
    face_fails = sum(1 for r in results if r.get("face_match") is False)
    attr_fails = sum(1 for r in results if r.get("attribute_match") is False)
    anatomy_fails = sum(1 for r in results if r.get("severe_anatomy_defect") is True)
    total = len(results)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Sample size : {total}")
    if total:
        print(f"Valid       : {valid} ({valid/total*100:.1f}%)")
        print(f"Invalid     : {invalid} ({invalid/total*100:.1f}%)")
        print(f"Errors      : {errors} ({errors/total*100:.1f}%)")
        print("\nFailure breakdown (an invalid image can fail more than one check):")
        print(f"  body_type_match failed       : {body_fails}")
        print(f"  face_match failed            : {face_fails}")
        print(f"  attribute_match failed       : {attr_fails}")
        print(f"  severe_anatomy_defect        : {anatomy_fails}")
        if args.manifest:
            print(f"\n(Manifest mode: {total} images verified, no dataset-wide projection.)")
        else:
            pass_rate = valid / total
            pool_size = len(candidates) if candidates is not None else 28000
            pool_desc = f"the {pool_size:,}-image '{args.filter}' category" if candidates is not None else "the full 28,000-image dataset"
            proj_invalid = round((1 - pass_rate) * pool_size)
            print(f"\nProjected across {pool_desc}: ~{proj_invalid:,} images may need review/regeneration "
                  f"if this sample's {invalid} invalid rate ({invalid/total*100:.1f}%) holds.")

    QA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label_part = f"_{args.label}" if args.label else ""
    out_path = QA_OUT_DIR / f"vlm_sample_report{label_part}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL,
            "sample_size": total,
            "seed": args.seed,
            "filter": args.filter,
            "filter_pool_size": len(candidates) if candidates is not None else None,
            "generated_at": ts,
            "summary": {
                "valid": valid, "invalid": invalid, "errors": errors,
                "body_type_match_failed": body_fails, "face_match_failed": face_fails,
                "attribute_match_failed": attr_fails, "severe_anatomy_defect": anatomy_fails,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nFull report written to: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
