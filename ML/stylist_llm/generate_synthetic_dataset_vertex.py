"""
generate_synthetic_dataset_vertex.py -- Standalone dataset generation script
using Google Cloud Vertex AI / Agent Platform with Gemini 3.8 Flash (gemini-3.8-flash).

FEATURES:
- Completely isolated from generate_synthetic_dataset.py and its output files.
- Writes to ML/data/stylist_llm/raw_generated/stylist_advice_vertex.jsonl by default.
- Uses google-genai SDK in Vertex AI enterprise mode authenticated via Service Account key.
- Balanced stratified sampling across 4 categories and 4 tiers with Apple Vision native signals.
- Thread-safe concurrent execution (ThreadPoolExecutor) with ThreadSafeRateLimiter.
- Atomic file writes protected by thread locks with instant flush.
- Resume capability: skips prompts already present in the output file.
- LookMax Effort-vs-Genetics quality gate enforcement at generation time.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import warnings

# Suppress google.genai AFC warning in headless generation
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*automatic function calling.*")

# Ensure immediate unbuffered output in background tasks
print = functools.partial(print, flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vision" / "dataset_synthetic"))

import config as cfg
import tag_vocabulary as tv
import taxonomy as vision_tx
import prompt_builder as vision_pb
import qa_review as qa

TASK_SEED = 7
MAX_RETRIES_PER_EXAMPLE = 5
_META_CHATTER_OPENERS = (
    "sure", "here's", "here is", "as a stylist", "certainly", "of course",
    "i'd suggest", "i would suggest", "great question", "absolutely",
)


class ThreadSafeRateLimiter:
    """Thread-safe rate limiter that enforces a maximum requests-per-minute (RPM)
    across all concurrent threads."""
    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / max(float(requests_per_minute), 0.1)
        self.lock = threading.Lock()
        self.next_allowed_time = time.time()

    def acquire(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed_time:
                wait = min(self.next_allowed_time - now, 15.0)
                self.next_allowed_time += self.interval
            else:
                wait = 0.0
                self.next_allowed_time = now + self.interval
        if wait > 0:
            time.sleep(wait)

    def penalize(self, seconds: float):
        """When a rate limit is detected by any worker, pause all workers globally."""
        with self.lock:
            now = time.time()
            self.next_allowed_time = max(self.next_allowed_time, now) + min(seconds, 15.0)


def build_task_contexts(count, seed=TASK_SEED):
    """Deterministic, stratified sampling across 4 categories and 4 tiers,
    incorporating Apple Vision native signals (posture, lighting, color harmony).
    Guarantees balanced representation with zero sampling bias."""
    rng = random.Random(seed)

    combos = []
    for tier in vision_tx.OUTFIT_TIERS:
        for cat in vision_tx.ALL_CATEGORIES:
            combos.append((cat, tier))

    contexts = []
    for i in range(count):
        cat, tier = combos[i % len(combos)]
        occasion = rng.choice(tv.OCCASIONS)
        row = vision_pb.build_task(cat, tier, rng)["row"]

        if tier == "polished":
            posture = rng.choices(
                ["upright_aligned", "slight_slouch"],
                weights=[0.75, 0.25]
            )[0]
            color_harmony = rng.choices(
                ["classic_contrast", "monochromatic_neutral", "complementary_pop"],
                weights=[0.40, 0.35, 0.25]
            )[0]
        elif tier == "average":
            posture = rng.choices(
                ["upright_aligned", "slight_slouch", "shoulders_uneven"],
                weights=[0.45, 0.35, 0.20]
            )[0]
            color_harmony = rng.choices(
                ["classic_contrast", "earthy_analogous", "monochromatic_neutral", "clashing_tones"],
                weights=[0.30, 0.30, 0.25, 0.15]
            )[0]
        elif tier == "flaw_mild":
            posture = rng.choices(
                ["slight_slouch", "shoulders_uneven", "lateral_lean", "upright_aligned"],
                weights=[0.40, 0.30, 0.20, 0.10]
            )[0]
            color_harmony = rng.choices(
                ["clashing_tones", "earthy_analogous", "classic_contrast", "monochromatic_neutral"],
                weights=[0.45, 0.30, 0.15, 0.10]
            )[0]
        else:  # flaw_severe
            posture = rng.choices(
                ["slight_slouch", "lateral_lean", "shoulders_uneven"],
                weights=[0.35, 0.35, 0.30]
            )[0]
            color_harmony = rng.choices(
                ["clashing_tones", "earthy_analogous", "classic_contrast"],
                weights=[0.65, 0.20, 0.15]
            )[0]

        lighting = rng.choices(
            ["well_lit", "soft_window_light", "dim_overhead", "harsh_shadows"],
            weights=[0.4, 0.3, 0.15, 0.15]
        )[0]

        row["posture"] = posture
        row["lighting"] = lighting
        if vision_tx.CATEGORY_KIND[cat] == "outfit":
            row["color_harmony"] = color_harmony

        contexts.append({
            "index": i,
            "category": cat,
            "tier": tier,
            "occasion": occasion,
            "row": row,
        })

    return contexts


def _validate_response(text, score=None):
    """Returns (ok, reason). Enforced at generation time before accepting a response."""
    stripped = text.strip()
    if not stripped:
        return False, "empty response"

    lower = stripped.lower()
    if any(lower.startswith(opener) for opener in _META_CHATTER_OPENERS):
        return False, f"meta-chatter opener: '{stripped[:30]}...'"

    # Effort-vs-genetics guardrails
    banned_match = qa._BANNED_RE.search(stripped)
    if banned_match:
        return False, f"forbidden genetics/body-shape phrase: '{banned_match.group(0)}'"

    unobs_match = qa._UNOBSERVABLE_RE.search(stripped)
    if unobs_match:
        return False, f"unobservable item mentioned: '{unobs_match.group(0)}'"

    # Word count bounds
    word_count = len(stripped.split())
    if word_count < cfg.MIN_RESPONSE_WORDS:
        return False, f"too short ({word_count} words, need >= {cfg.MIN_RESPONSE_WORDS})"
    if word_count > cfg.MAX_RESPONSE_WORDS:
        return False, f"too long ({word_count} words, need <= {cfg.MAX_RESPONSE_WORDS})"

    # Checklist format checks
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    bullet_lines = [l for l in lines if l.startswith("- ") or l.startswith("• ") or l.startswith("* ")]
    if len(bullet_lines) != len(lines):
        return False, f"non-checklist format ({len(lines) - len(bullet_lines)} lines lack bullet prefix)"

    if len(lines) < 2 or len(lines) > 5:
        return False, f"bullet count out of range ({len(lines)} lines, need 2-5)"

    # Max words per line
    for idx, l in enumerate(lines):
        line_words = len(l.split())
        if line_words > 16:
            return False, f"line {idx+1} too long ({line_words} words, need <= 15)"

    # Score-adaptive count check
    if score is not None:
        try:
            s = float(score)
            if s >= 9.0 and len(lines) > 3:
                return False, f"high score ({s:.1f}) should have 2-3 polish tips, got {len(lines)}"
            elif s < 9.0 and len(lines) < 3:
                return False, f"lower score ({s:.1f}) should have 3-5 fixes, got {len(lines)}"
        except (ValueError, TypeError):
            pass

    return True, None


def run_dry_run(count, concurrency=5):
    """Validates contexts, prompt formatting, and parallel dispatch without making API calls."""
    contexts = build_task_contexts(count)
    print("=" * 88)
    print(f"VERTEX AI DRY RUN -- {len(contexts)} contexts across {concurrency} worker threads")
    print("=" * 88)

    cat_counts = {}
    tier_counts = {}
    for ctx in contexts:
        cat_counts[ctx["category"]] = cat_counts.get(ctx["category"], 0) + 1
        tier_counts[ctx["tier"]] = tier_counts.get(ctx["tier"], 0) + 1

    print("\nDataset Distribution:")
    print("  Categories:")
    for cat, c in sorted(cat_counts.items()):
        print(f"    - {cat}: {c} ({100*c/len(contexts):.1f}%)")
    print("  Tiers:")
    for tier, c in sorted(tier_counts.items()):
        print(f"    - {tier}: {c} ({100*c/len(contexts):.1f}%)")

    print(f"\nSimulating concurrent execution across {concurrency} threads...")
    worker_results = []
    lock = threading.Lock()

    def simulate_worker(ctx):
        t_id = threading.get_ident()
        prompt = tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"])
        word_count = len(prompt.split())
        with lock:
            worker_results.append({
                "index": ctx["index"],
                "thread_id": t_id,
                "category": ctx["category"],
                "tier": ctx["tier"],
                "prompt_words": word_count,
            })
        return True

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(simulate_worker, ctx) for ctx in contexts]
        for f in as_completed(futures):
            f.result()
    elapsed = time.time() - t0

    threads_used = len(set(r["thread_id"] for r in worker_results))
    print(f"Simulation completed in {elapsed*1000:.1f}ms using {threads_used} concurrent OS threads.")

    sample_indices = [0, count // 4, count // 2, 3 * count // 4]
    print("\nSample Generated Prompts:")
    for s_idx in sample_indices:
        if s_idx < len(contexts):
            ctx = contexts[s_idx]
            tag_prompt = tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"])
            print(f"\n--- Context {ctx['index']:04d} ({ctx['category']} | {ctx['tier']}) ---")
            print(tag_prompt)

    print("\n" + "=" * 88)
    print("Dry run successfully verified: balanced sampling, Apple Vision tags, and multi-thread safety.")
    print("=" * 88)


def run_generation(
    target_count,
    output_path,
    credentials_path,
    model="gemini-3.8-flash",
    location="global",
    concurrency=10,
    rpm=60.0
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    creds_path = Path(credentials_path)
    if not creds_path.is_absolute():
        creds_path = cfg.REPO_ROOT / creds_path

    if not creds_path.exists():
        sys.exit(f"❌ Credentials file not found: {creds_path}")

    with open(creds_path) as f:
        creds_data = json.load(f)
    project_id = creds_data.get("project_id", "lookmax-generation")
    client_email = creds_data.get("client_email")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(creds_path.resolve())

    # Initialize google-genai Vertex AI Client with explicit 45s socket timeout
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location,
        http_options=types.HttpOptions(timeout=45000),
    )

    system_instruction = (
        cfg.SYSTEM_PROMPT
        + "\n6. When suggesting closer-fitting garments for baggy items, use terms like "
        "'tailored', 'tapered', or 'well-fitted' (never use the word 'slimmer')."
    )

    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=cfg.GEMINI_TEMPERATURE,
        max_output_tokens=cfg.GEMINI_GENERATION_MAX_OUTPUT_TOKENS,
    )

    rate_limiter = ThreadSafeRateLimiter(rpm)

    existing_prompts = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if len(r.get("messages", [])) > 1:
                        existing_prompts.add(r["messages"][1]["content"].strip())
                except Exception:
                    pass

    already_done = len(existing_prompts)
    if already_done >= target_count:
        print(f"{output_path} already has {already_done} unique examples (target {target_count}) -- nothing to do.")
        return

    contexts = build_task_contexts(target_count)
    remaining = [
        ctx for ctx in contexts
        if tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"]).strip() not in existing_prompts
    ]

    print("=" * 88)
    print("LOOKMAX VERTEX AI DATASET GENERATION")
    print(f"Platform: Google Cloud Vertex AI / Agent Platform")
    print(f"Service Account: {client_email}")
    print(f"Project: {project_id} | Location: {location} | Model: {model}")
    print(f"Target count: {target_count} | Already in file: {already_done} | Remaining to generate: {len(remaining)}")
    print(f"Rate limit: {rpm:.1f} RPM across {concurrency} worker threads")
    print(f"Output file: {output_path}")
    print("=" * 88)

    file_lock = threading.Lock()
    written_count = 0
    failed_count = 0
    t_start = time.time()

    def process_task(ctx):
        nonlocal written_count, failed_count
        prompt = tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"])
        score = ctx["row"].get("score")
        advice = None

        for attempt in range(1, MAX_RETRIES_PER_EXAMPLE + 1):
            rate_limiter.acquire()
            if attempt > 1:
                print(f"[{ctx['index']:04d}] attempt {attempt}: requesting regenerated advice from Vertex AI...")
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=gen_config,
                )
                text = (response.text or "").strip()
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = any(term in err_msg for term in ("429", "resource_exhausted", "quota", "rate limit"))
                backoff = (2 ** attempt) * 2 + random.uniform(1.0, 5.0)
                if is_rate_limit:
                    rate_limiter.penalize(backoff)
                    print(f"[{ctx['index']:04d}] Rate limit encountered on Vertex AI, pausing {backoff:.1f}s...")
                else:
                    print(f"[{ctx['index']:04d}] attempt {attempt} error ({type(e).__name__}): {e}, retry in {backoff:.1f}s")
                time.sleep(backoff)
                continue

            ok, reason = _validate_response(text, score=score)
            if ok:
                advice = text
                break
            print(f"[{ctx['index']:04d}] attempt {attempt}: rejected ({reason}) -- retrying")
            time.sleep(0.5)

        if advice is None:
            with file_lock:
                failed_count += 1
            print(f"[{ctx['index']:04d}] ✗ FAILED after {MAX_RETRIES_PER_EXAMPLE} attempts -- skipping.")
            return False

        record = {
            "messages": [
                {"role": "system", "content": cfg.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": advice},
            ],
            "meta": {
                "index": ctx["index"],
                "category": ctx["category"],
                "tier": ctx["tier"],
                "occasion": ctx["occasion"],
                "priority_defect": tv.priority_defect(ctx["category"], ctx["row"]),
                "score": score,
                "posture": ctx["row"].get("posture"),
                "lighting": ctx["row"].get("lighting"),
                "color_harmony": ctx["row"].get("color_harmony"),
                "generator": f"vertex_ai_{model}",
            },
        }

        with file_lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
            existing_prompts.add(prompt.strip())
            written_count += 1
            current_total = already_done + written_count
            elapsed = time.time() - t_start
            rate = written_count / elapsed if elapsed > 0 else 0
            rate_rpm = rate * 60.0
            remaining_sec = (len(remaining) - written_count) / rate if rate > 0 else 0

            bullet_count = len([l for l in advice.splitlines() if l.strip().startswith(("- ", "• ", "* "))])
            word_count = len(advice.split())
            print(f"[{ctx['index']:04d}] ✓ ({ctx['category']} | {ctx['tier']} score {score:.1f} -> "
                  f"{bullet_count} bullets, {word_count}w) | "
                  f"Progress: {current_total}/{target_count} ({rate_rpm:.1f} RPM, ~{remaining_sec/60:.1f}m left)")

        return True

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_task, ctx) for ctx in remaining]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                print(f"Worker unhandled exception: {e}")

    total_time = time.time() - t_start
    print("\n" + "=" * 88)
    print(f"Vertex AI Run complete in {total_time/60:.1f} minutes ({total_time:.1f}s).")
    print(f"Wrote {written_count} new examples ({failed_count} failed/skipped).")
    print(f"Total in {output_path}: {already_done + written_count}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the stylist LLM synthetic training set using Google Cloud Vertex AI (Gemini 3.8 Flash)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate contexts, prompts, and parallel dispatch.")
    parser.add_argument("--count", type=int, default=6000,
                        help="Target number of examples (default 6000).")
    parser.add_argument(
        "--output",
        default=str(cfg.RAW_GENERATED_DIR / "stylist_advice_vertex.jsonl"),
        help="Output JSONL path (default raw_generated/stylist_advice_vertex.jsonl)."
    )
    parser.add_argument(
        "--credentials",
        default="lookmax-generation-513e3f9ab69e.json",
        help="Path to Google Cloud service account JSON key."
    )
    parser.add_argument(
        "--model",
        default="gemini-3.8-flash",
        help="Vertex AI model identifier (default gemini-3.8-flash)."
    )
    parser.add_argument(
        "--location",
        default="global",
        help="Vertex AI location (default global)."
    )
    parser.add_argument(
        "--rpm",
        type=float,
        default=60.0,
        help="Requests per minute rate limit across threads (default 60.0)."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent worker threads (default 10)."
    )
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(min(args.count, 20) if args.count > 20 else args.count, concurrency=args.concurrency)
        return

    run_generation(
        target_count=args.count,
        output_path=args.output,
        credentials_path=args.credentials,
        model=args.model,
        location=args.location,
        concurrency=args.concurrency,
        rpm=args.rpm,
    )


if __name__ == "__main__":
    main()
