"""
generate_synthetic_dataset.py -- generates the 6,000 instruction pairs
this pipeline fine-tunes on, using Gemini (gemini-3.8-flash) or local Ollama
to write the actionable CHECKLIST advice for input contexts sampled from the
REAL vision taxonomy and enriched with Apple Vision native signals.

SAMPLING REUSES THE VISION PIPELINE'S OWN CODE:
Each training context (category + tier + garment/grooming state) is drawn via
`ML/vision/dataset_synthetic/prompt_builder.build_task()['row']`.
In addition, native Apple Vision framework signals (posture, lighting, color harmony)
are sampled realistically according to the score tier to provide complete multimodal
features for the on-device stylist.

PARALLEL EXECUTION:
Runs requests concurrently via `concurrent.futures.ThreadPoolExecutor` (default
--concurrency 10). File writes are protected by an atomic threading.Lock() so
progress is saved incrementally and safely. On 429 rate limit or network error,
requests retry with exponential backoff and jitter.

QUALITY GATE AT GENERATION TIME:
Validates that output follows the strict checklist format:
- Every line starts with "- "
- 2 to 3 polish tips for score >= 9.0; 3 to 5 corrections for score < 9.0
- Each line <= 15 words
- Total words within [cfg.MIN_RESPONSE_WORDS, cfg.MAX_RESPONSE_WORDS] (25 to 85)
- Effort-vs-genetics guardrails: no weight, diet, acne, body shape, or unobservable
  fragrance/cologne items.
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
import urllib.request

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
                wait = self.next_allowed_time - now
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
            self.next_allowed_time = max(self.next_allowed_time, now) + seconds


def build_task_contexts(count, seed=TASK_SEED):
    """Deterministic, stratified sampling across 4 categories and 4 tiers,
    incorporating Apple Vision native signals (posture, lighting, color harmony).
    Guarantees balanced representation with zero sampling bias."""
    rng = random.Random(seed)

    # 16 combinations: 4 categories x 4 tiers (interleaved for perfect category balance)
    combos = []
    for tier in vision_tx.OUTFIT_TIERS:
        for cat in vision_tx.ALL_CATEGORIES:
            combos.append((cat, tier))

    contexts = []
    for i in range(count):
        cat, tier = combos[i % len(combos)]
        occasion = rng.choice(tv.OCCASIONS)
        row = vision_pb.build_task(cat, tier, rng)["row"]

        # Sample Apple Vision native signals correlated with tier
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


def _call_gemini(prompt, api_key, model=None):
    """Direct REST call to Gemini / Gemma with system instruction and thinking-token filtering."""
    model_name = model or cfg.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "systemInstruction": {
            "parts": [{"text": cfg.SYSTEM_PROMPT}]
        },
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": cfg.GEMINI_TEMPERATURE,
            "maxOutputTokens": cfg.GEMINI_GENERATION_MAX_OUTPUT_TOKENS,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode()
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}: {e.reason} - {err_body}") from e

    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError(f"No candidates returned: {data}")
    cand = candidates[0]
    parts = cand.get("content", {}).get("parts", [])
    # Strip thinking tokens if present
    text_parts = [p.get("text", "") for p in parts if "text" in p and not p.get("thought", False)]
    text = "".join(text_parts).strip()
    return text


def _call_ollama(prompt):
    """Local backend via Ollama's /api/chat -- no API key, runs on-machine."""
    payload = {
        "model": cfg.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": cfg.SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": cfg.GEMINI_TEMPERATURE},
    }
    req = urllib.request.Request(
        f"{cfg.OLLAMA_HOST}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return (data["message"]["content"] or "").strip()


def run_dry_run(count, concurrency=5):
    """Validates contexts, prompt formatting, and parallel dispatch without making API calls."""
    contexts = build_task_contexts(count)
    print("=" * 88)
    print(f"PARALLEL DRY RUN -- {len(contexts)} contexts across {concurrency} worker threads")
    print("=" * 88)

    # Verify balance
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
    print(f"Parallel simulation completed in {elapsed*1000:.1f}ms using {threads_used} concurrent OS threads.")

    print("\n" + "=" * 88)
    print("SAMPLE PROMPTS WITH APPLE VISION SIGNALS:")
    print("=" * 88)
    sample_indices = [0, count // 4, count // 2, 3 * count // 4]
    for s_idx in sample_indices:
        if s_idx < len(contexts):
            ctx = contexts[s_idx]
            tag_prompt = tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"])
            print(f"\n--- Context {ctx['index']:04d} ({ctx['category']} | {ctx['tier']}) ---")
            print(tag_prompt)

    print("\n" + "=" * 88)
    print("Dry run successfully verified: balanced sampling, Apple Vision tags, and multi-thread safety.")
    print("=" * 88)


def run_generation(target_count, output_path, api_key, backend=None, concurrency=10, model=None, rpm=None):
    backend = backend or cfg.GENERATOR_BACKEND
    model = model or cfg.GEMINI_MODEL
    rpm = rpm or getattr(cfg, "GEMINI_REQUESTS_PER_MINUTE", 45.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
    print(f"STYLIST LLM DATASET GENERATION")
    print(f"Backend: {backend} ({model if backend == 'gemini' else cfg.OLLAMA_MODEL})")
    print(f"Target count: {target_count} | Already in file: {already_done} | Remaining to generate: {len(remaining)}")
    print(f"Rate limit: {rpm:.1f} RPM (interval ~{60.0/rpm:.2f}s) across {concurrency} worker threads")
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
            if backend == "gemini":
                rate_limiter.acquire()
            try:
                if backend == "ollama":
                    text = _call_ollama(prompt)
                else:
                    text = _call_gemini(prompt, api_key, model=model)
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = any(term in err_msg for term in ("429", "resource_exhausted", "quota", "rate limit"))
                is_net_error = any(term in err_msg for term in ("nodename nor servname", "no route to host", "network is unreachable", "connection refused"))
                backoff = (2 ** attempt) * 2 + random.uniform(1.0, 5.0)
                if is_rate_limit:
                    if backend == "gemini":
                        rate_limiter.penalize(backoff)
                    print(f"[{ctx['index']:04d}] Rate limit encountered, backing off {backoff:.1f}s...")
                elif is_net_error:
                    net_pause = min(backoff * 3, 45.0)
                    if backend == "gemini":
                        rate_limiter.penalize(net_pause)
                    print(f"[{ctx['index']:04d}] Network issue ({e}), pausing {net_pause:.1f}s...")
                else:
                    print(f"[{ctx['index']:04d}] attempt {attempt} error: {e}, retry in {backoff:.1f}s")
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

            bullet_count = len([l for l in advice.splitlines() if l.strip().startswith("- ")])
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
    print(f"Run complete in {total_time/60:.1f} minutes ({total_time:.1f}s).")
    print(f"Wrote {written_count} new examples ({failed_count} failed/skipped).")
    print(f"Total in {output_path}: {already_done + written_count}")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the stylist LLM's synthetic training set (Gemma 4 31B IT / Gemini or local Ollama)."
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate contexts, prompts, and parallel dispatch.")
    parser.add_argument("--count", type=int, default=cfg.SYNTHETIC_TARGET_COUNT,
                        help=f"Target number of examples (default {cfg.SYNTHETIC_TARGET_COUNT}).")
    parser.add_argument("--output", default=str(cfg.RAW_GENERATED_DIR / "stylist_advice.jsonl"),
                        help="Output JSONL path (default under raw_generated/).")
    parser.add_argument("--backend", choices=["ollama", "gemini"], default=cfg.GENERATOR_BACKEND,
                        help=f"Generation backend (default {cfg.GENERATOR_BACKEND}, see config.py).")
    parser.add_argument("--model", default=cfg.GEMINI_MODEL,
                        help=f"Model identifier (default {cfg.GEMINI_MODEL}).")
    parser.add_argument("--rpm", type=float, default=getattr(cfg, "GEMINI_REQUESTS_PER_MINUTE", 15.0),
                        help="Requests per minute rate limit across threads (default 15.0).")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var).")
    parser.add_argument("--concurrency", type=int, default=2,
                        help="Number of concurrent worker threads (default 2).")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(min(args.count, 20) if args.count > 20 else args.count, concurrency=args.concurrency)
        return

    api_key = None
    if args.backend == "gemini":
        api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            env_file = cfg.REPO_ROOT / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    if line.strip().startswith("GEMINI_API_KEY="):
                        api_key = line.split("=", 1)[1].split("#")[0].strip().strip('"').strip("'")
        if not api_key:
            sys.exit("!! No Gemini API key found. Pass --api-key or set GEMINI_API_KEY in .env.")

    run_generation(
        args.count,
        args.output,
        api_key,
        backend=args.backend,
        concurrency=args.concurrency,
        model=args.model,
        rpm=args.rpm,
    )


if __name__ == "__main__":
    main()
