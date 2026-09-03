"""
generate_synthetic_dataset.py -- generates the 3,500-5,000 instruction pairs
this pipeline fine-tunes on, using Gemini to write the ADVICE text for
input contexts sampled from the REAL vision taxonomy.

SAMPLING REUSES THE VISION PIPELINE'S OWN CODE, not a hand-invented
combinatorial list: each training context (category + tier + garment/
grooming state) is drawn via
`ML/pipeline/dataset_generator/prompt_builder.build_task()['row']` --
literally the same function that labels the Qwen image dataset. This
guarantees the tag combinations this LLM trains on are exactly as coherent
and realistic as the ones the vision model itself was trained to recognize
(same formality-consistent outfit sampling, same severity bands), rather
than a second, independently-invented tag vocabulary that could drift from
what the real on-device vision model will ever actually emit. See
tag_vocabulary.py's module docstring for the full rationale.

SCOPE: this script only builds the pipeline. Per project decision, it does
NOT run automatically -- calling Gemini for 5,000 real examples costs real
money and needs your own GEMINI_API_KEY. Everything here is fully runnable
and tested up to the API call itself:
    python3 generate_synthetic_dataset.py --dry-run --count 10
prints 10 example contexts and the EXACT prompt that would be sent to
Gemini for each, with no network call and no API key required. Read this
before spending anything on the real 5,000-example run.

OUTPUT FORMAT: standard Hugging Face JSONL `{"messages": [...]}`, one JSON
object per line -- ready for `trl.SFTTrainer` with no reformatting.

RESUME: output is a plain append-only JSONL file. The number of lines
already in it IS the resume point (task contexts are sampled from a single
seeded rng consumed sequentially -- see build_task_contexts() -- so
"already have N lines" means "skip the first N sampled contexts", the same
resume principle full_run.py uses for the image pipeline, just simpler
because there's no per-file existence check needed for a single JSONL).

QUALITY GATE AT GENERATION TIME: any response outside the 30-50 word
window, or containing an obvious meta-chatter opener ("Sure!", "Here's",
"As a stylist,"), is rejected and retried (up to MAX_RETRIES_PER_EXAMPLE)
rather than silently kept -- catching this here is cheaper than catching
it in qa_review.py after 5,000 examples are already generated.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "dataset_generator"))
import config as cfg
import tag_vocabulary as tv
import taxonomy as vision_tx
import prompt_builder as vision_pb

TASK_SEED = 7  # separate deterministic sequence from the vision pipeline's TASK_SEED=42 / variation_test.py's 99
MAX_RETRIES_PER_EXAMPLE = 3
_META_CHATTER_OPENERS = (
    "sure", "here's", "here is", "as a stylist", "certainly", "of course",
    "i'd suggest", "i would suggest", "great question", "absolutely",
)


def build_task_contexts(count, seed=TASK_SEED):
    """Deterministic: `count` (category, tier, occasion, row) contexts, in a
    fixed order. Same seed every run -- resuming a partial run or re-running
    after a taxonomy.py tweak means index N is always the same context."""
    rng = random.Random(seed)
    contexts = []
    for i in range(count):
        category = rng.choice(vision_tx.ALL_CATEGORIES)
        tier = rng.choice(vision_tx.OUTFIT_TIERS)  # GROOMING_TIERS/OUTFIT_TIERS are identical
        occasion = rng.choice(tv.OCCASIONS)
        row = vision_pb.build_task(category, tier, rng)["row"]
        contexts.append({"index": i, "category": category, "tier": tier, "occasion": occasion, "row": row})
    return contexts


def _validate_response(text):
    """Returns (ok, reason). Enforced here, not just left for qa_review.py --
    catching a bad generation immediately is cheap (one retry); catching it
    after the whole run is 5,000 wasted API calls' worth of hindsight."""
    stripped = text.strip()
    if not stripped:
        return False, "empty response"
    lower = stripped.lower()
    if any(lower.startswith(opener) for opener in _META_CHATTER_OPENERS):
        return False, f"meta-chatter opener: '{stripped[:30]}...'"
    word_count = len(stripped.split())
    if word_count < cfg.MIN_RESPONSE_WORDS:
        return False, f"too short ({word_count} words, need >= {cfg.MIN_RESPONSE_WORDS})"
    if word_count > cfg.MAX_RESPONSE_WORDS:
        return False, f"too long ({word_count} words, need <= {cfg.MAX_RESPONSE_WORDS})"
    return True, None


def _gemini_client(api_key):
    from google import genai
    return genai.Client(api_key=api_key)


def _call_gemini(client, tag_block):
    from google.genai import types as genai_types

    response = client.models.generate_content(
        model=cfg.GEMINI_MODEL,
        contents=[tag_block],
        config=genai_types.GenerateContentConfig(
            system_instruction=cfg.SYSTEM_PROMPT,
            temperature=cfg.GEMINI_TEMPERATURE,
            max_output_tokens=cfg.MAX_NEW_TOKENS + 20,  # headroom over the 50-word cap before truncation
        ),
    )
    return (response.text or "").strip()


class _PacedRateLimiter:
    """Minimal local rate limiter -- deliberately not importing
    real_data_pipeline's PacedRateLimiter to keep this pipeline's isolation
    (see PLAN.md); same requests-per-minute value already proven safe for
    the Gemini API in that script."""
    def __init__(self, requests_per_minute):
        self._min_interval = 60.0 / requests_per_minute
        self._last_call = 0.0

    def wait(self):
        elapsed = time.monotonic() - self._last_call
        remaining = self._min_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_call = time.monotonic()


def run_dry_run(count):
    contexts = build_task_contexts(count)
    print(f"{'='*88}\nDRY RUN -- {len(contexts)} training contexts, no API call\n{'='*88}\n")
    for ctx in contexts:
        prompt = tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"])
        print(f"--- context {ctx['index']:04d}  ({ctx['category']} / {ctx['tier']}) ---")
        print(f"[system]\n{cfg.SYSTEM_PROMPT}\n")
        print(f"[user]\n{prompt}\n")
    print(f"Dry run complete -- {len(contexts)} contexts. Read a sample before running the real "
          f"generation (needs GEMINI_API_KEY and costs real money).")


def run_generation(target_count, output_path, api_key):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done = 0
    if output_path.exists():
        with open(output_path) as f:
            already_done = sum(1 for _ in f)
    if already_done >= target_count:
        print(f"{output_path} already has {already_done} examples (target {target_count}) -- nothing to do.")
        return

    contexts = build_task_contexts(target_count)
    remaining = contexts[already_done:]
    print(f"Resuming at index {already_done} -- {len(remaining)} examples left to generate "
          f"(target {target_count}).")

    client = _gemini_client(api_key)
    limiter = _PacedRateLimiter(cfg.GEMINI_REQUESTS_PER_MINUTE)

    written = 0
    failed = 0
    with open(output_path, "a") as f:
        for ctx in remaining:
            prompt = tv.format_tag_prompt(ctx["category"], ctx["occasion"], ctx["row"])
            advice = None
            for attempt in range(1, MAX_RETRIES_PER_EXAMPLE + 1):
                limiter.wait()
                try:
                    text = _call_gemini(client, prompt)
                except Exception as e:
                    print(f"[{ctx['index']:04d}] attempt {attempt}: API error: {e}")
                    continue
                ok, reason = _validate_response(text)
                if ok:
                    advice = text
                    break
                print(f"[{ctx['index']:04d}] attempt {attempt}: rejected ({reason})")

            if advice is None:
                print(f"[{ctx['index']:04d}] FAILED after {MAX_RETRIES_PER_EXAMPLE} attempts -- skipping.")
                failed += 1
                continue

            record = {
                "messages": [
                    {"role": "system", "content": cfg.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": advice},
                ],
                "meta": {
                    "category": ctx["category"], "tier": ctx["tier"], "occasion": ctx["occasion"],
                    "priority_defect": tv.priority_defect(ctx["category"], ctx["row"]),
                    "score": ctx["row"].get("score"),
                },
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            written += 1
            if written % 50 == 0:
                print(f"... {already_done + written}/{target_count} written ({failed} failed so far)")

    print(f"\nDone. Wrote {written} new examples ({failed} failed/skipped) to {output_path}")
    print(f"Total in file: {already_done + written}")


def main():
    parser = argparse.ArgumentParser(description="Generate the stylist LLM's synthetic training set via Gemini.")
    parser.add_argument("--dry-run", action="store_true", help="Print contexts + prompts, no API call.")
    parser.add_argument("--count", type=int, default=cfg.SYNTHETIC_TARGET_COUNT,
                         help=f"Target number of examples (default {cfg.SYNTHETIC_TARGET_COUNT}).")
    parser.add_argument("--output", default=str(cfg.RAW_GENERATED_DIR / "stylist_advice.jsonl"),
                         help="Output JSONL path (default under raw_generated/).")
    parser.add_argument("--api-key", default=None, help="Gemini API key (or set GEMINI_API_KEY env var).")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run(min(args.count, 20) if args.count > 20 else args.count)
        return

    import os
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("!! No Gemini API key found. Pass --api-key or set GEMINI_API_KEY.")

    run_generation(args.count, args.output, api_key)


if __name__ == "__main__":
    main()
