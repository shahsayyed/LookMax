"""
qa_review.py -- QA gate on generated advice text before fine-tuning on it.

Unlike the vision pipeline (synthetic images + 14,079 real photos as a
sanity anchor -- see ML/README.md), this text model's training data is
100% LLM-generated with no independent ground truth. This script is the
mitigation: an automated pass that never deletes a row (same principle as
dataset_generator/extract_measured_labels.py -- an over-strict gate can
skew the trained distribution more than the bad examples it flags), but
marks every row pass/fail with a reason, so a human can spot-check the
failures before they're excluded from training.

MOST IMPORTANT CHECK -- effort vs. genetics: ML/README.md's Architecture
Philosophy is explicit and has already been a real, caught-and-fixed bug
once in the vision pipeline (see its "taxonomy v4" note): advice must never
imply losing weight, changing body shape, or clearing up a skin condition
as the "fix" -- only things changeable in minutes to hours (styling, fit,
grooming choices). An LLM given a tag like `skin_neglected: 2` or a
"heavy set" build description in its context is exactly the kind of prompt
that could drift into "lose weight" advice if not explicitly checked. This
is the same failure mode the image pipeline hit, showing up in a different
pipeline -- checked here for the same reason it was fixed there.

Usage:
    python3 qa_review.py --self-check                          # verify the QA rules themselves, no data needed
    python3 qa_review.py raw_generated/stylist_advice.jsonl     # review a real generated file
    python3 qa_review.py raw_generated/stylist_advice.jsonl --output qa_reviewed/stylist_advice_reviewed.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

# Mirrors ML/README.md's Architecture Philosophy almost verbatim -- these
# are the exact categories of thing a generated fix must never suggest.
_BANNED_PATTERNS = [
    r"\blose weight\b", r"\bweight loss\b", r"\blosing weight\b",
    r"\bdiet\b", r"\bcalorie", r"\bslim down\b", r"\bslimmer\b",
    r"\bclear up (your|the) skin\b", r"\backne\b", r"\bpimple",
    r"\bbody shape\b", r"\bchange your body\b",
    r"\bplastic surgery\b", r"\bbotox\b",
    r"\btone (your|the) (body|muscles?)\b", r"\bwork ?out to\b",
]
_BANNED_RE = re.compile("|".join(_BANNED_PATTERNS), re.IGNORECASE)

_META_CHATTER_OPENERS = (
    "sure", "here's", "here is", "as a stylist", "certainly", "of course",
    "i'd suggest", "i would suggest", "great question", "absolutely",
)

# Loose actionable-verb heuristic -- not a hard fail (advice phrasing is
# varied), just a soft warning surfaced for human review when NONE match,
# since a "fix" that never tells the user to DO something specific is
# suspect regardless of word count.
_ACTIONABLE_VERBS = (
    "cuff", "tuck", "roll", "unbutton", "button", "iron", "steam", "press",
    "trim", "shave", "size down", "size up", "swap", "tie", "polish",
    "shine", "brush", "comb", "style", "layer", "untuck", "hem", "belt",
    "wash", "moisturize", "line up", "clean up", "shorten", "lengthen",
)


def check_record(record):
    """Returns (passed: bool, reasons: list[str]) -- reasons is non-empty
    only on failure or warning; passed is False only on a hard failure
    (word count / meta-chatter / banned phrase), never on the soft
    actionable-verb warning alone."""
    reasons = []
    messages = record.get("messages", [])
    assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return False, ["no assistant message found"]
    content = assistant_msgs[-1].strip()

    if not content:
        return False, ["empty assistant response"]

    word_count = len(content.split())
    if word_count < cfg.MIN_RESPONSE_WORDS:
        reasons.append(f"too short ({word_count} words, need >= {cfg.MIN_RESPONSE_WORDS})")
    if word_count > cfg.MAX_RESPONSE_WORDS:
        reasons.append(f"too long ({word_count} words, need <= {cfg.MAX_RESPONSE_WORDS})")

    lower = content.lower()
    if any(lower.startswith(opener) for opener in _META_CHATTER_OPENERS):
        reasons.append(f"meta-chatter opener: '{content[:30]}...'")

    banned_match = _BANNED_RE.search(content)
    if banned_match:
        reasons.append(f"banned genetics/body-shape phrase: '{banned_match.group(0)}'")

    hard_fail = bool(reasons)

    if not any(verb in lower for verb in _ACTIONABLE_VERBS):
        reasons.append("WARNING: no recognizable actionable verb -- review for vagueness")

    return not hard_fail, reasons


# --------------------------------------------------------------------------
# Self-check -- verifies the QA logic itself against hardcoded fixtures,
# no generated data required. Run this once after editing this file.
# --------------------------------------------------------------------------
_FIXTURES = [
    ("good, should pass",
     "Cuff the denim twice to eliminate ankle stacking; the bunching drags down your silhouette and reads as unintentional rather than styled, especially in bright lighting where the fabric pool is obvious to anyone looking.",
     True),
    ("meta-chatter, should fail",
     "Sure! Here's a tip: cuff your jeans twice at the ankle to clean up the silhouette and make the outfit look more intentional overall today.",
     False),
    ("banned genetics phrase, should fail",
     "Losing weight and clearing up your skin would help this outfit look better on you overall before your interview this week for sure.",
     False),
    ("too short, should fail",
     "Cuff your jeans.",
     False),
    ("too long, should fail",
     "Cuff the denim twice to eliminate ankle stacking, then also consider ironing the shirt, changing the shoes, adjusting the belt, re-tucking the shirt, "
     "swapping the jacket, steaming the trousers, re-lacing the sneakers, brushing the hair, trimming the beard, and polishing the belt buckle before you "
     "leave the house today for the best results overall in every single category available.",
     False),
]


def run_self_check():
    print("Running QA self-check against hardcoded fixtures...\n")
    all_ok = True
    for label, text, expected_pass in _FIXTURES:
        record = {"messages": [{"role": "assistant", "content": text}]}
        passed, reasons = check_record(record)
        status = "OK" if passed == expected_pass else "MISMATCH"
        if status == "MISMATCH":
            all_ok = False
        print(f"[{status}] {label}: passed={passed} (expected {expected_pass})")
        if reasons:
            for r in reasons:
                print(f"    - {r}")
    print(f"\nSelf-check {'PASSED' if all_ok else 'FAILED'}.")
    if not all_ok:
        sys.exit(1)


def run_review(input_path, output_path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0
    warned = 0
    hard_failed = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ok, reasons = check_record(record)
            total += 1
            if ok:
                passed += 1
                if reasons:
                    warned += 1
            else:
                hard_failed += 1
            record["qa"] = {"pass": ok, "reasons": reasons}
            fout.write(json.dumps(record) + "\n")

    print(f"Reviewed {total} examples: {passed} passed ({warned} with warnings), {hard_failed} hard-failed.")
    print(f"Wrote {output_path} (every row kept -- filter on qa.pass at training time, never deleted here).")
    if total and hard_failed / total > 0.10:
        print(f"\n⚠ {hard_failed}/{total} ({100*hard_failed/total:.0f}%) hard-failed QA -- inspect the "
              f"'reasons' field before fine-tuning. A high failure rate usually means the generation "
              f"prompt needs adjusting (see generate_synthetic_dataset.py's SYSTEM_PROMPT), not that "
              f"this gate is over-strict.")


def main():
    parser = argparse.ArgumentParser(description="QA review for generated stylist advice.")
    parser.add_argument("input", nargs="?", help="Input JSONL from generate_synthetic_dataset.py.")
    parser.add_argument("--output", default=None, help="Output path (default: qa_reviewed/<input filename>).")
    parser.add_argument("--self-check", action="store_true", help="Verify the QA rules against fixtures, no data needed.")
    args = parser.parse_args()

    if args.self_check:
        run_self_check()
        return

    if not args.input:
        parser.error("input is required unless --self-check is passed.")

    output_path = args.output or str(cfg.QA_REVIEWED_DIR / Path(args.input).name)
    run_review(args.input, output_path)


if __name__ == "__main__":
    main()
