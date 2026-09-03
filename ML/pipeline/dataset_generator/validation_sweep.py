"""
validation_sweep.py -- the gate before the full run.

  --coverage-only   Simulate the FULL 28,000-image plan (same deterministic
                     task list full_run.py actually uses -- imported from
                     there, not re-derived, so this can never drift out of
                     sync with what the real run produces) with NO GPU.
                     Prints projected per-class counts for every trained
                     categorical/ordinal column and flags anything under
                     250 examples. Run this BEFORE spending any GPU time.

  --check-binding    After generating some images (e.g. via smoke_test.py
                     --per-tier or a partial full_run.py), measure how
                     often the requested colour actually landed in the
                     rendered pixels. Gate: colour >= 70%.
                     Garment TYPE and PATTERN adherence are explicitly
                     NOT measured here -- see the note printed by this
                     mode. Distinguishing "wearing a blazer" from "wearing
                     a cardigan" or "striped" from "checked" reliably from
                     raw pixels needs a real classifier/VLM, which is out
                     of scope for a pixel-only script; the spec's 80%/65%
                     thresholds for those two are left as documented
                     targets for a future VLM-based pass (or manual
                     review -- see the checklist this mode also prints).

Usage:
    python3 validation_sweep.py --coverage-only
    python3 validation_sweep.py --check-binding <labels_csv> <images_dir>
"""
import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx
import full_run

THIN_CLASS_THRESHOLD = 250

# Gate thresholds from the build spec.
COLOR_MATCH_GATE = 0.70
GARMENT_TYPE_GATE = 0.80   # documented target only -- see module docstring
PATTERN_GATE = 0.65        # documented target only -- see module docstring

# Fields that aren't classification targets -- skip thin-class flagging
# for these (meta/provenance and the continuous score).
SKIP_FIELDS = {"score"}


def run_coverage_only():
    print("Simulating the full 28,000-image plan (no GPU) using full_run.build_full_task_list()...\n")
    tasks = full_run.build_full_task_list()
    print(f"Total planned images: {len(tasks)}")

    per_category_counts = defaultdict(lambda: defaultdict(Counter))
    for item in tasks:
        category = item["category"]
        row = item["task"]["row"]
        for field, value in row.items():
            if field in SKIP_FIELDS:
                continue
            per_category_counts[category][field][value] += 1

    any_thin = False
    for category in tx.ALL_CATEGORIES:
        total = tx.CATEGORY_COUNTS[category]
        print(f"\n{'='*78}\n{category}  (planned total: {total})\n{'='*78}")
        for field in sorted(per_category_counts[category]):
            counter = per_category_counts[category][field]
            values_str = ", ".join(f"{k}={v}" for k, v in sorted(counter.items(), key=lambda kv: str(kv[0])))
            print(f"  {field:28s} {values_str}")
            thin = {k: v for k, v in counter.items() if v < THIN_CLASS_THRESHOLD}
            if thin:
                any_thin = True
                print(f"    !! THIN (<{THIN_CLASS_THRESHOLD}): {thin}")

    print(f"\n{'='*78}")
    if any_thin:
        print(f"RESULT: THIN CLASSES FOUND (< {THIN_CLASS_THRESHOLD} examples). "
              f"Do not run full_run.py until this is resolved.")
        sys.exit(1)
    else:
        print(f"RESULT: no class under {THIN_CLASS_THRESHOLD} examples across the full planned run. Clear to generate.")


def run_check_binding(labels_csv, images_dir):
    import extract_measured_labels as eml

    print(f"Measuring colour binding for images referenced in {labels_csv}...")
    output_csv = str(Path(labels_csv).with_name(Path(labels_csv).stem + "_binding_check.csv"))
    eml.process(labels_csv, images_dir, output_csv)

    import csv as csv_mod
    with open(output_csv, newline="") as f:
        rows = list(csv_mod.DictReader(f))

    outfit_rows = [r for r in rows if tx.CATEGORY_KIND.get(r.get("category", "")) == "outfit"]
    if not outfit_rows:
        print("No outfit rows found -- colour binding only applies to Outfit categories.")
        return

    upper_matches = [int(r["color_match_upper"]) for r in outfit_rows if r.get("color_match_upper") not in ("", None)]
    lower_matches = [int(r["color_match_lower"]) for r in outfit_rows
                      if r.get("color_match_lower") not in ("", None) and r.get("requested_lower_color") not in ("", "none")]

    def _rate(vals):
        return (sum(vals) / len(vals)) if vals else None

    upper_rate = _rate(upper_matches)
    lower_rate = _rate(lower_matches)
    combined = upper_matches + lower_matches
    combined_rate = _rate(combined)

    print(f"\n{'='*78}\nCOLOUR BINDING\n{'='*78}")
    if upper_rate is not None:
        print(f"  Upper-body colour match: {upper_rate:.1%}  ({sum(upper_matches)}/{len(upper_matches)})")
    if lower_rate is not None:
        print(f"  Lower-body colour match: {lower_rate:.1%}  ({sum(lower_matches)}/{len(lower_matches)})")
    if combined_rate is not None:
        gate_str = "PASS" if combined_rate >= COLOR_MATCH_GATE else "FAIL"
        print(f"  Combined: {combined_rate:.1%}  (gate >= {COLOR_MATCH_GATE:.0%})  [{gate_str}]")

    print(f"\n  Garment TYPE adherence (gate target >= {GARMENT_TYPE_GATE:.0%}): NOT AUTOMATICALLY MEASURED.")
    print(f"  Pattern adherence (gate target >= {PATTERN_GATE:.0%}): NOT AUTOMATICALLY MEASURED.")
    print("  Both require a real garment classifier/VLM to check reliably from pixels; use the manual "
          "review checklist below for these two until such a pass exists.")

    print(f"\n{'='*78}\nMANUAL REVIEW CHECKLIST\n{'='*78}")
    print("  For a sample of generated images, confirm by eye:")
    print("  [ ] flaw_severe / flaw_mild images genuinely look sloppy -- not just 'slightly casual'.")
    print("  [ ] flaw / average / polished read as THREE visually distinct groups side by side, not a")
    print("      blur of near-identical images. Qwen-2512's improved realism can silently soften grease,")
    print("      wrinkles, and bad fit toward its aesthetic prior -- this failure is SILENT (the label")
    print("      says flaw_severe either way) and poisons exactly the end of the scale the product")
    print("      depends on most (a low score has to look genuinely earned).")
    print("  [ ] requested garment TYPE actually appears (e.g. a blazer request actually renders a blazer,")
    print("      not a generic jacket).")
    print("  [ ] requested pattern actually appears where prominent (striped/checked especially).")
    print("  [ ] no implausible garment/colour pairing slipped past taxonomy.py's restrictions.")
    print(f"\nWrote per-row measured/QA columns to {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Pre-flight / post-generation validation gates.")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--check-binding", nargs=2, metavar=("labels_csv", "images_dir"))
    args = parser.parse_args()

    if not args.coverage_only and not args.check_binding:
        parser.error("Pass --coverage-only and/or --check-binding <labels_csv> <images_dir>.")

    if args.coverage_only:
        run_coverage_only()
    if args.check_binding:
        run_check_binding(*args.check_binding)


if __name__ == "__main__":
    main()
