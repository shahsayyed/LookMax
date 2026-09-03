"""
smoke_test.py -- cheap sanity checks before spending any GPU time.

  --dry-run             Print one prompt + label row per (category x tier)
                         combination (16 total) with NO GPU / NO model
                         load. Read every one of these before running
                         anything else -- this is where an implausible
                         garment/colour pairing or an off-tone effort
                         phrase is cheapest to catch.
  --per-tier             Actually generate one image per (category x tier)
                         (16 images total) using the real model, and write
                         a manifest (prompt + label row + filename) next to
                         them so a human can eyeball flaw vs average vs
                         polished side by side. Requires a GPU.

Usage:
    python3 smoke_test.py --dry-run
    python3 smoke_test.py --dry-run --per-tier    # same as --dry-run alone;
                                                    # --per-tier only adds
                                                    # generation when GPU-backed
    python3 smoke_test.py --per-tier
    python3 smoke_test.py --per-tier --output-dir /data/smoke_test_output
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx
import prompt_builder as pb

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "smoke_test_output"


def iter_smoke_tasks(seed=1234):
    """Deterministic: one task per (category, tier), in a fixed order."""
    import random
    rng = random.Random(seed)
    for category in tx.ALL_CATEGORIES:
        for tier in tx.OUTFIT_TIERS:  # GROOMING_TIERS and OUTFIT_TIERS are identical
            task = pb.build_task(category, tier, rng)
            yield category, tier, task


def run_dry_run():
    print(f"{'='*88}\nDRY RUN -- {len(tx.ALL_CATEGORIES) * len(tx.OUTFIT_TIERS)} prompts, no GPU\n{'='*88}\n")
    for category, tier, task in iter_smoke_tasks():
        filename = f"{category}_{tier}.png"
        row = pb.row_for_csv(category, tier, filename, task)
        print(f"--- {category} / {tier}  (resolution {task['resolution']}, score={row['score']}) ---")
        print(task["prompt"])
        print(f"LABELS: {json.dumps({k: v for k, v in row.items() if k not in ('filename', 'category', 'tier')})}")
        print()
    print(f"NEGATIVE_PROMPT (shared): {tx.NEGATIVE_PROMPT}\n")
    print("Dry run complete. Read every prompt above before running --per-tier or any GPU script.")


def run_per_tier(output_dir):
    import qwen_pipeline as qp

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    print("Loading Qwen-Image-2512 (this can take a while on first run)...")
    pipe, can_batch = qp.load_pipeline()

    tasks = list(iter_smoke_tasks())
    print(f"Generating {len(tasks)} images (one per category x tier), sequentially "
          f"(smoke test -- no reason to batch a 16-image sanity check).")

    all_field_names = set()
    rows = []
    for i, (category, tier, task) in enumerate(tasks):
        filename = f"{i:02d}_{category}_{tier}.png"
        print(f"[{i+1}/{len(tasks)}] {category} / {tier} -> {filename}")
        images = qp.generate(pipe, [task], seeds=[i], num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
        images[0].save(images_dir / filename)

        row = pb.row_for_csv(category, tier, filename, task)
        row["prompt"] = task["prompt"]
        all_field_names.update(row.keys())
        rows.append(row)

    qp.unload(pipe)

    field_order = ["filename", "category", "tier", "score"] + sorted(
        f for f in all_field_names if f not in ("filename", "category", "tier", "score", "prompt")
    ) + ["prompt"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} images to {images_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print("Review images side by side: flaw_severe / flaw_mild / average / polished must read as "
          "visibly distinct groups, not near-identical. Qwen-2512's realism can silently soften "
          "grease/wrinkles/bad fit -- this is the cheapest point to catch that before the full run.")


def main():
    parser = argparse.ArgumentParser(description="Cheap sanity checks for the dataset generator taxonomy/prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts + labels, no GPU.")
    parser.add_argument("--per-tier", action="store_true", help="Generate one real image per category x tier (needs GPU).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where --per-tier writes images/manifest.")
    args = parser.parse_args()

    if not args.dry_run and not args.per_tier:
        parser.error("Pass --dry-run and/or --per-tier.")

    if args.dry_run:
        run_dry_run()
        if args.per_tier:
            print("(--per-tier was also passed, but --dry-run means no GPU/model is touched -- "
                  "drop --dry-run to actually generate the per-tier images.)")
    elif args.per_tier:
        run_per_tier(args.output_dir)


if __name__ == "__main__":
    main()
