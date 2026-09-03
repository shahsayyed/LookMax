"""
variation_test.py -- broader diversity/quality check than smoke_test.py's
--per-tier (16 images, exactly one per category x tier): generates SEVERAL
samples per (category, tier) cell so you can eyeball diversity WITHIN a
tier (different identities, garments, colours, environments), not just the
severity gradient ACROSS tiers.

Default: 4 categories x 4 tiers x 4 samples/cell = 64 images -- the v8
counterpart of the archived pipeline's 64-image
ML/archive/dataset_generator_v7/test_qwen_variations_v7.py. Unlike that
archived script, this one is NOT a standalone duplicate of the sampling
logic -- it calls the real taxonomy.py/prompt_builder.py that full_run.py
uses, just at a small, cheap scale, so what you review here is exactly
what the full run will produce more of.

Run this AFTER quick_prompt_test.py and smoke_test.py --per-tier both look
right, and BEFORE spending real GPU time on full_run.py --benchmark / the
full 28,000-image run -- this is the last and most thorough visual gate.

What to check across the resulting images:
  - Severity gradient: does flaw_severe read as visibly worse than
    flaw_mild for a given category, across the several different
    identities sampled into each cell (not just one lucky pair)?
  - Bad day, not a costume: do flaw-tier images look like a realistic bad
    day, not theatrical distress?
  - Polished diversity: across the polished-tier outfit images, is there
    more than one archetype -- not a jacket/blazer every time?
  - Background/lighting independence: do backgrounds and lighting vary
    freely across tiers (not "polished = studio-lit" as a shortcut the
    model could learn)?
  - Colour/pattern plausibility: do the requested colour pairs and
    patterns render as coherent, wearable garments, not a patchwork?

Usage:
    python3 variation_test.py --dry-run               # print prompts + labels, no GPU
    python3 variation_test.py                          # 64 real images (needs GPU)
    python3 variation_test.py --samples-per-cell 3      # 48 images instead of 64
    python3 variation_test.py --output-dir /data/variation_test_output
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx
import prompt_builder as pb

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "variation_test_output"
DEFAULT_SAMPLES_PER_CELL = 4  # 4 categories x 4 tiers x 4 = 64 images
VARIATION_SEED = 99  # deliberately different from full_run.py's TASK_SEED=42 -- this test's
                      # identities/garments/colours are a separate deterministic sequence, not
                      # just a replay of the first N rows of the real run
MIN_FREE_GB = 70  # just the model cache -- 64 images is a trivial footprint on disk


def build_variation_tasks(samples_per_cell, seed=VARIATION_SEED):
    """Deterministic: samples_per_cell tasks per (category, tier), in a
    fixed order, consuming one seeded rng sequentially. Same seed every
    run -- so re-running this after a taxonomy.py tweak is a like-for-like
    before/after comparison, not fresh random noise each time."""
    rng = random.Random(seed)
    tasks = []
    for category in tx.ALL_CATEGORIES:
        for tier in tx.OUTFIT_TIERS:  # GROOMING_TIERS and OUTFIT_TIERS are identical
            for sample_idx in range(samples_per_cell):
                task = pb.build_task(category, tier, rng)
                filename = f"{len(tasks):03d}_{category}_{tier}_{sample_idx}.png"
                tasks.append({"category": category, "tier": tier, "sample_idx": sample_idx,
                              "filename": filename, "task": task})
    return tasks


def run_dry_run(tasks):
    print(f"{'='*88}\nDRY RUN -- {len(tasks)} prompts, no GPU\n{'='*88}\n")
    for t in tasks:
        row = pb.row_for_csv(t["category"], t["tier"], t["filename"], t["task"])
        print(f"--- {t['filename']}  (score={row['score']}) ---")
        print(t["task"]["prompt"])
        print()
    print(f"Dry run complete -- {len(tasks)} prompts. Read a sample of these before spending "
          f"GPU time on the real run.")


def check_disk_space(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / (1024 ** 3)
    if free_gb < MIN_FREE_GB:
        sys.exit(
            f"!! Only {free_gb:.1f}GB free on {output_dir} -- need at least {MIN_FREE_GB}GB for the "
            f"model cache. Run 'df -h' and confirm this is your LARGE disk. Pass --output-dir to "
            f"point elsewhere if needed."
        )
    print(f"Disk check OK: {free_gb:.1f}GB free on {output_dir} (need >= {MIN_FREE_GB}GB).")


def run_generation(tasks, output_dir):
    import qwen_pipeline as qp

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    print("Loading Qwen-Image-2512 (this can take a while on first run)...")
    pipe, can_batch = qp.load_pipeline()

    print(f"Generating {len(tasks)} images, sequentially -- measured to give no throughput "
          f"benefit from batching on this model (see qwen_pipeline.py's module docstring), and "
          f"this is a quality check, not a throughput benchmark (that's full_run.py --benchmark).")

    all_field_names = set()
    rows = []
    for i, t in enumerate(tasks):
        print(f"[{i+1}/{len(tasks)}] {t['filename']}")
        images = qp.generate(pipe, [t["task"]], seeds=[i], num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
        images[0].save(images_dir / t["filename"])

        row = pb.row_for_csv(t["category"], t["tier"], t["filename"], t["task"])
        row["prompt"] = t["task"]["prompt"]
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
    print("\nReview checklist:")
    print("  - flaw_severe vs flaw_mild: worse, not identical, across the several samples per cell")
    print("  - flaw tiers read as a bad day, not a costume")
    print("  - polished tier: more than one archetype across samples (not a jacket every time)")
    print("  - backgrounds/lighting vary freely across tiers, not just clustering around polished")
    print("  - requested colour pairs/patterns render as coherent, wearable garments")


def main():
    parser = argparse.ArgumentParser(
        description="Broader taxonomy-driven diversity/quality check, run before full_run.py."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts + labels, no GPU.")
    parser.add_argument("--samples-per-cell", type=int, default=DEFAULT_SAMPLES_PER_CELL,
                         help=f"Images per (category, tier) cell (default {DEFAULT_SAMPLES_PER_CELL} "
                              f"-> {DEFAULT_SAMPLES_PER_CELL * 16} total).")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                         help=f"Where images/manifest are written (default {DEFAULT_OUTPUT_DIR}).")
    args = parser.parse_args()

    tasks = build_variation_tasks(args.samples_per_cell)

    if args.dry_run:
        run_dry_run(tasks)
        return

    check_disk_space(args.output_dir)
    run_generation(tasks, args.output_dir)


if __name__ == "__main__":
    main()
