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

Parallelism & Throughput Comparison:
  - In-pipeline batching (--batch-size N): Batches same-resolution prompts
    into a single forward pass.
  - Multi-worker concurrency (--num-workers N): Runs N independent worker
    processes simultaneously on the GPU/machine, allowing two or more sets
    to run in memory independently.
  - Mode comparison (--compare-modes / --benchmark): Runs a quick benchmark
    on test prompts across single vs batched vs multi-worker configurations,
    printing a throughput comparison table before running the full 64 images
    or the 28,000-image production run.

Usage:
    python3 variation_test.py --dry-run
    python3 variation_test.py --compare-modes           # compare single vs batch vs 2 workers
    python3 variation_test.py                          # 64 real images (sequential)
    python3 variation_test.py --batch-size 2           # batched generation
    python3 variation_test.py --num-workers 2          # 2 independent workers running concurrently
    python3 variation_test.py --samples-per-cell 3     # 48 images instead of 64
    python3 variation_test.py --output-dir /data/variation_test_output
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
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
                index = len(tasks)
                filename = f"{index:03d}_{category}_{tier}_{sample_idx}.png"
                tasks.append({
                    "index": index,
                    "category": category,
                    "tier": tier,
                    "sample_idx": sample_idx,
                    "filename": filename,
                    "task": task
                })
    return tasks


def group_by_resolution(pending, batch_size):
    """Group consecutive same-resolution tasks up to batch_size."""
    group = []
    group_res = None
    for item in pending:
        res = item["task"]["resolution"]
        if group and (res != group_res or len(group) >= batch_size):
            yield group
            group = []
        group.append(item)
        group_res = res
    if group:
        yield group


def run_dry_run(tasks, batch_size=1, shard=None, num_shards=None, num_workers=1):
    print(f"{'='*88}\nDRY RUN -- {len(tasks)} total tasks planned, no GPU\n{'='*88}\n")
    if num_workers > 1 and shard is None:
        print(f"Multi-worker configuration: {num_workers} concurrent workers will be spawned.")
        print(f"Each worker will handle ~{len(tasks) // num_workers} tasks.\n")

    if shard is not None:
        if num_shards is None:
            sys.exit("--shard requires --num-shards")
        tasks = [t for t in tasks if t["index"] % num_shards == shard]
        print(f"Shard {shard}/{num_shards}: {len(tasks)} tasks assigned to this worker.")

    batches = list(group_by_resolution(tasks, batch_size))
    print(f"Batch configuration: batch_size={batch_size} -> {len(batches)} batch(es) for {len(tasks)} tasks.\n")

    for t in tasks[:6]:
        row = pb.row_for_csv(t["category"], t["tier"], t["filename"], t["task"])
        print(f"--- {t['filename']}  (res={t['task']['resolution']}, score={row['score']}) ---")
        print(t["task"]["prompt"])
        print()
    if len(tasks) > 6:
        print(f"... and {len(tasks) - 6} more prompts.")
    print(f"\nDry run complete -- {len(tasks)} prompts verified. Read a sample before spending GPU time.")


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


def run_generation(tasks, output_dir, batch_size=1, shard=None, num_shards=None, device=None):
    import qwen_pipeline as qp

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if shard is not None:
        if num_shards is None:
            sys.exit("--shard requires --num-shards")
        tasks = [t for t in tasks if t["index"] % num_shards == shard]
        manifest_path = output_dir / f"manifest_shard{shard}.csv"
        worker_tag = f"[Shard {shard}/{num_shards}] "
    else:
        manifest_path = output_dir / "manifest.csv"
        worker_tag = ""

    print(f"{worker_tag}Loading Qwen-Image-2512 on device {device or 'default'}...")
    pipe, can_batch = qp.load_pipeline(device=device)

    effective_batch = batch_size
    if not can_batch and effective_batch > 1:
        print(f"{worker_tag}Note: requested batch_size={effective_batch} but this GPU needs CPU offload -- forcing 1.")
        effective_batch = 1

    batches = list(group_by_resolution(tasks, effective_batch))
    print(f"{worker_tag}Generating {len(tasks)} images in {len(batches)} batch(es) (batch_size={effective_batch})...")

    all_field_names = set()
    rows = []
    generated_count = 0
    t_start = time.time()

    for b_idx, batch in enumerate(batches):
        filenames = [item["filename"] for item in batch]
        task_specs = [item["task"] for item in batch]
        seeds = [item["index"] for item in batch]

        t0 = time.time()
        images = qp.generate(pipe, task_specs, seeds=seeds, num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
        elapsed = time.time() - t0

        for item, filename, image in zip(batch, filenames, images):
            image.save(images_dir / filename)
            row = pb.row_for_csv(item["category"], item["tier"], filename, item["task"])
            row["prompt"] = item["task"]["prompt"]
            all_field_names.update(row.keys())
            rows.append(row)

        generated_count += len(batch)
        print(f"{worker_tag}[{generated_count}/{len(tasks)}] Generated {len(batch)} image(s) in {elapsed:.1f}s "
              f"({elapsed / len(batch):.2f}s/image): {', '.join(filenames)}")

    qp.unload(pipe)
    total_elapsed = time.time() - t_start

    field_order = ["filename", "category", "tier", "score"] + sorted(
        f for f in all_field_names if f not in ("filename", "category", "tier", "score", "prompt")
    ) + ["prompt"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n{worker_tag}Wrote {len(rows)} images to {images_dir}")
    print(f"{worker_tag}Wrote manifest to {manifest_path}")
    if len(rows) > 0:
        print(f"{worker_tag}Finished in {total_elapsed:.1f}s (average {total_elapsed / len(rows):.2f}s/image, "
              f"{len(rows) / (total_elapsed / 60.0):.1f} images/min).")

    if shard is None:
        print("\nReview checklist:")
        print("  - flaw_severe vs flaw_mild: worse, not identical, across the several samples per cell")
        print("  - flaw tiers read as a bad day, not a costume")
        print("  - polished tier: more than one archetype across samples (not a jacket every time)")
        print("  - backgrounds/lighting vary freely across tiers, not just clustering around polished")
        print("  - requested colour pairs/patterns render as coherent, wearable garments")


def run_multi_worker(samples_per_cell, output_dir, num_workers, batch_size=1, device=None):
    """Launch num_workers concurrent worker processes on the GPU, each handling a shard."""
    output_dir = Path(output_dir)
    check_disk_space(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*88}\nCONCURRENT MULTI-WORKER EXECUTION ({num_workers} workers, batch_size={batch_size})\n{'='*88}\n")
    print(f"Spawning {num_workers} independent worker processes simultaneously on the GPU...")

    script_path = Path(__file__).resolve()
    processes = []
    t_start = time.time()

    for k in range(num_workers):
        cmd = [
            sys.executable, str(script_path),
            "--samples-per-cell", str(samples_per_cell),
            "--shard", str(k),
            "--num-shards", str(num_workers),
            "--batch-size", str(batch_size),
            "--output-dir", str(output_dir),
        ]
        if device:
            cmd.extend(["--device", device])
        print(f"Launching Worker {k}: {' '.join(cmd)}")
        p = subprocess.Popen(cmd)
        processes.append((k, p))

    failed = False
    for k, p in processes:
        ret = p.wait()
        if ret != 0:
            print(f"Worker {k} failed with exit code {ret}!")
            failed = True

    if failed:
        sys.exit("One or more worker processes failed. Check logs above.")

    total_elapsed = time.time() - t_start

    # Merge shard manifests into one unified manifest.csv
    print("\nAll workers completed. Merging shard manifests...")
    shard_csvs = sorted(output_dir.glob("manifest_shard*.csv"))
    merged_rows = []
    seen_files = set()
    all_fieldnames = set()

    for sc in shard_csvs:
        with open(sc, newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fn = r.get("filename")
                if fn and fn not in seen_files:
                    seen_files.add(fn)
                    merged_rows.append(r)
                    all_fieldnames.update(r.keys())
        sc.unlink()  # Remove temporary shard CSV

    field_order = ["filename", "category", "tier", "score"] + sorted(
        f for f in all_fieldnames if f not in ("filename", "category", "tier", "score", "prompt")
    ) + ["prompt"]

    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=field_order)
        writer.writeheader()
        for r in merged_rows:
            writer.writerow(r)

    print(f"Successfully merged {len(merged_rows)} rows into {manifest_path}")
    print(f"Total multi-worker run time: {total_elapsed:.1f}s ({total_elapsed / len(merged_rows):.2f}s/image, "
          f"{len(merged_rows) / (total_elapsed / 60.0):.1f} images/min).")
    print("\nReview checklist:")
    print("  - flaw_severe vs flaw_mild: worse, not identical, across the several samples per cell")
    print("  - flaw tiers read as a bad day, not a costume")
    print("  - polished tier: more than one archetype across samples (not a jacket every time)")
    print("  - backgrounds/lighting vary freely across tiers, not just clustering around polished")
    print("  - requested colour pairs/patterns render as coherent, wearable garments")


def run_compare_modes(samples_per_cell=4, output_dir=None, device=None):
    """Benchmark and compare generation modes:
    1. Sequential (1 worker, batch=1)
    2. Batched (1 worker, batch=2)
    3. Concurrent Independent Sets (2 workers, batch=1)
    4. Batched Concurrent Sets (2 workers, batch=2)
    """
    import qwen_pipeline as qp

    print(f"\n{'='*88}\nBENCHMARK & MODE COMPARISON\n{'='*88}")
    print("Evaluating throughput across Sequential, Batched, and Concurrent Multi-Worker modes.")
    print("Test sample: 4 tasks (2 grooming square, 2 outfit portrait).\n")

    tasks = build_variation_tasks(samples_per_cell)
    # Pick 2 grooming and 2 outfit tasks for a balanced sample
    sample_tasks = [tasks[0], tasks[1], tasks[32], tasks[33]]

    test_out = (Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR) / "mode_benchmark_tmp"
    test_out.mkdir(parents=True, exist_ok=True)

    results = []

    # 1. Warm-up
    print("Running warm-up forward pass...")
    pipe, can_batch = qp.load_pipeline(device=device)
    qp.generate(pipe, [sample_tasks[0]["task"]], seeds=[999], num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
    qp.unload(pipe)

    # Mode 1: Sequential (1 worker, batch=1)
    print("\n[Mode 1] Testing Sequential Generation (workers=1, batch=1)...")
    pipe, can_batch = qp.load_pipeline(device=device)
    t0 = time.time()
    for item in sample_tasks:
        qp.generate(pipe, [item["task"]], seeds=[item["index"]], num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
    t_seq = time.time() - t0
    qp.unload(pipe)
    results.append(("Sequential (Workers=1, Batch=1)", 1, 1, t_seq))
    print(f"  -> {len(sample_tasks)} images in {t_seq:.1f}s ({t_seq/len(sample_tasks):.2f}s/image)")

    # Mode 2: Batched (1 worker, batch=2)
    if can_batch:
        print("\n[Mode 2] Testing Batched Generation (workers=1, batch=2)...")
        pipe, can_batch = qp.load_pipeline(device=device)
        t0 = time.time()
        for batch in group_by_resolution(sample_tasks, 2):
            qp.generate(pipe, [item["task"] for item in batch], seeds=[item["index"] for item in batch],
                        num_inference_steps=tx.NUM_INFERENCE_STEPS_TEST)
        t_batch = time.time() - t0
        qp.unload(pipe)
        results.append(("Batched (Workers=1, Batch=2)", 1, 2, t_batch))
        print(f"  -> {len(sample_tasks)} images in {t_batch:.1f}s ({t_batch/len(sample_tasks):.2f}s/image)")
    else:
        print("\n[Mode 2] Batched generation skipped (VRAM offload mode forces batch=1).")

    # Mode 3: Concurrent Independent Workers (workers=2, batch=1)
    print("\n[Mode 3] Testing Concurrent Independent Workers (workers=2, batch=1)...")
    sub_dir = test_out / "concurrent_workers"
    sub_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    script_path = Path(__file__).resolve()
    p0 = subprocess.Popen([
        sys.executable, str(script_path), "--samples-per-cell", "1", "--shard", "0", "--num-shards", "2",
        "--batch-size", "1", "--output-dir", str(sub_dir)
    ])
    p1 = subprocess.Popen([
        sys.executable, str(script_path), "--samples-per-cell", "1", "--shard", "1", "--num-shards", "2",
        "--batch-size", "1", "--output-dir", str(sub_dir)
    ])
    p0.wait()
    p1.wait()
    t_concurrent = time.time() - t0
    # samples_per_cell=1 gives 16 tasks total (8 per worker)
    results.append(("Concurrent (Workers=2, Batch=1)", 2, 1, (t_concurrent / 16.0) * len(sample_tasks)))
    print(f"  -> Normalized {len(sample_tasks)} images in {results[-1][3]:.1f}s ({results[-1][3]/len(sample_tasks):.2f}s/image)")

    # Clean up benchmark temp dir
    shutil.rmtree(test_out, ignore_errors=True)

    # Print Summary Table
    total_variation_images = samples_per_cell * 16
    total_production_images = sum(tx.CATEGORY_COUNTS.values())

    print(f"\n{'='*88}\nMODE COMPARISON RESULTS SUMMARY\n{'='*88}")
    print(f"{'Mode':<34} | {'Workers':>7} | {'Batch':>5} | {'Sec/Image':>9} | {'Img/Min':>7} | {'64-Img Test':>11} | {'28k Full Run':>12}")
    print("-" * 98)

    base_sec = results[0][3] / len(sample_tasks)
    for name, workers, b_size, total_t in results:
        sec_img = total_t / len(sample_tasks)
        img_min = 60.0 / sec_img
        var_min = (sec_img * total_variation_images) / 60.0
        full_hrs = (sec_img * total_production_images) / 3600.0
        speedup = base_sec / sec_img
        speedup_str = f" ({speedup:.2f}x)" if speedup != 1.0 else ""
        print(f"{name:<34} | {workers:>7} | {b_size:>5} | {sec_img:>8.2f}s | {img_min:>7.1f} | {var_min:>9.1f}m | {full_hrs:>10.1f}h{speedup_str}")

    print("-" * 98)
    print("Recommendation:")
    fastest = min(results, key=lambda x: x[3])
    print(f"  -> Fastest mode on this hardware: '{fastest[0]}' ({fastest[3]/len(sample_tasks):.2f}s/image)")
    print(f"  -> Use this configuration with variation_test.py and full_run.py for maximum throughput.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Broader taxonomy-driven diversity/quality check, run before full_run.py."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print prompts + labels, no GPU.")
    parser.add_argument("--compare-modes", "--benchmark", action="store_true",
                        help="Benchmark and compare single vs batched vs concurrent multi-worker modes.")
    parser.add_argument("--samples-per-cell", type=int, default=DEFAULT_SAMPLES_PER_CELL,
                         help=f"Images per (category, tier) cell (default {DEFAULT_SAMPLES_PER_CELL} "
                              f"-> {DEFAULT_SAMPLES_PER_CELL * 16} total).")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of same-resolution images per forward pass (default 1).")
    parser.add_argument("--num-workers", "--workers", type=int, default=1,
                        help="Number of concurrent worker processes to run on the GPU (default 1).")
    parser.add_argument("--shard", type=int, default=None,
                        help="Specific shard index for this worker (0-based).")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards.")
    parser.add_argument("--device", default=None,
                        help="Target PyTorch device, e.g. 'cuda:0'.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                         help=f"Where images/manifest are written (default {DEFAULT_OUTPUT_DIR}).")
    args = parser.parse_args()

    tasks = build_variation_tasks(args.samples_per_cell)

    if args.compare_modes:
        run_compare_modes(samples_per_cell=args.samples_per_cell, output_dir=args.output_dir, device=args.device)
        return

    if args.dry_run:
        run_dry_run(tasks, batch_size=args.batch_size, shard=args.shard,
                    num_shards=args.num_shards, num_workers=args.num_workers)
        return

    if args.num_workers > 1 and args.shard is None:
        run_multi_worker(args.samples_per_cell, args.output_dir, args.num_workers,
                         batch_size=args.batch_size, device=args.device)
        return

    check_disk_space(args.output_dir)
    run_generation(tasks, args.output_dir, batch_size=args.batch_size,
                   shard=args.shard, num_shards=args.num_shards, device=args.device)


if __name__ == "__main__":
    main()
