"""
LookMax ML Pipeline — FairFace In-The-Wild Dataset Ingestion Engine
===================================================================
load_fairface_dataset.py

Ingests, filters, and balances candid, in-the-wild, non-celebrity everyday face portraits
from the FairFace Dataset directly into ML/data/1_Raw_Scrapes/ for LookMax Phase 3 VLM
processing and CoreML training.

Why FairFace?
  1. Solves the Celebrity / "Too Attractive" Bias:
     • Contains 108,501 raw Flickr portraits of everyday people with un-retouched skin,
       natural lighting, diverse angles, and ordinary/average grooming.
     • Essential for training Tier 1 ("Needs Improvement") and Tier 2 ("Average") baselines.
  2. Balanced Demographics:
     • Covers 7 ethnic backgrounds and 9 age groups with 50/50 gender balance.
     • Maps cleanly into LookMax demographic buckets (Men/Women Under 35, 35–50, Over 50).
  3. Wide Crop Margins (1.25):
     • Captures full head, hair styling/texture, neckline, and shoulder posture context.

Usage Examples:
    # 1. Inspect dataset matching statistics (Dry Run)
    python3 ML/pipeline/load_fairface_dataset.py --dry-run

    # 2. Ingest 5,000 balanced candid images across all 6 demographic categories
    python3 ML/pipeline/load_fairface_dataset.py --limit 5000

    # 3. Ingest with custom category quotas (e.g. 6,000 total, max 1,000 per demographic)
    python3 ML/pipeline/load_fairface_dataset.py --limit 6000 --limit-per-category 1000

    # 4. Ingest only specific demographic group (e.g. men_u35, men_35_50, men_o50)
    python3 ML/pipeline/load_fairface_dataset.py --category men_u35 --limit 1500
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

# Pipeline Configuration
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import (
        RAW_SCRAPES_DIR,
        VLM_PROCESSING_DIR,
        METADATA_LOGS_DIR,
        ML_ROOT,
        DOWNLOAD_WORKERS,
    )
except ImportError:
    SCRIPT_DIR = Path(__file__).resolve().parent
    ML_ROOT = SCRIPT_DIR.parent
    DATA_ROOT = ML_ROOT / "data"
    RAW_SCRAPES_DIR = DATA_ROOT / "1_Raw_Scrapes"
    VLM_PROCESSING_DIR = DATA_ROOT / "2_VLM_Processing"
    METADATA_LOGS_DIR = VLM_PROCESSING_DIR / "metadata_logs"
    DOWNLOAD_WORKERS = 8

try:
    import pandas as pd
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing dependencies. Please run:")
    print("       pip install pyarrow huggingface-hub pillow pandas tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LookMaxFairFaceLoader")

HF_FAIRFACE_REPO = "HuggingFaceM4/FairFace"

# 1.25 margin wide-crop shards (provides full hair, neck, and shoulder context)
HF_FAIRFACE_SHARDS = [
    "1.25/validation-00000-of-00001-09e3e67bb00ab4ec.parquet",  # 10,954 images (~236 MB)
    "1.25/train-00000-of-00004-e715178553977907.parquet",       # 21,686 images (~465 MB)
    "1.25/train-00001-of-00004-f38b58e3987f3fbf.parquet",       # 21,686 images (~465 MB)
    "1.25/train-00002-of-00004-239e931aa9c3b3e6.parquet",       # 21,686 images (~465 MB)
    "1.25/train-00003-of-00004-847c279691a19548.parquet",       # 21,686 images (~465 MB)
]

AGE_CODE_NAMES = {
    0: "0-2",
    1: "3-9",
    2: "10-19",
    3: "20-29",
    4: "30-39",
    5: "40-49",
    6: "50-59",
    7: "60-69",
    8: "70+",
}

GENDER_CODE_NAMES = {
    0: "Male",
    1: "Female",
}

RACE_CODE_NAMES = {
    0: "East Asian",
    1: "Indian",
    2: "Black",
    3: "White",
    4: "Middle Eastern",
    5: "Latino_Hispanic",
    6: "Southeast Asian",
}

SCRAPED_LOG_FILE = METADATA_LOGS_DIR / "scraped_urls.txt"
FAIRFACE_LOG_FILE = METADATA_LOGS_DIR / "fairface_scrapes.jsonl"


def map_lookmax_demographic(age_code: int, gender_code: int) -> Optional[str]:
    """
    Maps FairFace age & gender integer codes to LookMax demographic categories.
    Rejects infants/kids (0-9 years).
    """
    # Exclude infants and small children
    if age_code in (0, 1):
        return None

    gender_str = "men" if gender_code == 0 else "women"

    if age_code in (2, 3):  # 10-19, 20-29
        return f"face_{gender_str}_u35_average_candid"
    elif age_code in (4, 5):  # 30-39, 40-49
        return f"face_{gender_str}_35_50_average_candid"
    elif age_code in (6, 7, 8):  # 50-59, 60-69, 70+
        return f"face_{gender_str}_o50_average_candid"

    return None


def stream_fairface_records(
    max_records_needed: int,
    limit_per_category: int,
    selected_category: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams candid FairFace records shard-by-shard, balancing demographics
    until category quotas are satisfied.
    """
    category_counts: Dict[str, int] = defaultdict(int)
    total_yielded = 0
    global_seq_id = 0

    for shard_idx, shard_path in enumerate(HF_FAIRFACE_SHARDS):
        if total_yielded >= max_records_needed:
            break

        logger.info(
            "Fetching FairFace candid shard [%d/%d]: %s",
            shard_idx + 1,
            len(HF_FAIRFACE_SHARDS),
            shard_path.split("/")[-1].split("-")[0],
        )

        try:
            local_parquet = hf_hub_download(
                repo_id=HF_FAIRFACE_REPO,
                filename=shard_path,
                repo_type="dataset",
            )
            table = pq.read_table(local_parquet)
        except Exception as e:
            logger.error("Failed to download shard %s: %s", shard_path, e)
            continue

        images = table["image"]
        ages = table["age"].to_pylist()
        genders = table["gender"].to_pylist()
        races = table["race"].to_pylist()
        num_rows = len(table)

        for i in range(num_rows):
            age_code = ages[i]
            gender_code = genders[i]
            race_code = races[i]

            cat = map_lookmax_demographic(age_code, gender_code)
            if not cat:
                continue

            if selected_category and selected_category.lower() not in (
                "all",
                cat.lower(),
                cat.replace("face_", "").lower(),
                cat.replace("face_", "").replace("_average_candid", "").lower(),
            ):
                continue

            if category_counts[cat] >= limit_per_category:
                continue

            img_data = images[i].as_py()
            img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else None
            if not img_bytes:
                continue

            global_seq_id += 1
            category_counts[cat] += 1
            total_yielded += 1

            record = {
                "id": f"{shard_idx}_{global_seq_id:06d}",
                "category": cat,
                "bytes": img_bytes,
                "age_bracket": AGE_CODE_NAMES.get(age_code, str(age_code)),
                "gender": GENDER_CODE_NAMES.get(gender_code, str(gender_code)),
                "race": RACE_CODE_NAMES.get(race_code, str(race_code)),
            }

            yield record

            if total_yielded >= max_records_needed:
                return


def save_image_record(
    record: Dict[str, Any],
    output_dir: Path,
    target_resolution: int = 512,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates, optionally resizes to standard training dimension, and writes image to disk."""
    rec_id = record["id"]
    category = record["category"]
    img_bytes = record["bytes"]

    target_folder = output_dir / category
    target_folder.mkdir(parents=True, exist_ok=True)
    target_path = target_folder / f"fairface_{rec_id}.jpg"

    if target_path.exists() and target_path.stat().st_size > 3000:
        return True, "already_exists", record

    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            # Resize from native 448x448 to standard 512x512 with Lanczos if specified
            if target_resolution and (img.width != target_resolution or img.height != target_resolution):
                img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
            img.save(target_path, "JPEG", quality=95)

        return True, "saved", record
    except Exception as e:
        return False, str(e), record


def run_pipeline(
    output_dir: Path = RAW_SCRAPES_DIR,
    limit: int = 5000,
    limit_per_category: int = 900,
    selected_category: Optional[str] = None,
    resolution: int = 512,
    workers: int = DOWNLOAD_WORKERS,
    dry_run: bool = False,
    stats_only: bool = False,
) -> None:
    """Main execution function for FairFace ingestion."""
    print("=" * 70)
    print(" LookMax — FairFace In-The-Wild (Tier 1 & 2) Ingestion Pipeline")
    print("=" * 70)
    print(f"  Target Limit   : {limit:,} images total (max {limit_per_category} per category)")
    print(f"  Destination    : {output_dir}")
    print(f"  Output Quality : {resolution}x{resolution} px (Lanczos-upscaled from native 448px)")
    print(f"  Parallelism    : {workers} worker threads")
    if selected_category:
        print(f"  Category Filter: {selected_category}")
    print("-" * 70)

    if dry_run or stats_only:
        print("\n🔍 Evaluating FairFace validation shard (~10.9k images) breakdown...")
        val_parquet = hf_hub_download(
            repo_id=HF_FAIRFACE_REPO,
            filename=HF_FAIRFACE_SHARDS[0],
            repo_type="dataset",
        )
        table = pq.read_table(val_parquet)
        df_val = table.to_pandas()

        counts: Dict[str, int] = defaultdict(int)
        for _, r in df_val.iterrows():
            cat = map_lookmax_demographic(r["age"], r["gender"])
            if cat:
                if selected_category and selected_category.lower() not in (
                    "all",
                    cat.lower(),
                    cat.replace("face_", "").lower(),
                    cat.replace("face_", "").replace("_average_candid", "").lower(),
                ):
                    continue
                counts[cat] += 1

        print("\n📊 FairFace Single-Shard (10.9k images) Demographic Breakdown:")
        for cat, count in sorted(counts.items()):
            target_alloc = min(count, limit_per_category)
            print(f"    • {cat:<36}: {count:>5} available (allocating: {target_alloc:>4})")

        total_avail = sum(counts.values())
        print(f"\n  Total Available in Single Shard : {total_avail:,} adult in-the-wild faces")
        print(f"  Full 5-Shard Dataset Capacity   : ~{total_avail * 8:,} adult in-the-wild faces")

        if dry_run:
            print("\n[DRY RUN] Sample record mappings:")
            for i, r in df_val.head(6).iterrows():
                cat = map_lookmax_demographic(r["age"], r["gender"])
                age_str = AGE_CODE_NAMES.get(r["age"], str(r["age"]))
                gen_str = GENDER_CODE_NAMES.get(r["gender"], str(r["gender"]))
                race_str = RACE_CODE_NAMES.get(r["race"], str(r["race"]))
                print(f"  [{i:02d}] Age: {age_str:<6} | Gender: {gen_str:<6} | Race: {race_str:<16} → Category: {cat}")
            print("\nDry run complete. No files written.")
        return

    # Ingestion Run
    output_dir.mkdir(parents=True, exist_ok=True)
    METADATA_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Downloading and exporting {limit:,} in-the-wild everyday faces...")

    saved_count = 0
    skipped_count = 0
    failed_count = 0
    metadata_entries: List[str] = []

    stream = stream_fairface_records(
        max_records_needed=limit,
        limit_per_category=limit_per_category,
        selected_category=selected_category,
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        active_futures = []
        with tqdm(total=limit, desc="Saving FairFace Images", unit="img") as pbar:
            for rec in stream:
                f = executor.submit(save_image_record, rec, output_dir, resolution)
                active_futures.append((f, rec))

                if len(active_futures) >= workers * 4:
                    done_futures = [pair for pair in active_futures if pair[0].done()]
                    for future, r in done_futures:
                        success, status, _ = future.result()
                        if success:
                            if status == "already_exists":
                                skipped_count += 1
                            else:
                                saved_count += 1
                                entry = {
                                    "id": f"fairface_{r['id']}",
                                    "category": r["category"],
                                    "source": "FairFace",
                                    "age_bracket": r["age_bracket"],
                                    "gender": r["gender"],
                                    "race": r["race"],
                                    "resolution": f"{resolution}x{resolution}",
                                }
                                metadata_entries.append(json.dumps(entry) + "\n")
                        else:
                            failed_count += 1
                        pbar.update(1)
                        active_futures.remove((future, r))

            # Complete remaining
            for future, r in active_futures:
                success, status, _ = future.result()
                if success:
                    if status == "already_exists":
                        skipped_count += 1
                    else:
                        saved_count += 1
                        entry = {
                            "id": f"fairface_{r['id']}",
                            "category": r["category"],
                            "source": "FairFace",
                            "age_bracket": r["age_bracket"],
                            "gender": r["gender"],
                            "race": r["race"],
                            "resolution": f"{resolution}x{resolution}",
                        }
                        metadata_entries.append(json.dumps(entry) + "\n")
                else:
                    failed_count += 1
                pbar.update(1)

    # Append to metadata logs
    if metadata_entries:
        with open(FAIRFACE_LOG_FILE, "a", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(entry)

    print("\n" + "=" * 70)
    print(" FairFace Ingestion Complete!")
    print("=" * 70)
    print(f"  ✅ Newly Ingested   : {saved_count:,}")
    print(f"  ⏭️  Already Existing  : {skipped_count:,}")
    print(f"  ❌ Failed / Corrupt  : {failed_count:,}")
    print(f"  📁 Output Directory  : {output_dir}")
    print(f"  📝 Metadata Log      : {FAIRFACE_LOG_FILE}")
    print("=" * 70)
    print("\nNext Step:")
    print("  Run Phase 3 VLM Auto-Sorter on newly aggregated raw scrapes:")
    print("    python3 ML/pipeline/03_classify_and_sort.py --engine ollama")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LookMax — Ingest candid FairFace portraits (Tier 1 & 2) into 1_Raw_Scrapes/"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RAW_SCRAPES_DIR),
        help="Destination directory for raw scrapes (default: ML/data/1_Raw_Scrapes)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Total number of images to ingest across all demographic categories (default: 5000)",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=900,
        help="Maximum images per demographic category (default: 900)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by specific demographic category (e.g. men_u35, men_35_50, men_o50, women_u35, all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Target square image dimension in pixels (default: 512)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help="Number of concurrent worker threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate demographics and display distribution without writing files",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show dataset statistics and exit",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    run_pipeline(
        output_dir=output_dir,
        limit=args.limit,
        limit_per_category=args.limit_per_category,
        selected_category=args.category,
        resolution=args.resolution,
        workers=args.workers,
        dry_run=args.dry_run,
        stats_only=args.stats_only,
    )


if __name__ == "__main__":
    main()
