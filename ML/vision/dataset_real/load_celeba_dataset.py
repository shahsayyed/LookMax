"""
LookMax ML Pipeline — CelebA-HQ ($1024\\times1024$) Facial Ingestion Engine
==========================================================================
load_celeba_dataset.py

Ingests, semantically filters, and categorizes studio-grade $1024\\times1024$ portrait
images from the CelebA-HQ (High Quality) Dataset directly into ML/data/1_Raw_Scrapes/
for LookMax Phase 3 VLM processing and CoreML demographic/grooming training.

Key Features:
  1. Ultra-High Resolution ($1024\\times1024$):
     • Downloads pristine, studio-quality portrait images with fine skin, hair, and facial detail.
     • Replaces legacy downscaled $178\\times218$ thumbnails.
  2. Exact 1-to-1 Image ID & Attribute Alignment:
     • Extracts true numerical image IDs from internal parquet image paths (`{id}.jpg`),
       preventing alphabetical sorting mismatches across shards.
  3. Multi-Attribute Semantic Routing:
     • Classifies portraits into LookMax demographic and grooming subfolders:
       - Men Under 35 (Beard Styled, Styled Hair, Thinning/Balding, Clean Shaven, Formal)
       - Men 35–50 (Mature Beard, Balding, Mature Clean, Formal)
       - Men Over 50 (Distinguished Gray/Silver Hair, Gray Beard)
       - Women Under 35 (Polished Glam, Styled Hair, Natural Smile, Formal)
       - Women 35–50 & Over 50 (Polished Mature, Distinguished Silver Hair)
       - Feature Focus (Eyeglasses, Formal Necktie, High Cheekbones/Jawline)
  4. Quality & Balancing:
     • Drops blurry/low-quality entries.
     • Caps quotas per category for a balanced, unbiased training distribution.
     • Multithreaded validation and export with metadata logging.

Usage:
    # 1. Preview qualification and balance across categories (Dry Run)
    python3 ML/vision/dataset_real/load_celeba_dataset.py --dry-run

    # 2. Ingest 5,500 balanced $1024\times1024$ images (~300-400 per category)
    python3 ML/vision/dataset_real/load_celeba_dataset.py --limit 5500

    # 3. Ingest custom number of images (e.g. 6,000 total or 300 per category)
    python3 ML/vision/dataset_real/load_celeba_dataset.py --limit 6000 --limit-per-category 300

    # 4. Filter by specific demographic/category
    python3 ML/vision/dataset_real/load_celeba_dataset.py --category men_u35 --limit 1500
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
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
logger = logging.getLogger("LookMaxCelebAHQLoader")

# HuggingFace CelebA-HQ Repositories
HF_HQ_IMAGES_REPO = "bitmind/celeb-a-hq"
HF_HQ_ANNOTATIONS_REPO = "bitmind/celeb-a-hq___annotations"

HF_HQ_SHARDS = [
    "data/train-00000-of-00006.parquet",
    "data/train-00001-of-00006.parquet",
    "data/train-00002-of-00006.parquet",
    "data/train-00003-of-00006.parquet",
    "data/train-00004-of-00006.parquet",
    "data/train-00005-of-00006.parquet",
]

SCRAPED_LOG_FILE = METADATA_LOGS_DIR / "scraped_urls.txt"
CELEBA_LOG_FILE = METADATA_LOGS_DIR / "celeba_scrapes.jsonl"


def classify_description(text: str) -> Optional[str]:
    """
    Evaluates a CelebA-HQ natural language annotation and routes into
    LookMax demographic and facial grooming categories.
    """
    t = text.lower()
    if any(neg in t for neg in ["blurry", "distorted", "grainy", "low quality", "out of focus"]):
        return None

    is_male = bool(re.search(r"\b(man|men|male|guy|gentleman|boy|actor|businessman|groom|him|he)\b", t))
    is_female = bool(re.search(r"\b(woman|women|female|girl|lady|actress|businesswoman|bride|pageant|her|she)\b", t))

    is_young = bool(re.search(r"\b(young|youth|teen|20s|twenty|early twenties|mid-twenties)\b", t))
    is_mature = bool(re.search(r"\b(middle-aged|mature|older|senior|elderly|gray hair|grey hair|silver hair|white hair|white beard|gray beard|grey beard|40s|50s|60s|wrinkles)\b", t))
    is_gray = bool(re.search(r"\b(gray hair|grey hair|silver hair|white hair|white beard|gray beard|grey beard)\b", t))

    has_beard = bool(re.search(r"\b(beard|bearded|goatee|mustache|stubble|facial hair|5 o'clock shadow|sideburns)\b", t))
    has_bald = bool(re.search(r"\b(bald|balding|receding hairline|shaved head|thinning hair)\b", t))
    has_styled_hair = bool(re.search(r"\b(styled hair|wavy hair|curly hair|blonde hair|blond hair|brown hair|black hair|brunette|bun|ponytail|bangs|short hair|long hair|bob haircut)\b", t))

    has_makeup = bool(re.search(r"\b(makeup|lipstick|glam|eyeliner|eyeshadow|gloss|mascara)\b", t))
    has_smile = bool(re.search(r"\b(smile|smiling|smiles|cheerful|laughing|grin)\b", t))
    has_glasses = bool(re.search(r"\b(glasses|eyeglasses|sunglasses|spectacles)\b", t))
    has_formal = bool(re.search(r"\b(suit|blazer|tuxedo|formal|tie|necktie|bow tie|gown|evening dress|tux)\b", t))

    # ── Category 1: Men Demographics & Grooming ──
    if is_male and not is_female:
        if is_gray or "senior" in t or "elderly" in t:
            return "face_men_o50_distinguished"
        elif is_mature:
            if has_beard:
                return "face_men_35_50_beard"
            elif has_bald:
                return "face_men_35_50_balding"
            elif has_formal:
                return "face_men_35_50_formal"
            else:
                return "face_men_35_50_portrait"
        else: # Young / Baseline Men (Under 35)
            if has_beard:
                return "face_men_u35_beard_styled"
            elif has_bald:
                return "face_men_u35_hair_thinning"
            elif has_styled_hair and has_smile:
                return "face_men_u35_hair_styled"
            elif has_formal:
                return "face_men_u35_formal"
            elif has_smile:
                return "face_men_u35_smile_clean"
            else:
                return "face_men_u35_portrait_neutral"

    # ── Category 2: Women Demographics & Grooming ──
    elif is_female and not is_male:
        if is_gray or "senior" in t or "elderly" in t:
            return "face_women_o50_distinguished"
        elif is_mature:
            if has_formal or has_makeup:
                return "face_women_35_50_polished"
            else:
                return "face_women_35_50_portrait"
        else: # Young / Baseline Women (Under 35)
            if has_makeup and has_styled_hair:
                return "face_women_u35_polished_glam"
            elif has_styled_hair and has_formal:
                return "face_women_u35_formal"
            elif has_styled_hair:
                return "face_women_u35_hair_styled"
            elif has_smile:
                return "face_women_u35_smile_natural"
            else:
                return "face_women_u35_portrait_neutral"

    # ── Category 3: General Styling Portraits ──
    if has_glasses:
        return "face_portrait_eyeglasses"
    if has_formal:
        return "face_portrait_formal"
    return "face_portrait_general"


def load_celebahq_annotations() -> pd.DataFrame:
    """Downloads and loads the 30,000 CelebA-HQ natural language annotations."""
    logger.info("Loading CelebA-HQ annotations from %s...", HF_HQ_ANNOTATIONS_REPO)
    ann_path = hf_hub_download(
        repo_id=HF_HQ_ANNOTATIONS_REPO,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
    )
    df_ann = pq.read_table(ann_path).to_pandas()
    logger.info("Loaded %d CelebA-HQ annotation records", len(df_ann))
    return df_ann


def stream_celebahq_records(
    df_ann: pd.DataFrame,
    max_records_needed: int,
    limit_per_category: int,
    selected_category: Optional[str] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streams $1024x1024$ CelebA-HQ images shard-by-shard, mapping the true
    numerical image ID directly from `img_data['path']` (e.g. `1444.jpg` -> `1444`).
    """
    id_to_desc: Dict[int, str] = {int(r["id"]): str(r["description"]) for _, r in df_ann.iterrows()}
    id_to_cat: Dict[int, str] = {}

    for img_id, desc in id_to_desc.items():
        cat = classify_description(desc)
        if cat:
            if selected_category and selected_category.lower() not in (
                "all",
                cat.lower(),
                cat.replace("face_", "").lower(),
            ):
                continue
            id_to_cat[img_id] = cat

    category_counts: Dict[str, int] = defaultdict(int)
    total_yielded = 0

    for shard_idx, shard_file in enumerate(HF_HQ_SHARDS):
        if total_yielded >= max_records_needed:
            break

        logger.info(
            "Fetching CelebA-HQ $1024\\times1024$ shard [%d/%d]: %s",
            shard_idx + 1,
            len(HF_HQ_SHARDS),
            shard_file,
        )

        try:
            local_parquet = hf_hub_download(
                repo_id=HF_HQ_IMAGES_REPO,
                filename=shard_file,
                repo_type="dataset",
            )
            table = pq.read_table(local_parquet)
        except Exception as e:
            logger.error("Failed to load shard %s: %s", shard_file, e)
            continue

        images = table["image"]
        num_rows = len(table)

        for offset in range(num_rows):
            img_data = images[offset].as_py()
            if not isinstance(img_data, dict):
                continue

            path_str = img_data.get("path", "")
            img_bytes = img_data.get("bytes")
            if not path_str or not img_bytes:
                continue

            try:
                real_id = int(Path(path_str).stem)
            except ValueError:
                continue

            if real_id not in id_to_cat:
                continue

            cat = id_to_cat[real_id]
            if category_counts[cat] >= limit_per_category:
                continue

            category_counts[cat] += 1
            total_yielded += 1

            yield {
                "id": f"{real_id:05d}",
                "category": cat,
                "bytes": img_bytes,
                "description": id_to_desc[real_id],
            }

            if total_yielded >= max_records_needed:
                return


def save_image_record(
    record: Dict[str, Any],
    output_dir: Path,
    target_resolution: int = 1024,
) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates, optionally resizes, and writes high-res image to disk."""
    rec_id = record["id"]
    category = record["category"]
    img_bytes = record["bytes"]

    target_folder = output_dir / category
    target_folder.mkdir(parents=True, exist_ok=True)
    target_path = target_folder / f"celebahq_{rec_id}.jpg"

    if target_path.exists() and target_path.stat().st_size > 5000:
        return True, "already_exists", record

    try:
        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            if target_resolution != 1024 and (img.width != target_resolution or img.height != target_resolution):
                img = img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)
            img.save(target_path, "JPEG", quality=95)

        return True, "saved", record
    except Exception as e:
        return False, str(e), record


def run_pipeline(
    output_dir: Path = RAW_SCRAPES_DIR,
    limit: int = 5500,
    limit_per_category: int = 350,
    selected_category: Optional[str] = None,
    resolution: int = 1024,
    workers: int = DOWNLOAD_WORKERS,
    dry_run: bool = False,
    stats_only: bool = False,
) -> None:
    """Main execution function for CelebA-HQ ingestion."""
    print("=" * 70)
    print(" LookMax — CelebA-HQ ($1024\\times1024$) Facial Ingestion Pipeline")
    print("=" * 70)
    print(f"  Target Limit   : {limit:,} images total (max {limit_per_category} per category)")
    print(f"  Destination    : {output_dir}")
    print(f"  Image Quality  : {resolution}x{resolution} px Studio Portraits")
    if selected_category:
        print(f"  Category Filter: {selected_category}")
    print("-" * 70)

    # 1. Load annotations
    df_ann = load_celebahq_annotations()

    # 2. Categorize all 30k records
    print("\n🔍 Evaluating CelebA-HQ 30k records for demographic & grooming balance...")
    distribution: Dict[str, int] = defaultdict(int)
    for _, row in df_ann.iterrows():
        cat = classify_description(str(row["description"]))
        if cat:
            if selected_category and selected_category.lower() not in (
                "all",
                cat.lower(),
                cat.replace("face_", "").lower(),
            ):
                continue
            distribution[cat] += 1

    print("\n📊 CelebA-HQ Full Dataset Potential:")
    for cat, count in sorted(distribution.items(), key=lambda x: -x[1]):
        target_allocation = min(count, limit_per_category)
        print(f"    • {cat:<36}: {count:>5} available (allocating: {target_allocation:>4})")

    if stats_only or dry_run:
        if dry_run:
            print("\n[DRY RUN] Sample classifications:")
            for i, row in df_ann.head(10).iterrows():
                cat = classify_description(str(row["description"]))
                print(f"  [{i:02d}] ID: {row['id']} | Category: {str(cat):<30} | Desc: {str(row['description'])[:60]}...")
            print("\nDry run complete. No files written.")
        return

    # 3. Stream & Download
    output_dir.mkdir(parents=True, exist_ok=True)
    METADATA_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 Downloading and exporting up to {limit:,} $1024\\times1024$ images...")

    saved_count = 0
    skipped_count = 0
    failed_count = 0
    metadata_entries: List[str] = []

    stream = stream_celebahq_records(
        df_ann=df_ann,
        max_records_needed=limit,
        limit_per_category=limit_per_category,
        selected_category=selected_category,
    )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        active_futures = []
        with tqdm(total=limit, desc="Processing $1024\\times1024$ Images", unit="img") as pbar:
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
                                    "id": f"celebahq_{r['id']}",
                                    "category": r["category"],
                                    "source": "CelebA-HQ",
                                    "resolution": f"{resolution}x{resolution}",
                                    "description": r["description"],
                                }
                                metadata_entries.append(json.dumps(entry) + "\n")
                        else:
                            failed_count += 1
                        pbar.update(1)
                        active_futures.remove((future, r))

            # Complete remaining futures
            for future, r in active_futures:
                success, status, _ = future.result()
                if success:
                    if status == "already_exists":
                        skipped_count += 1
                    else:
                        saved_count += 1
                        entry = {
                            "id": f"celebahq_{r['id']}",
                            "category": r["category"],
                            "source": "CelebA-HQ",
                            "resolution": f"{resolution}x{resolution}",
                            "description": r["description"],
                        }
                        metadata_entries.append(json.dumps(entry) + "\n")
                else:
                    failed_count += 1
                pbar.update(1)

    # Append to metadata logs
    if metadata_entries:
        with open(CELEBA_LOG_FILE, "a", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(entry)

    print("\n" + "=" * 70)
    print(" Ingestion Complete!")
    print("=" * 70)
    print(f"  ✅ Newly Ingested   : {saved_count:,}")
    print(f"  ⏭️  Already Existing  : {skipped_count:,}")
    print(f"  ❌ Failed / Corrupt  : {failed_count:,}")
    print(f"  📁 Output Directory  : {output_dir}")
    print(f"  📝 Metadata Log      : {CELEBA_LOG_FILE}")
    print("=" * 70)
    print("\nNext Step:")
    print("  Run Phase 3 VLM Auto-Sorter on high-res raw scrapes:")
    print("    python3 ML/vision/dataset_real/03_classify_and_sort.py --engine ollama")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LookMax — Ingest and balance CelebA-HQ ($1024x1024$) portraits into 1_Raw_Scrapes/"
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
        default=5500,
        help="Total number of images to ingest across all categories (default: 5500)",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=350,
        help="Maximum images per demographic/grooming category (default: 350)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by specific category or demographic prefix (e.g. men_u35, women_35_50, beard, all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Target square image dimension in pixels (default: 1024)",
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
        help="Evaluate attributes and display category distribution without writing files",
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
