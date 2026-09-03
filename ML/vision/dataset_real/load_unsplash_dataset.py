"""
LookMax ML Pipeline — Unsplash Research Dataset Loader
======================================================
load_unsplash_dataset.py

Parses, filters, and downloads high-quality fashion, grooming, portrait, and posture
images from the local Unsplash Research Dataset (Lite or Full) directly into
ML/data/1_Raw_Scrapes/ for LookMax Phase 3 VLM processing.

Key Features:
  1. Offline Metadata Querying: Uses local TSV metadata (photos.tsv000 & keywords.tsv000)
     without requiring API keys or being restricted by API rate limits.
  2. Multi-Stage Semantic Qualification:
     • Strict negative filtering (removes animals, pets, infants/kids, landscapes, vehicles, food, interiors, flatlays)
     • Human verification (requires clear adult person/human presence)
     • LookMax domain routing (men's fashion, women's fashion, facial grooming, posture & fit)
     • Aspect ratio & resolution verification
  3. High-Speed Concurrent Downloads:
     • Unsplash dynamic CDN resizing (?w=1080&q=85&auto=format)
     • Multithreaded downloading with PIL image validation
     • Automatic deduplication with existing scrapes and scraped_urls.txt
     • Metadata logging to metadata_logs/unsplash_scrapes.jsonl

Usage:
    # 1. Inspect dataset matching statistics (Dry run)
    python3 ML/vision/dataset_real/load_unsplash_dataset.py --dry-run

    # 2. Download all qualified relevant images
    python3 ML/vision/dataset_real/load_unsplash_dataset.py

    # 3. Download with limit (e.g. 500 images total or 100 per category)
    python3 ML/vision/dataset_real/load_unsplash_dataset.py --limit 500
    python3 ML/vision/dataset_real/load_unsplash_dataset.py --limit-per-category 100

    # 4. Download only specific categories
    python3 ML/vision/dataset_real/load_unsplash_dataset.py --category men_fashion --limit 200
    python3 ML/vision/dataset_real/load_unsplash_dataset.py --category face_grooming --limit 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Pipeline Configuration
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import (
        RAW_SCRAPES_DIR,
        VLM_PROCESSING_DIR,
        METADATA_LOGS_DIR,
        ML_ROOT,
        USER_AGENT,
        DOWNLOAD_WORKERS,
    )
except ImportError:
    # Standalone fallback paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    ML_ROOT = SCRIPT_DIR.parent
    DATA_ROOT = ML_ROOT / "data"
    RAW_SCRAPES_DIR = DATA_ROOT / "1_Raw_Scrapes"
    VLM_PROCESSING_DIR = DATA_ROOT / "2_VLM_Processing"
    METADATA_LOGS_DIR = VLM_PROCESSING_DIR / "metadata_logs"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 LookMaxPipeline/1.0"
    )
    DOWNLOAD_WORKERS = 8

try:
    import pandas as pd
    import requests
    from PIL import Image
    from tqdm import tqdm
except ImportError:
    print("ERROR: Missing dependencies. Please run:")
    print("       pip install pandas requests pillow tqdm")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("LookMaxUnsplashLoader")

# ─── Dataset Paths ────────────────────────────────────────────────────────────
DEFAULT_DATASET_DIRS = [
    ML_ROOT / "unsplash-research-dataset-lite-latest",
    ML_ROOT / "unsplash-research-dataset",
    ML_ROOT.parent / "unsplash-research-dataset-lite-latest",
]

SCRAPED_LOG_FILE = METADATA_LOGS_DIR / "scraped_urls.txt"
UNSPLASH_LOG_FILE = METADATA_LOGS_DIR / "unsplash_scrapes.jsonl"

# ─── Semantic Filter Definitions ──────────────────────────────────────────────

# Strict exclusion terms — if present in description or keywords, reject
STRICT_EXCLUDE_TERMS = {
    # Animals / Fauna
    "animal", "dog", "cat", "puppy", "kitten", "canine", "feline", "pet", "pets",
    "mammal", "wildlife", "bird", "birds", "horse", "horses", "bear", "lion", "tiger",
    "wolf", "fox", "deer", "duck", "ducks", "sheep", "cow", "cattle", "elephant",
    "monkey", "ape", "fish", "shark", "whale", "dolphin", "insect", "spider",
    "butterfly", "reptile", "snake", "lizard", "amphibian", "frog", "fauna",
    
    # Infants / Children (LookMax models adult styling/grooming)
    "baby", "infant", "toddler", "newborn", "kid", "child", "children", "preschooler",
    "boyhood", "girlhood",

    # Scenery / Pure Nature / Flora (without human clothing focus)
    "flower", "flowers", "blossom", "petal", "rose", "tulip", "succulent", "cactus",
    "tree trunk", "leaf", "plant", "forest", "woodland", "mountain", "mountains",
    "valley", "canyon", "volcano", "glacier", "iceberg", "waterfall", "desert dune",
    "sand dune", "galaxy", "milky way", "nebula", "astronomy", "moon", "eclipse",
    "underwater",
    
    # Architecture / Interiors / Objects / Vehicles
    "building", "buildings", "skyscraper", "architecture", "interior design",
    "living room", "bedroom", "kitchen", "bathroom", "furniture", "table", "chair",
    "couch", "sofa", "desk", "lamp", "car", "automobile", "vehicle", "vehicles",
    "truck", "bus", "motorcycle", "bicycle", "airplane", "aircraft", "helicopter",
    "boat", "ship", "yacht", "train", "locomotive",
    
    # Food & Tech
    "food", "dish", "meal", "fruit", "vegetable", "coffee cup", "drink", "cocktail",
    "pizza", "burger", "cake", "dessert", "laptop", "computer", "keyboard",
    "monitor", "smartphone", "iphone", "macbook", "gadget",
    
    # Text / Abstract / Graphics / Still life
    "texture", "abstract", "wallpaper", "flatlay", "flat lay", "still life",
    "product mockup", "graphic", "illustration", "3d render", "render",
}

HUMAN_CORE_TERMS = {
    "person", "human", "people", "man", "woman", "girl", "boy", "guy", "male",
    "female", "model", "adult", "portrait", "individual", "lady", "gentleman",
}

MALE_TERMS = {
    "man", "male", "guy", "gentleman", "menswear", "men's fashion",
    "men fashion", "businessman", "groom", "him", "he", "bearded man",
    "beard", "mustache", "stubble", "tuxedo",
}

FEMALE_TERMS = {
    "woman", "female", "girl", "lady", "womenswear", "women's fashion",
    "women fashion", "businesswoman", "bride", "her", "she", "dress",
    "gown", "skirt", "blouse", "sundress", "evening dress", "cocktail dress",
    "wedding gown", "heels", "high heels",
}

FACE_GROOMING_TERMS = {
    "face", "portrait", "headshot", "facial hair", "beard", "mustache", "stubble",
    "hair", "hairstyle", "haircut", "skin", "eyes", "smile", "jawline", "eyebrows",
    "clean shaven", "sideburns", "pompadour", "fade haircut", "makeup", "lips",
    "facial expression", "close up portrait", "beauty portrait", "head shot",
}

FASHION_OUTFIT_TERMS = {
    "fashion", "clothing", "apparel", "outfit", "suit", "blazer", "tuxedo", "dress",
    "gown", "skirt", "blouse", "jacket", "coat", "overcoat", "t-shirt", "shirt",
    "jeans", "trousers", "pants", "streetwear", "style", "formal wear", "casual wear",
    "attire", "garment", "hoodie", "sweater", "cardigan", "trench coat",
    "leather jacket", "denim jacket", "lookbook", "runway", "sundress",
    "evening dress", "cocktail dress", "vest", "tie", "necktie", "bow tie",
    "smart casual", "business attire", "stylish", "fashion model", "denim",
}

POSTURE_FIT_TERMS = {
    "full body", "standing", "posture", "standing portrait", "pose", "model posing",
    "walking", "stride", "fit check", "runway walk", "standing tall", "full length",
}


def find_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """Locate the Unsplash dataset directory containing TSV files."""
    if custom_path:
        p = Path(custom_path).resolve()
        if p.exists() and p.is_dir():
            return p
        raise FileNotFoundError(f"Specified dataset directory not found: {custom_path}")

    for candidate in DEFAULT_DATASET_DIRS:
        if candidate.exists() and candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not automatically locate the Unsplash dataset folder. "
        "Please specify --dataset-dir <path>."
    )


def load_dataset_metadata(dataset_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Load photos.tsv000 and keywords.tsv000 from the dataset folder using high-speed indexing."""
    photos_path = None
    for fname in ["photos.tsv000", "photos.tsv", "photos.csv"]:
        candidate = dataset_dir / fname
        if candidate.exists():
            photos_path = candidate
            break

    if not photos_path:
        raise FileNotFoundError(f"photos.tsv000 not found in {dataset_dir}")

    keywords_path = None
    for fname in ["keywords.tsv000", "keywords.tsv", "keywords.csv"]:
        candidate = dataset_dir / fname
        if candidate.exists():
            keywords_path = candidate
            break

    logger.info("Loading photos metadata: %s", photos_path.name)
    photos_df = pd.read_csv(photos_path, sep="\t", low_memory=False)
    logger.info("Loaded %d photo records", len(photos_df))

    kw_map: Dict[str, Set[str]] = defaultdict(set)
    if keywords_path and keywords_path.exists():
        logger.info("Loading keywords metadata: %s", keywords_path.name)
        keywords_df = pd.read_csv(
            keywords_path,
            sep="\t",
            usecols=["photo_id", "keyword"],
            low_memory=False,
        )
        for pid, kw in keywords_df.dropna(subset=["keyword"]).itertuples(index=False):
            kw_map[str(pid)].add(str(kw).lower().strip())
        logger.info("Indexed %d photo keyword mappings", len(kw_map))

    return photos_df, kw_map


def classify_and_score_photo(
    row: pd.Series,
    kws: Set[str],
) -> Optional[Dict[str, Any]]:
    """
    Evaluates an Unsplash photo record for LookMax domain relevance.
    Returns structured metadata dict if qualified, or None if rejected.
    """
    desc = f"{str(row.get('photo_description', ''))} {str(row.get('ai_description', ''))}".lower()
    
    # ── Stage 1: Hard Negative Phrase Checks in Descriptions ──
    for neg_phrase in [
        "dog", "cat", "puppy", "kitten", "pet", "animal", "bird", "horse", "bear",
        "wildlife", "macro photography", "landscape photography of", "body of water near",
        "close up of flower", "silhouette of trees", "view of mountains", "scenic view",
        "interior of room", "aerial photography", "empty street", "still life",
        "baby", "infant", "toddler", "little girl", "little boy", "kid",
    ]:
        if neg_phrase in desc:
            return None

    # ── Stage 2: Hard Negative Keyword Matching ──
    if kws & STRICT_EXCLUDE_TERMS:
        neg_matches = kws & STRICT_EXCLUDE_TERMS
        pos_matches = kws & (MALE_TERMS | FEMALE_TERMS | FACE_GROOMING_TERMS | FASHION_OUTFIT_TERMS | POSTURE_FIT_TERMS)
        if len(neg_matches) >= 2 or len(pos_matches) == 0:
            return None

    # ── Stage 3: Human Subject Presence Verification ──
    has_human = (
        bool(kws & HUMAN_CORE_TERMS)
        or any(
            w in desc
            for w in [
                "person", "human", "man", "woman", "girl", "boy", "model",
                "wearing", "outfit", "portrait", "face", "standing",
            ]
        )
    )
    if not has_human:
        return None

    # ── Stage 4: Aspect Ratio & Dimension Verification ──
    ar = row.get("photo_aspect_ratio")
    try:
        ar_val = float(ar) if ar is not None and not pd.isna(ar) else 1.0
        if ar_val < 0.35 or ar_val > 2.6:
            return None
    except (ValueError, TypeError):
        pass

    width = row.get("photo_width")
    height = row.get("photo_height")
    try:
        if width and height and (int(width) < 480 or int(height) < 480):
            return None
    except (ValueError, TypeError):
        pass

    # ── Stage 5: Domain Feature Extraction & Scoring ──
    is_male = (
        bool(kws & MALE_TERMS)
        or any(w in desc for w in ["man", "men", "guy", "male", "gentleman", "boy", "groom", "businessman", "beard"])
    )
    is_female = (
        bool(kws & FEMALE_TERMS)
        or any(w in desc for w in ["woman", "women", "girl", "female", "lady", "bride", "businesswoman", "dress", "gown", "skirt"])
    )

    face_matches = (kws & FACE_GROOMING_TERMS)
    fashion_matches = (kws & FASHION_OUTFIT_TERMS)
    posture_matches = (kws & POSTURE_FIT_TERMS)

    has_face_desc = any(
        w in desc
        for w in [
            "portrait", "face", "headshot", "hair", "beard", "mustache",
            "eyes", "smile", "grooming", "jawline", "stubble",
        ]
    )
    has_fashion_desc = any(
        w in desc
        for w in [
            "suit", "dress", "outfit", "jacket", "coat", "blazer", "t-shirt",
            "shirt", "jeans", "wearing", "fashion", "streetwear", "style",
            "gown", "skirt", "attire", "tuxedo",
        ]
    )
    has_posture_desc = any(
        w in desc
        for w in ["standing", "full body", "posture", "pose", "walking", "posing", "runway"]
    )

    is_face = len(face_matches) > 0 or has_face_desc
    is_fashion = len(fashion_matches) > 0 or has_fashion_desc
    is_posture = len(posture_matches) > 0 or has_posture_desc

    if not (is_face or is_fashion or is_posture):
        return None

    # Calculate composite relevance score
    score = (
        len(face_matches) * 2
        + len(fashion_matches) * 2
        + len(posture_matches) * 1
        + (3 if has_fashion_desc else 0)
        + (3 if has_face_desc else 0)
        + (1 if is_male or is_female else 0)
    )

    # ── Stage 6: Category Routing ──
    if is_male and not is_female:
        if is_fashion and not is_face:
            category = "unsplash_men_fashion"
        elif is_face and not is_fashion:
            category = "unsplash_men_grooming"
        else:
            category = (
                "unsplash_men_fashion"
                if len(fashion_matches) >= len(face_matches)
                else "unsplash_men_grooming"
            )
    elif is_female and not is_male:
        if is_fashion and not is_face:
            category = "unsplash_women_fashion"
        elif is_face and not is_fashion:
            category = "unsplash_women_grooming"
        else:
            category = (
                "unsplash_women_fashion"
                if len(fashion_matches) >= len(face_matches)
                else "unsplash_women_grooming"
            )
    elif is_face:
        category = "unsplash_face_grooming"
    elif is_posture:
        category = "unsplash_posture_and_fit"
    elif is_fashion:
        category = "unsplash_general_fashion"
    else:
        category = "unsplash_general_style"

    photo_id = str(row["photo_id"])
    image_url = str(row["photo_image_url"])

    return {
        "photo_id": photo_id,
        "photo_url": str(row.get("photo_url", f"https://unsplash.com/photos/{photo_id}")),
        "image_url": image_url,
        "category": category,
        "relevance_score": score,
        "is_male": is_male,
        "is_female": is_female,
        "is_face": is_face,
        "is_fashion": is_fashion,
        "is_posture": is_posture,
        "description": desc[:120].strip(),
        "photographer": str(row.get("photographer_username", "")),
        "aspect_ratio": ar_val,
    }


def download_image_task(
    record: Dict[str, Any],
    output_dir: Path,
    resolution: int = 1080,
    timeout: int = 15,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Downloads and verifies a single Unsplash image with dynamic URL resizing.
    """
    photo_id = record["photo_id"]
    base_url = record["image_url"]
    category = record["category"]
    
    # Target file path
    target_folder = output_dir / category
    target_folder.mkdir(parents=True, exist_ok=True)
    target_path = target_folder / f"unsplash_{photo_id}.jpg"

    if target_path.exists() and target_path.stat().st_size > 5000:
        return True, "already_exists", record

    # Append dynamic resizing parameters
    sep = "&" if "?" in base_url else "?"
    download_url = f"{base_url}{sep}w={resolution}&q=85&auto=format&fit=crop"

    temp_path = target_folder / f".tmp_{photo_id}_{os.getpid()}.jpg"

    headers = {"User-Agent": USER_AGENT}

    for attempt in range(3):
        try:
            resp = requests.get(download_url, headers=headers, timeout=timeout, stream=True)
            if resp.status_code == 200:
                with open(temp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=32768):
                        if chunk:
                            f.write(chunk)

                # Validate image integrity with PIL
                with Image.open(temp_path) as img:
                    img.verify()

                temp_path.rename(target_path)
                return True, "downloaded", record
            elif resp.status_code in (403, 404):
                if temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                return False, f"http_{resp.status_code}", record
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)
            if attempt == 2:
                return False, str(e), record
            time.sleep(1.0 * (attempt + 1))

    if temp_path.exists():
        temp_path.unlink(missing_ok=True)
    return False, "max_retries_exceeded", record


def run_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    limit_per_category: Optional[int] = None,
    selected_category: Optional[str] = None,
    resolution: int = 1080,
    workers: int = DOWNLOAD_WORKERS,
    dry_run: bool = False,
    stats_only: bool = False,
) -> None:
    """Main execution function."""
    print("=" * 70)
    print(" LookMax — Unsplash Dataset Semantic Ingestion Pipeline")
    print("=" * 70)
    print(f"  Dataset Source : {dataset_dir}")
    print(f"  Output Dir     : {output_dir}")
    print(f"  Target Width   : {resolution}px (dynamic Unsplash CDN)")
    print(f"  Parallel Threads: {workers}")
    if limit:
        print(f"  Total Limit    : {limit} images")
    if limit_per_category:
        print(f"  Category Limit : {limit_per_category} images per category")
    if selected_category:
        print(f"  Category Filter: {selected_category}")
    print("-" * 70)

    # 1. Load TSVs
    photos_df, kw_map = load_dataset_metadata(dataset_dir)

    # 2. Filter & Classify
    print("\n🔍 Running multi-stage semantic qualification...")
    qualified_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_scanned = len(photos_df)

    for _, row in photos_df.iterrows():
        pid = str(row["photo_id"])
        kws = kw_map.get(pid, set())
        record = classify_and_score_photo(row, kws)
        if record:
            cat = record["category"]
            if selected_category and selected_category.lower() not in ("all", cat.lower(), cat.replace("unsplash_", "").lower()):
                continue
            qualified_by_category[cat].append(record)

    # Sort each category by relevance score descending
    for cat in qualified_by_category:
        qualified_by_category[cat].sort(key=lambda x: x["relevance_score"], reverse=True)

    total_qualified = sum(len(items) for items in qualified_by_category.values())

    print("\n📊 Semantic Qualification Results:")
    print(f"  Total Dataset Photos Scanned : {total_scanned:,}")
    print(f"  Total Qualified Human Photos : {total_qualified:,} ({total_qualified / max(1, total_scanned):.1%})")
    print("\n  Category Breakdown:")
    for cat, items in sorted(qualified_by_category.items(), key=lambda x: -len(x[1])):
        print(f"    • {cat:<30}: {len(items):>5} images (top score: {items[0]['relevance_score'] if items else 0})")

    if stats_only:
        return

    # Select target download list based on limits
    targets_to_download: List[Dict[str, Any]] = []

    for cat, items in qualified_by_category.items():
        cat_items = items
        if limit_per_category:
            cat_items = cat_items[:limit_per_category]
        targets_to_download.extend(cat_items)

    if limit and len(targets_to_download) > limit:
        # Balance proportionally across categories
        targets_to_download.sort(key=lambda x: x["relevance_score"], reverse=True)
        targets_to_download = targets_to_download[:limit]

    print(f"\n🎯 Selected for Ingestion: {len(targets_to_download):,} qualified photos")

    if dry_run:
        print("\n[DRY RUN] Top qualified photos across categories:")
        for i, item in enumerate(targets_to_download[:10], 1):
            print(f"  [{i:02d}] {item['photo_id']} | Category: {item['category']:<26} | Score: {item['relevance_score']:>2} | Desc: {item['description']}")
        print("\nDry run complete. No images were downloaded.")
        return

    # 3. Setup output directory & metadata logs
    output_dir.mkdir(parents=True, exist_ok=True)
    METADATA_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing scraped URLs to skip
    existing_urls: Set[str] = set()
    if SCRAPED_LOG_FILE.exists():
        with open(SCRAPED_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                u = line.strip()
                if u:
                    existing_urls.add(u)

    print(f"\n🚀 Downloading {len(targets_to_download):,} images with {workers} workers...")

    successful_count = 0
    skipped_count = 0
    failed_count = 0

    new_scraped_urls: List[str] = []
    metadata_entries: List[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_image_task, rec, output_dir, resolution): rec
            for rec in targets_to_download
        }

        with tqdm(total=len(futures), desc="Downloading", unit="img") as pbar:
            for future in as_completed(futures):
                success, status, rec = future.result()
                if success:
                    if status == "already_exists":
                        skipped_count += 1
                    else:
                        successful_count += 1
                        new_scraped_urls.append(rec["image_url"])
                        metadata_entries.append(json.dumps(rec) + "\n")
                else:
                    failed_count += 1
                pbar.update(1)

    # Append to logs
    if new_scraped_urls:
        with open(SCRAPED_LOG_FILE, "a", encoding="utf-8") as f:
            for url in new_scraped_urls:
                f.write(url + "\n")

    if metadata_entries:
        with open(UNSPLASH_LOG_FILE, "a", encoding="utf-8") as f:
            for entry in metadata_entries:
                f.write(entry)

    print("\n" + "=" * 70)
    print(" Ingestion Complete!")
    print("=" * 70)
    print(f"  ✅ Newly Downloaded : {successful_count:,}")
    print(f"  ⏭️  Already Existing  : {skipped_count:,}")
    print(f"  ❌ Failed / Skipped : {failed_count:,}")
    print(f"  📁 Output Directory  : {output_dir}")
    print(f"  📝 Metadata Log      : {UNSPLASH_LOG_FILE}")
    print("=" * 70)
    print("\nNext Step:")
    print("  Run Phase 3 VLM Auto-Sorter on downloaded raw scrapes:")
    print("    python3 ML/vision/dataset_real/03_classify_and_sort.py --engine ollama")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LookMax — Ingest and filter Unsplash Research Dataset into 1_Raw_Scrapes/"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to unzipped Unsplash dataset directory (default: ML/unsplash-research-dataset-lite-latest)",
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
        default=None,
        help="Maximum total images to download across all categories",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        default=None,
        help="Maximum images to download per category",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter by specific category (e.g. men_fashion, women_fashion, men_grooming, women_grooming, face, posture, all)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1080,
        help="Image width in pixels for dynamic CDN resize (default: 1080)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DOWNLOAD_WORKERS,
        help="Number of concurrent download threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate filters and display match statistics without downloading images",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show dataset statistics and exit",
    )

    args = parser.parse_args()

    try:
        dataset_dir = find_dataset_dir(args.dataset_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve()

    run_pipeline(
        dataset_dir=dataset_dir,
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
