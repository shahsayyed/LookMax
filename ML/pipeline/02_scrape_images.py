"""
LookMax ML Pipeline — Phase 2 (v3)
=====================================
02_scrape_images.py

Multi-source image scraper for fashion, posture, and style training data.

Sources (all public, no auth required):
  1. Unsplash API  — high-quality fashion & portrait images (free tier: 50 req/hr)
  2. Pexels API    — fashion, outfit, posture images (free, 200 req/hr)
  3. Pixabay API   — additional fashion imagery (free, unlimited)

All three APIs are free, return high-resolution images, and their licenses
permit usage in ML model training datasets.

Setup (one-time):
  Get free API keys (takes 2 minutes) at:
    Unsplash : https://unsplash.com/developers  → Create App
    Pexels   : https://www.pexels.com/api/      → Get API Key
    Pixabay  : https://pixabay.com/api/docs/    → Instant key on registration

  Set as environment variables or pass via --keys flag:
    export UNSPLASH_KEY=your_key
    export PEXELS_KEY=your_key
    export PIXABAY_KEY=your_key

Usage:
    python3 ML/pipeline/02_scrape_images.py
    python3 ML/pipeline/02_scrape_images.py --sources unsplash pexels --limit 200
    python3 ML/pipeline/02_scrape_images.py --dry-run
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_SCRAPES_DIR, IMAGE_EXTENSIONS, USER_AGENT,
    RATE_LIMIT_MIN_SEC, RATE_LIMIT_MAX_SEC, DOWNLOAD_WORKERS,
)

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("ERROR: Run: pip install -r ML/pipeline/requirements.txt")
    sys.exit(1)

# ─── Scraped URLs log ─────────────────────────────────────────────────────────
SCRAPED_LOG = (
    RAW_SCRAPES_DIR.parent.parent
    / "2_VLM_Processing" / "metadata_logs" / "scraped_urls.txt"
)

HEADERS = {"User-Agent": USER_AGENT}

# ─── Search queries per demographic category ─────────────────────────────────
FASHION_QUERIES = [
    # --- BUSINESS & FORMAL MEETING ---
    "man business meeting suit", "woman corporate office attire", 
    "man formal meeting clothes", "woman professional presentation outfit",
    "messy business suit man", "ill fitting corporate clothes", "business casual everyday",
    
    # --- WEDDING EVENT ---
    "man wedding guest suit", "woman wedding guest dress", 
    "formal wedding attire", "groomsmen suit outfit",
    "casual clothes at wedding", "awkward wedding guest outfit", "plain dress formal event",
    
    # --- NIGHT OUT & PARTY (Impress) ---
    "man night out club outfit", "woman party dress night out",
    "man date night style", "woman evening date outfit",
    "stylish party outfit impress", "glamorous night out style",
    "bad party outfit", "messy club clothes", "average bar outfit",
    
    # --- HOLIDAYS & VACATION ---
    "man summer holiday beach outfit", "woman vacation resort wear", 
    "man winter holiday coat", "woman tropical holiday style",
    "tourist holiday outfit casual", "awkward vacation clothes", "plain travel outfit",
    
    # --- CASUAL EVERYDAY ---
    "weekend casual outfit man", "relaxed everyday style woman",
    "lazy weekend clothes", "comfortable home outfit", "running errands outfit"
]

# ─── Utilities ────────────────────────────────────────────────────────────────
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_seen(path: Path) -> set:
    if not path.exists():
        return set()
    return set(l.strip() for l in path.read_text().splitlines() if l.strip())

def log_seen(path: Path, url: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(url + "\n")

def rate_limit():
    time.sleep(random.uniform(RATE_LIMIT_MIN_SEC, RATE_LIMIT_MAX_SEC))

def download_image(url: str, dest_dir: Path, seen_hashes: set,
                   dry_run: bool, filename_hint: str = "") -> str | None:
    if dry_run:
        return "dry_run"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=25, stream=True)
        if resp.status_code != 200:
            return None
        raw = resp.content
        if len(raw) < 10_000:          # skip tiny/corrupt images
            return None
        h = sha256_bytes(raw)
        if h in seen_hashes:
            return None
        seen_hashes.add(h)
        ext = ".jpg"
        out = dest_dir / f"{h[:16]}{ext}"
        out.write_bytes(raw)
        return out.name
    except Exception:
        return None


# ─── Source 1: Unsplash ───────────────────────────────────────────────────────
def scrape_unsplash(api_key: str, dest_dir: Path, seen_urls: set,
                    seen_hashes: set, limit: int, dry_run: bool) -> int:
    if not api_key:
        print("    ⚠  UNSPLASH_KEY not set — skipping Unsplash")
        return 0

    print(f"\n  📸 Unsplash API  →  {dest_dir.name}/")
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_urls: list[str] = []
    per_page = min(30, limit // len(FASHION_QUERIES) + 1)

    for query in FASHION_QUERIES:
        for page in range(1, 4):   # up to 3 pages per query
            params = {
                "query": query, "per_page": per_page,
                "page": page, "orientation": "portrait",
            }
            try:
                resp = requests.get(
                    "https://api.unsplash.com/search/photos",
                    params=params,
                    headers={**HEADERS, "Authorization": f"Client-ID {api_key}"},
                    timeout=15,
                )
                if resp.status_code == 403:
                    print("    ⚠  Unsplash rate limit hit — pausing 60s")
                    time.sleep(60)
                    continue
                if resp.status_code != 200:
                    break
                results = resp.json().get("results", [])
                if not results:
                    break
                for photo in results:
                    url = photo.get("urls", {}).get("regular") or photo.get("urls", {}).get("full")
                    if url and url not in seen_urls:
                        image_urls.append(url)
                        seen_urls.add(url)
            except Exception as e:
                print(f"    ⚠  Unsplash error: {e}")
                break
            rate_limit()
            if len(image_urls) >= limit:
                break
        if len(image_urls) >= limit:
            break

    image_urls = image_urls[:limit]
    print(f"    → {len(image_urls)} URLs queued")
    if not image_urls or dry_run:
        print(f"    [DRY-RUN] Would download {len(image_urls)} images" if dry_run else "")
        return 0

    downloaded = 0
    with tqdm(total=len(image_urls), unit="img", desc="    Unsplash") as pbar:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as exe:
            futs = {exe.submit(download_image, u, dest_dir, seen_hashes, dry_run): u
                    for u in image_urls}
            for fut in as_completed(futs):
                if fut.result():
                    downloaded += 1
                    log_seen(SCRAPED_LOG, futs[fut])
                pbar.update(1)

    print(f"    ✅ {downloaded} saved")
    return downloaded


# ─── Source 2: Pexels ────────────────────────────────────────────────────────
def scrape_pexels(api_key: str, dest_dir: Path, seen_urls: set,
                  seen_hashes: set, limit: int, dry_run: bool) -> int:
    if not api_key:
        print("    ⚠  PEXELS_KEY not set — skipping Pexels")
        return 0

    print(f"\n  📸 Pexels API  →  {dest_dir.name}/")
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_urls: list[str] = []
    per_page = 80

    for query in FASHION_QUERIES:
        for page in range(1, 4):
            try:
                resp = requests.get(
                    "https://api.pexels.com/v1/search",
                    params={"query": query, "per_page": per_page, "page": page},
                    headers={**HEADERS, "Authorization": api_key},
                    timeout=15,
                )
                if resp.status_code != 200:
                    break
                photos = resp.json().get("photos", [])
                if not photos:
                    break
                for p in photos:
                    url = p.get("src", {}).get("large") or p.get("src", {}).get("original")
                    if url and url not in seen_urls:
                        image_urls.append(url)
                        seen_urls.add(url)
            except Exception as e:
                print(f"    ⚠  Pexels error: {e}")
                break
            rate_limit()
            if len(image_urls) >= limit:
                break
        if len(image_urls) >= limit:
            break

    image_urls = image_urls[:limit]
    print(f"    → {len(image_urls)} URLs queued")
    if not image_urls or dry_run:
        print(f"    [DRY-RUN] Would download {len(image_urls)} images" if dry_run else "")
        return 0

    downloaded = 0
    with tqdm(total=len(image_urls), unit="img", desc="    Pexels") as pbar:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as exe:
            futs = {exe.submit(download_image, u, dest_dir, seen_hashes, dry_run): u
                    for u in image_urls}
            for fut in as_completed(futs):
                if fut.result():
                    downloaded += 1
                    log_seen(SCRAPED_LOG, futs[fut])
                pbar.update(1)

    print(f"    ✅ {downloaded} saved")
    return downloaded


# ─── Source 3: Pixabay ───────────────────────────────────────────────────────
def scrape_pixabay(api_key: str, dest_dir: Path, seen_urls: set,
                   seen_hashes: set, limit: int, dry_run: bool) -> int:
    if not api_key:
        print("    ⚠  PIXABAY_KEY not set — skipping Pixabay")
        return 0

    print(f"\n  📸 Pixabay API  →  {dest_dir.name}/")
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_urls: list[str] = []
    per_page = 200

    for query in FASHION_QUERIES:
        for page in range(1, 4):
            try:
                resp = requests.get(
                    "https://pixabay.com/api/",
                    params={
                        "key": api_key, "q": query, "per_page": per_page,
                        "page": page, "image_type": "photo",
                        "orientation": "vertical", "safesearch": "true",
                    },
                    headers=HEADERS, timeout=15,
                )
                if resp.status_code != 200:
                    break
                hits = resp.json().get("hits", [])
                if not hits:
                    break
                for h in hits:
                    url = h.get("largeImageURL") or h.get("webformatURL")
                    if url and url not in seen_urls:
                        image_urls.append(url)
                        seen_urls.add(url)
            except Exception as e:
                print(f"    ⚠  Pixabay error: {e}")
                break
            rate_limit()
            if len(image_urls) >= limit:
                break
        if len(image_urls) >= limit:
            break

    image_urls = image_urls[:limit]
    print(f"    → {len(image_urls)} URLs queued")
    if not image_urls or dry_run:
        print(f"    [DRY-RUN] Would download {len(image_urls)} images" if dry_run else "")
        return 0

    downloaded = 0
    with tqdm(total=len(image_urls), unit="img", desc="    Pixabay") as pbar:
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as exe:
            futs = {exe.submit(download_image, u, dest_dir, seen_hashes, dry_run): u
                    for u in image_urls}
            for fut in as_completed(futs):
                if fut.result():
                    downloaded += 1
                    log_seen(SCRAPED_LOG, futs[fut])
                pbar.update(1)

    print(f"    ✅ {downloaded} saved")
    return downloaded


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LookMax Phase 2 — Image Scraper")
    parser.add_argument("--limit",   type=int,    default=300,
                        help="Images per source (default: 300)")
    parser.add_argument("--sources", nargs="+",
                        choices=["unsplash", "pexels", "pixabay"],
                        default=["unsplash", "pexels", "pixabay"],
                        help="Which sources to scrape")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without downloading")
    args = parser.parse_args()

    # API keys from environment
    unsplash_key = os.environ.get("UNSPLASH_KEY", "")
    pexels_key   = os.environ.get("PEXELS_KEY", "")
    pixabay_key  = os.environ.get("PIXABAY_KEY", "")

    print(f"\n{'═'*62}")
    print(f"  LookMax ML Pipeline — Phase 2: Image Scraping")
    print(f"{'═'*62}")
    print(f"  Sources    : {', '.join(args.sources)}")
    print(f"  Limit/src  : {args.limit}")
    print(f"  Workers    : {DOWNLOAD_WORKERS} threads")
    print(f"  Dry-run    : {args.dry_run}")
    print(f"  Keys set   : Unsplash={'✓' if unsplash_key else '✗'}  "
          f"Pexels={'✓' if pexels_key else '✗'}  "
          f"Pixabay={'✓' if pixabay_key else '✗'}")

    if not any([unsplash_key, pexels_key, pixabay_key]):
        print(f"\n  ⚠  No API keys configured!")
        print(f"  Get free keys (2 min signup):")
        print(f"    Unsplash : https://unsplash.com/developers")
        print(f"    Pexels   : https://www.pexels.com/api/")
        print(f"    Pixabay  : https://pixabay.com/api/docs/")
        print(f"\n  Then set them:")
        print(f"    export UNSPLASH_KEY=your_key")
        print(f"    export PEXELS_KEY=your_key")
        print(f"    export PIXABAY_KEY=your_key")
        print(f"\n  Re-run: python3 ML/pipeline/02_scrape_images.py")
        sys.exit(0)

    seen_urls:   set = load_seen(SCRAPED_LOG)
    seen_hashes: set = set()
    print(f"  Skip log   : {len(seen_urls)} previously seen URLs")

    dest_unsplash = RAW_SCRAPES_DIR / "unsplash_fashion"
    dest_pexels   = RAW_SCRAPES_DIR / "pexels_fashion"
    dest_pixabay  = RAW_SCRAPES_DIR / "pixabay_fashion"

    total = 0
    if "unsplash" in args.sources:
        total += scrape_unsplash(unsplash_key, dest_unsplash, seen_urls,
                                 seen_hashes, args.limit, args.dry_run)
    if "pexels" in args.sources:
        total += scrape_pexels(pexels_key, dest_pexels, seen_urls,
                               seen_hashes, args.limit, args.dry_run)
    if "pixabay" in args.sources:
        total += scrape_pixabay(pixabay_key, dest_pixabay, seen_urls,
                                seen_hashes, args.limit, args.dry_run)

    all_imgs = sum(1 for ext in IMAGE_EXTENSIONS
                   for _ in RAW_SCRAPES_DIR.rglob(f"*{ext}"))

    print(f"\n{'═'*62}")
    print(f"  ✅ Session complete  — {total} new images downloaded")
    print(f"  📁 Total in 1_Raw_Scrapes/ : {all_imgs} images")
    print(f"\n  Next: python3 ML/pipeline/03_classify_and_sort.py")
    print(f"{'═'*62}\n")


if __name__ == "__main__":
    main()
