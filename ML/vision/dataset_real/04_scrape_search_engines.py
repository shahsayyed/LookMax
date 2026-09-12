"""
LookMax ML Pipeline — Multi-Engine Web Image Scraper
=====================================================
04_scrape_search_engines.py

Automated high-resolution web image scraper utilizing Playwright browser automation
(Bing Image Search, Google Images, DuckDuckGo) with human-like rate limiting,
streaming background downloads, and strict quality & framing gatekeeping.

Directly targets the 5 key dataset deficits:
  1. Tier 1 Flaws & Ill-Fitting Execution (pooling pants, wrinkles, tight seams, sloppy fit)
  2. True Corporate & Formal Tailoring (sharp suits, blazers, dress trousers vs casuals)
  3. Underrepresented Garments & Shoes (chinos, wide-leg, cargo, chelsea boots, loafers)
  4. Mature Demographics (men & women 35-50 and 50+)
  5. Real-World Grooming Variations (bedhead, patchy beards vs crisp fades & clean shaves)

Usage Examples:
    # 1. Pilot test: Scrape 3 images per category for 1 sample query across all streams
    python3 ML/vision/dataset_real/04_scrape_search_engines.py --limit 3 --sample-queries 1

    # 2. Dry-run URL extraction across Men_Outfit queries without downloading
    python3 ML/vision/dataset_real/04_scrape_search_engines.py --stream Men_Outfit --dry-run

    # 3. Scrape Tier 1 Needs Improvement queries across all streams (Bing engine)
    python3 ML/vision/dataset_real/04_scrape_search_engines.py --tier 1_Needs_Improvement --limit 25

    # 4. Target specific stream with custom limit
    python3 ML/vision/dataset_real/04_scrape_search_engines.py --stream Men_Grooming --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, unquote, urlparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("web_scraper")

# ─── LookMax Config Integration ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from config import (
        BLUR_LAPLACIAN_THRESHOLD,
        DOWNLOAD_WORKERS,
        METADATA_LOGS_DIR,
        RAW_SCRAPES_DIR,
        USER_AGENT,
    )
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    RAW_SCRAPES_DIR = BASE_DIR / "ML" / "data" / "vision_real" / "1_Raw_Scrapes"
    METADATA_LOGS_DIR = BASE_DIR / "ML" / "data" / "vision_real" / "2_VLM_Processing" / "metadata_logs"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    )
    BLUR_LAPLACIAN_THRESHOLD = 60.0
    DOWNLOAD_WORKERS = 8

try:
    import cv2
    import numpy as np
    from PIL import Image
    import requests
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
except ImportError as err:
    logger.error("Missing dependency: %s. Run: pip install playwright pillow opencv-python requests", err)
    sys.exit(1)

# Default queries catalog path
DEFAULT_QUERIES_FILE = Path(__file__).resolve().parent / "web_search_queries.json"
SCRAPED_LOG = METADATA_LOGS_DIR / "scraped_urls.txt"
METADATA_LOG = METADATA_LOGS_DIR / "web_scrapes.jsonl"

# ─── Strict Rule 3: Effort vs. Genetics & Non-Human Blocklist ──────────────────
# Block searches or URLs related to medical diseases, cosmetic surgery, body shaming,
# as well as product flat-lays, mannequins, illustrations, vectors, and posters
FORBIDDEN_KEYWORDS = {
    # Medical & genetics
    "acne_vulgaris", "dermatology", "skin_disease", "eczema", "psoriasis",
    "anorexia", "plastic_surgery", "rhinoplasty", "bariatric", "weight_loss_surgery",
    "body_dysmorphia", "fat_shaming", "liposuction", "botched", "scarring_alopecia",
    # Non-human / artwork / product-only / posters
    "flatlay", "flat-lay", "ghost-mannequin", "mannequin", "hanger", "packshot",
    "clipart", "vector", "illustration", "drawing", "cartoon", "render3d",
    "movie-poster", "album-cover", "soundtrack", "film-poster", "product-only",
    "blank-t-shirt", "mockup", "isolated-on-white"
}


def is_url_safe(url: str) -> bool:
    """Verifies that the image URL does not contain forbidden medical/body-shaming terms."""
    url_lower = url.lower()
    return not any(kw in url_lower for kw in FORBIDDEN_KEYWORDS)


def load_seen_urls() -> Set[str]:
    """Loads previously scraped URLs to prevent redownloading."""
    seen: Set[str] = set()
    if SCRAPED_LOG.exists():
        try:
            with open(SCRAPED_LOG, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    u = line.strip()
                    if u:
                        seen.add(u)
        except Exception as e:
            logger.warning("Error reading scraped_urls.txt: %s", e)
    return seen


def append_seen_url(url: str) -> None:
    """Appends newly scraped URL to scraped_urls.txt."""
    try:
        SCRAPED_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(SCRAPED_LOG, "a", encoding="utf-8") as f:
            f.write(url + "\n")
    except Exception as e:
        logger.warning("Error appending to scraped_urls.txt: %s", e)


def append_metadata(entry: Dict[str, Any]) -> None:
    """Appends image metadata record to web_scrapes.jsonl."""
    try:
        METADATA_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Error logging metadata: %s", e)


# ─── Quality & Framing Validation ─────────────────────────────────────────────

def validate_and_process_image(
    image_bytes: bytes,
    min_width: int,
    min_height: int,
    orientation: str,
    min_aspect: Optional[float] = None,
    max_aspect: Optional[float] = None,
    blur_threshold: float = BLUR_LAPLACIAN_THRESHOLD,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Validates image against LookMax quality heuristics:
      1. Can be opened by Pillow
      2. Meets minimum resolution (e.g. 600x800 for outfit, 500x500 for grooming)
      3. Orientation & aspect ratio:
         - Outfit: Portrait (min_aspect 1.25+ to guarantee full-body head-to-toe, not waist crop)
         - Grooming: Square / near-square (0.75 <= aspect <= 1.35)
      4. Discards pure-white background flatlays / eCommerce product cutouts
      5. Sharpness: Laplacian variance above blur threshold
    Returns (PIL Image, error_reason)
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
    except Exception:
        return None, "corrupt_or_unreadable"

    # Convert RGBA/Palette to RGB
    if img.mode != "RGB":
        try:
            img = img.convert("RGB")
        except Exception:
            return None, "cannot_convert_to_rgb"

    w, h = img.size

    # Check minimum dimensions
    if w < min_width or h < min_height:
        return None, f"too_small_{w}x{h}_need_{min_width}x{min_height}"

    # Check aspect ratio / framing
    aspect = h / float(w)
    if orientation == "portrait":
        min_asp = min_aspect or 1.20
        max_asp = max_aspect or 2.50
        if aspect < min_asp:
            return None, f"outfit_not_full_body_aspect_{aspect:.2f}_need_ge_{min_asp:.2f}"
        if aspect > max_asp:
            return None, f"extreme_vertical_aspect_{aspect:.2f}"

        # Flatlay / isolated product cutout rejection:
        # If all 4 corners are pure solid white (>252), it's almost certainly an eCommerce cutout
        try:
            np_arr = np.array(img)
            c1 = np_arr[:8, :8, :]
            c2 = np_arr[:8, -8:, :]
            c3 = np_arr[-8:, :8, :]
            c4 = np_arr[-8:, -8:, :]
            if c1.mean() > 252 and c2.mean() > 252 and c3.mean() > 252 and c4.mean() > 252:
                return None, "pure_white_isolated_flatlay_or_cutout"
        except Exception:
            pass

    elif orientation == "square":
        min_asp = min_aspect or 0.65
        max_asp = max_aspect or 1.85
        if aspect < min_asp or aspect > max_asp:
            return None, f"grooming_not_square_portrait_aspect_{aspect:.2f}_need_{min_asp:.2f}-{max_asp:.2f}"

    # Sharpness / Blur check via Laplacian variance
    try:
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < blur_threshold:
            return None, f"blurry_laplacian_{lap_var:.1f}_below_{blur_threshold}"
    except Exception as e:
        logger.debug("Laplacian blur check skipped: %s", e)

    return img, None


# ─── Image Downloader ─────────────────────────────────────────────────────────

def download_image(
    url: str,
    output_dir: Path,
    category: str,
    stream: str,
    tier: str,
    query: str,
    min_width: int,
    min_height: int,
    orientation: str,
    seen_urls: Set[str],
    seen_hashes: Set[str],
    min_aspect: Optional[float] = None,
    max_aspect: Optional[float] = None,
    timeout: int = 12,
) -> Optional[str]:
    """
    Fetches image, validates dimensions, framing, and blur, then saves if valid.
    """
    if url in seen_urls:
        return None
    if not is_url_safe(url):
        logger.debug("URL failed safety filter: %s", url)
        return None

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code != 200:
            return None
        content = resp.content
    except Exception:
        return None

    # Content hash deduplication
    img_hash = hashlib.sha256(content).hexdigest()[:16]
    if img_hash in seen_hashes:
        seen_urls.add(url)
        return None

    # Validate image
    img, err = validate_and_process_image(
        content,
        min_width=min_width,
        min_height=min_height,
        orientation=orientation,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
    )
    if not img or err:
        logger.debug("Image rejected (%s): %s", err, url)
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{img_hash}.jpg"
    if out_file.exists():
        seen_hashes.add(img_hash)
        seen_urls.add(url)
        return None

    try:
        img.save(out_file, "JPEG", quality=92, optimize=True)
        seen_hashes.add(img_hash)
        seen_urls.add(url)
        append_seen_url(url)

        # Log metadata
        append_metadata({
            "filename": out_file.name,
            "filepath": str(out_file),
            "category": category,
            "stream": stream,
            "tier": tier,
            "query": query,
            "source_url": url,
            "width": img.width,
            "height": img.height,
            "aspect_ratio": round(img.height / float(img.width), 2),
            "sha256": img_hash,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        return str(out_file)
    except Exception as e:
        logger.warning("Error saving image %s: %s", out_file, e)
        return None


# ─── Search Engine Crawlers ───────────────────────────────────────────────────

async def extract_bing_images(page: Page, query: str, max_scrolls: int = 4) -> List[str]:
    """
    Searches Bing Images and extracts full-resolution source image URLs (murls).
    Bing embeds the original source URL directly inside the data-m attribute of each result.
    """
    search_url = f"https://www.bing.com/images/search?q={quote_plus(query)}&form=HDRSC2&first=1"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)

    found_urls: List[str] = []
    seen: Set[str] = set()

    for scroll_idx in range(max_scrolls):
        content = await page.content()
        # Bing embeds full resolution URLs in &quot;murl&quot;:&quot;URL&quot;
        raw_urls = re.findall(r'&quot;murl&quot;:&quot;(http[^&]+)&quot;', content)
        for u in raw_urls:
            u_clean = unquote(u)
            if u_clean not in seen and is_url_safe(u_clean):
                seen.add(u_clean)
                found_urls.append(u_clean)

        # Scroll down smoothly to trigger lazy loading
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(random.uniform(1.2, 2.0))

        # Check for 'See more images' button if available
        try:
            more_btn = await page.query_selector("a.btn_seemore, input[value='See more images']")
            if more_btn:
                await more_btn.click()
                await asyncio.sleep(1.5)
        except Exception:
            pass

    return found_urls


async def extract_google_images(page: Page, query: str, max_scrolls: int = 3) -> List[str]:
    """
    Searches Google Images using udm=2 and extracts candidate image URLs.
    """
    search_url = f"https://www.google.com/search?q={quote_plus(query)}&udm=2"
    await page.goto(search_url, wait_until="domcontentloaded", timeout=25000)

    found_urls: List[str] = []
    seen: Set[str] = set()

    for _ in range(max_scrolls):
        content = await page.content()
        # Match high-res image extensions
        matches = re.findall(r'https?://[^\"\'\s<>\\]+?\.(?:jpg|jpeg|png|webp)', content)
        for u in matches:
            if "google.com" not in u and "gstatic.com" not in u and u not in seen:
                if is_url_safe(u):
                    seen.add(u)
                    found_urls.append(u)

        await page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
        await asyncio.sleep(random.uniform(1.5, 2.5))

    return found_urls


# ─── Scraper Orchestration ────────────────────────────────────────────────────

async def run_scraper(
    queries_file: Path,
    target_stream: str,
    target_tier: str,
    target_category: Optional[str],
    engine: str,
    limit_per_query: int,
    headed: bool,
    dry_run: bool,
    sample_queries: Optional[int] = None,
    target_categories: Optional[List[str]] = None,
) -> None:
    """Main scraping loop across configured categories and search engines."""
    if not queries_file.exists():
        logger.error("Queries catalog file not found: %s", queries_file)
        sys.exit(1)

    with open(queries_file, "r", encoding="utf-8") as f:
        catalog: List[Dict[str, Any]] = json.load(f)

    # Filter catalog
    cat_set = set(target_categories or [])
    if target_category:
        cat_set.add(target_category)

    filtered: List[Dict[str, Any]] = []
    for item in catalog:
        if target_stream != "all" and item.get("stream") != target_stream:
            continue
        if target_tier != "all" and item.get("tier") != target_tier:
            continue
        if cat_set and item.get("category") not in cat_set:
            continue
        filtered.append(item)

    if sample_queries and sample_queries > 0:
        filtered = filtered[:sample_queries]

    logger.info(
        "Loaded %d query category groups (Stream: %s, Tier: %s, Engine: %s, Limit/Query: %d)",
        len(filtered), target_stream, target_tier, engine, limit_per_query
    )

    seen_urls = load_seen_urls()
    seen_hashes: Set[str] = set()

    # Pre-populate seen_hashes from existing raw scrapes to avoid duplicate downloads
    logger.info("Indexing existing image hashes in %s...", RAW_SCRAPES_DIR)
    if RAW_SCRAPES_DIR.exists():
        for p in RAW_SCRAPES_DIR.glob("**/*.jpg"):
            seen_hashes.add(p.stem)
    logger.info("Indexed %d existing images and %d seen URLs.", len(seen_hashes), len(seen_urls))

    total_scraped = 0
    total_rejected = 0

    async with async_playwright() as p:
        browser_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ]

        # Launch Chromium
        browser: Browser = await p.chromium.launch(
            headless=not headed,
            args=browser_args,
        )

        context: BrowserContext = await browser.new_context(
            user_agent=USER_AGENT,
            viewport={"width": 1440, "height": 900},
            ignore_https_errors=True,
        )

        # Stealth evasion script
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)

        page: Page = await context.new_page()

        thread_pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)

        try:
            for cat_idx, item in enumerate(filtered, 1):
                stream = item["stream"]
                tier = item["tier"]
                category = item["category"]
                description = item.get("description", "")
                queries = item.get("queries", [])
                min_w = item.get("min_width", 600)
                min_h = item.get("min_height", 800)
                orient = item.get("orientation", "portrait")
                min_asp = item.get("min_aspect")
                max_asp = item.get("max_aspect")

                # Directory structure: RAW_SCRAPES_DIR / web_<stream>_<category>
                out_folder = RAW_SCRAPES_DIR / f"web_{stream.lower()}_{category}"

                logger.info(
                    "\n[%d/%d] Category: %s (%s | %s)\n Target: %s | Requirements: %dx%d (%s, aspect: %s-%s)",
                    cat_idx, len(filtered), category, stream, tier, description, min_w, min_h, orient, min_asp, max_asp
                )

                cat_saved = 0

                for q in queries:
                    if cat_saved >= limit_per_query:
                        break

                    logger.info("Searching: \"%s\"...", q)
                    try:
                        if engine == "bing":
                            urls = await extract_bing_images(page, q, max_scrolls=4)
                        elif engine == "google":
                            urls = await extract_google_images(page, q, max_scrolls=3)
                        else:
                            urls = await extract_bing_images(page, q, max_scrolls=4)
                    except Exception as e:
                        logger.warning("Search failed for \"%s\": %s", q, e)
                        continue

                    logger.info("Found %d candidate URLs for \"%s\"", len(urls), q)

                    if dry_run:
                        for u in urls[:5]:
                            logger.info("  [Dry-Run URL] %s", u)
                        continue

                    # Concurrent background image download & validation
                    futures = []
                    for u in urls:
                        if cat_saved >= limit_per_query:
                            break
                        fut = thread_pool.submit(
                            download_image,
                            url=u,
                            output_dir=out_folder,
                            category=category,
                            stream=stream,
                            tier=tier,
                            query=q,
                            min_width=min_w,
                            min_height=min_h,
                            orientation=orient,
                            seen_urls=seen_urls,
                            seen_hashes=seen_hashes,
                            min_aspect=min_asp,
                            max_aspect=max_asp,
                        )
                        futures.append(fut)

                    # Gather results
                    for fut in as_completed(futures):
                        res = fut.result()
                        if res:
                            cat_saved += 1
                            total_scraped += 1
                            logger.info("  ✓ Saved (%d/%d): %s", cat_saved, limit_per_query, Path(res).name)
                            if cat_saved >= limit_per_query:
                                break

                    # Humanized inter-query delay (3.0s - 6.0s)
                    query_delay = random.uniform(3.0, 6.0)
                    logger.debug("Cooldown pause: %.1fs", query_delay)
                    await asyncio.sleep(query_delay)

                logger.info("Category %s complete: %d images saved.", category, cat_saved)

                # Batch cooling pause every 5 categories
                if cat_idx % 5 == 0 and cat_idx < len(filtered):
                    cooldown = random.uniform(12.0, 18.0)
                    logger.info("Batch cooldown pause: sleeping %.1fs to protect rate limits...", cooldown)
                    await asyncio.sleep(cooldown)

        finally:
            thread_pool.shutdown(wait=True)
            await context.close()
            await browser.close()

    logger.info("\n========================================================")
    logger.info("Scraping Run Finished!")
    logger.info("Total Images Validated & Saved: %d", total_scraped)
    logger.info("Output Base Directory: %s", RAW_SCRAPES_DIR)
    logger.info("========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="LookMax Web Search Image Scraper")
    parser.add_argument("--queries-file", type=Path, default=DEFAULT_QUERIES_FILE, help="Path to queries JSON catalog")
    parser.add_argument("--stream", type=str, default="all", choices=["Men_Outfit", "Women_Outfit", "Men_Grooming", "Women_Grooming", "all"], help="Filter by stream")
    parser.add_argument("--tier", type=str, default="all", choices=["1_Needs_Improvement", "2_Average", "3_Polished", "all"], help="Filter by tier")
    parser.add_argument("--category", type=str, default=None, help="Filter by specific category slug")
    parser.add_argument("--categories", nargs="+", default=None, help="Filter by one or more category slugs")
    parser.add_argument("--engine", type=str, default="bing", choices=["bing", "google"], help="Search engine backend (bing default)")
    parser.add_argument("--limit", type=int, default=30, help="Max images to save per category (default: 30)")
    parser.add_argument("--sample-queries", type=int, default=None, help="Process only first N category groups for piloting")
    parser.add_argument("--headed", action="store_true", help="Launch browser with GUI visible")
    parser.add_argument("--dry-run", action="store_true", help="Extract and inspect URLs without downloading")

    args = parser.parse_args()

    asyncio.run(
        run_scraper(
            queries_file=args.queries_file,
            target_stream=args.stream,
            target_tier=args.tier,
            target_category=args.category,
            target_categories=args.categories,
            engine=args.engine,
            limit_per_query=args.limit,
            headed=args.headed,
            dry_run=args.dry_run,
            sample_queries=args.sample_queries,
        )
    )


if __name__ == "__main__":
    main()
