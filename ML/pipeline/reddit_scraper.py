"""
LookMax ML Pipeline — Reddit Playwright JSON Image Scraper
============================================================
Scrapes high-resolution image URLs from Reddit's .json endpoints using an
authenticated Chromium persistent browser session via Playwright.

Key Capabilities:
  1. Persistent Authentication: Uses Chromium user_data_dir so you only log in once.
  2. Anti-Rate-Limit Protection:
     - Safe randomized delays (3.5s–7.0s per request)
     - Batch cooldowns (pause 15–25s every 10 requests) to keep sliding windows clear
     - Inter-category pauses (6–10s)
     - Intelligent exponential backoff (30s -> 60s -> 120s) on HTTP 429 or challenge pages
  3. Search & Subreddit Endpoints: Crawls standard feeds (hot/top/new) AND keyword search queries
     (e.g., r/malefashionadvice/search.json?q=tailored+suit&restrict_sr=1&sort=top).
  4. Gallery & Preview Extraction: Handles direct images, multi-image galleries (media_metadata),
     and high-res preview fallbacks, decoding HTML entities (&amp; -> &).
  5. 110+ Pre-Configured Curated Queries: Out-of-the-box catalog in reddit_queries.json covering all
     6 demographic brackets, 3 aesthetic tiers, and posture variations.
  6. Incremental State & Downloader: Saves JSON progress after every category and optionally
     downloads images directly into LookMax's 1_Raw_Scrapes/ directory.

Usage Examples:
  # 1. First-time setup: interactive login
  python3 ML/pipeline/reddit_scraper.py --login

  # 2. Verify stored session
  python3 ML/pipeline/reddit_scraper.py --check-session

  # 3. Scrape 110+ diverse queries catalog with anti-rate-limit protection
  python3 ML/pipeline/reddit_scraper.py --queries-file ML/pipeline/reddit_queries.json --limit 50

  # 4. Scrape with custom rate-limiting and direct download
  python3 ML/pipeline/reddit_scraper.py --download --limit 100 --delay-min 4.0 --delay-max 8.0 --batch-size 8

  # 5. Search a specific term across a subreddit
  python3 ML/pipeline/reddit_scraper.py --subreddit malefashionadvice --query "tailored suit" --limit 100
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import os
import platform
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from urllib.parse import parse_qs, quote_plus, urlencode, urlparse, urlunparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("reddit_scraper")

# ─── Optional LookMax Pipeline Config Integration ────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        DOWNLOAD_WORKERS,
        IMAGE_EXTENSIONS,
        RAW_SCRAPES_DIR,
        REDDIT_BATCH_COOLDOWN_SEC,
        REDDIT_BATCH_SIZE,
        REDDIT_CATEGORY_COOLDOWN_SEC,
        REDDIT_DELAY_MAX_SEC,
        REDDIT_DELAY_MIN_SEC,
        REDDIT_PROFILE_DIR,
        REDDIT_QUERIES_FILE,
        REDDIT_SCRAPE_OUTPUT_JSON,
        REDDIT_SOURCES,
        USER_AGENT,
    )
except ImportError:
    # Standalone defaults if running outside LookMax environment
    PIPELINE_DIR = Path(__file__).resolve().parent
    RAW_SCRAPES_DIR = PIPELINE_DIR.parent / "data" / "1_Raw_Scrapes"
    REDDIT_PROFILE_DIR = PIPELINE_DIR / "reddit_profile"
    REDDIT_QUERIES_FILE = PIPELINE_DIR / "reddit_queries.json"
    REDDIT_SCRAPE_OUTPUT_JSON = PIPELINE_DIR / "reddit_images.json"
    USER_AGENT = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )
    REDDIT_DELAY_MIN_SEC = 3.5
    REDDIT_DELAY_MAX_SEC = 7.0
    REDDIT_BATCH_SIZE = 10
    REDDIT_BATCH_COOLDOWN_SEC = 20.0
    REDDIT_CATEGORY_COOLDOWN_SEC = 8.0
    DOWNLOAD_WORKERS = 8
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    REDDIT_SOURCES = [
        {"subreddit": "OUTFITS", "folder": "reddit_outfits"},
        {"subreddit": "malefashionadvice", "folder": "reddit_malefashionadvice"},
        {"subreddit": "femalefashionadvice", "folder": "reddit_femalefashionadvice"},
        {"subreddit": "streetwear", "folder": "reddit_streetwear"},
        {"subreddit": "Posture", "folder": "reddit_posture"},
        {"subreddit": "mensfashion", "folder": "reddit_mensfashion"},
        {"subreddit": "femalefashion", "folder": "reddit_femalefashion"},
    ]

# ─── Playwright Import Check ──────────────────────────────────────────────────
try:
    from playwright.sync_api import (
        BrowserContext,
        Page,
        Playwright,
        sync_playwright,
    )
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False


def _ensure_playwright() -> None:
    """Ensure playwright package is available before executing browser flows."""
    if not HAS_PLAYWRIGHT:
        logger.error(
            "Playwright is not installed. Please install it using:\n"
            "    pip install playwright\n"
            "    playwright install chromium"
        )
        raise RuntimeError("playwright is required for reddit_scraper")


def _clean_stale_profile_locks(profile_dir: Path) -> None:
    """Clean leftover Chromium lock files/symlinks that cause launch failures after crashes."""
    for lock_name in ("SingletonLock", "SingletonCookie", "SingletonSocket", "RunningChromeVersion"):
        lock_file = profile_dir / lock_name
        try:
            if lock_file.is_symlink() or lock_file.exists():
                lock_file.unlink(missing_ok=True)
        except OSError:
            pass


def get_system_chrome_profiles() -> List[Dict[str, str]]:
    """Discover existing Google Chrome profiles on the host system."""
    if platform.system() == "Darwin":
        chrome_root = Path.home() / "Library/Application Support/Google/Chrome"
    elif platform.system() == "Windows":
        chrome_root = Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data"
    else:
        chrome_root = Path.home() / ".config/google-chrome"

    local_state_file = chrome_root / "Local State"
    results = []
    if not local_state_file.exists():
        return results

    try:
        with open(local_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        info_cache = data.get("profile", {}).get("info_cache", {})
        for folder_name, info in info_cache.items():
            results.append({
                "folder": folder_name,
                "name": info.get("name", "Unknown"),
                "email": info.get("user_name", ""),
            })
    except Exception as e:
        logger.debug("Failed reading Chrome profiles: %s", e)
    return results


def print_available_chrome_profiles() -> None:
    """Print all discovered Google Chrome profiles with their folder and display names."""
    profiles = get_system_chrome_profiles()
    if not profiles:
        print("\nNo Google Chrome profiles detected in standard Chrome directory.\n")
        return

    print("\n" + "═" * 78)
    print("  Available Google Chrome Profiles on your System")
    print("═" * 78)
    print(f"  {'Folder Name (--chrome-profile)':<32} | {'Display Name':<20} | Email")
    print("─" * 78)
    for p in profiles:
        email = f"({p['email']})" if p["email"] else ""
        print(f"  {p['folder']:<32} | {p['name']:<20} | {email}")
    print("═" * 78)
    print("  Usage Note:")
    print("    1. Completely quit Google Chrome (Cmd + Q) before running with --chrome-profile.")
    print('    2. Example: python ML/pipeline/reddit_scraper.py --chrome-profile "Profile 1"')
    print("═" * 78 + "\n")


def resolve_chrome_profile_folder(query: str) -> str:
    """Resolve a Chrome profile folder name from either folder name, display name, or email."""
    profiles = get_system_chrome_profiles()
    query_lower = query.strip().lower()
    for p in profiles:
        if p["folder"].lower() == query_lower:
            return p["folder"]
    for p in profiles:
        if query_lower == p["name"].lower() or query_lower == p["email"].lower():
            return p["folder"]
    for p in profiles:
        if query_lower in p["name"].lower() or (p["email"] and query_lower in p["email"].lower()):
            return p["folder"]
    return query


# ─── Browser Context & Authentication Helpers ─────────────────────────────────

def create_persistent_context(
    playwright: Playwright,
    user_data_dir: Union[Path, str],
    headless: bool = True,
    user_agent: Optional[str] = None,
    channel: Optional[str] = None,
    chrome_profile: Optional[str] = None,
) -> BrowserContext:
    """
    Launch a persistent Chromium context storing session cookies in user_data_dir.
    Automatically tries installed Google Chrome on macOS to avoid Apple Silicon ImageIO crashes.

    Args:
        playwright: Active Playwright instance.
        user_data_dir: Path to directory for profile storage.
        headless: Whether to run in headless mode.
        user_agent: Optional custom User-Agent string.
        channel: Optional browser channel ('chrome', 'msedge', or None).
        chrome_profile: Optional Chrome profile name (e.g. 'Default', 'Profile 1', or display name).

    Returns:
        BrowserContext: Configured persistent browser context.
    """
    _ensure_playwright()

    if chrome_profile:
        resolved_folder = resolve_chrome_profile_folder(chrome_profile)
        if platform.system() == "Darwin":
            profile_path = Path.home() / "Library/Application Support/Google/Chrome"
        elif platform.system() == "Windows":
            profile_path = Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/User Data"
        else:
            profile_path = Path.home() / ".config/google-chrome"
        channel = "chrome"
    else:
        resolved_folder = None
        profile_path = Path(user_data_dir).resolve()
        profile_path.mkdir(parents=True, exist_ok=True)
        _clean_stale_profile_locks(profile_path)

    ua = user_agent or USER_AGENT
    args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage",
    ]
    if resolved_folder:
        args.append(f"--profile-directory={resolved_folder}")

    channels_to_try = [channel] if channel else ["chrome", None]
    last_err: Optional[Exception] = None

    for ch in channels_to_try:
        try:
            kwargs = {
                "user_data_dir": str(profile_path),
                "headless": headless,
                "user_agent": ua,
                "args": args,
                "viewport": {"width": 1280, "height": 800},
                "ignore_https_errors": True,
            }
            if ch:
                kwargs["channel"] = ch
            return playwright.chromium.launch_persistent_context(**kwargs)
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "SingletonLock" in err_str or "ProcessSingleton" in err_str:
                logger.error(
                    "\n\n❌ Google Chrome is currently RUNNING and locking the profile directory.\n"
                    "   Please completely quit Google Chrome (Press Cmd + Q in Chrome) and try again,\n"
                    "   or omit --chrome-profile to use the isolated scraper profile.\n"
                )
                raise
            if ch:
                logger.warning("Failed launching with channel '%s': %s. Trying fallback...", ch, e)

    raise last_err or RuntimeError("Failed to launch persistent browser context")


def login_and_save_session(
    user_data_dir: Union[Path, str] = REDDIT_PROFILE_DIR,
    login_url: str = "https://www.reddit.com/login",
    channel: Optional[str] = None,
    chrome_profile: Optional[str] = None,
) -> None:
    """
    Launches headful browser to allow user to log in manually, persisting the session.

    Args:
        user_data_dir: Directory where the browser profile and cookies will be stored.
        login_url: Reddit login page URL.
        channel: Optional browser channel to use.
        chrome_profile: Optional Chrome profile name.
    """
    _ensure_playwright()
    profile_path = Path(user_data_dir).resolve()
    profile_path.mkdir(parents=True, exist_ok=True)

    logger.info("Starting interactive login session...")
    if chrome_profile:
        logger.info("Using existing Chrome profile: %s", chrome_profile)
    else:
        logger.info("Profile directory: %s", profile_path)

    with sync_playwright() as p:
        context = create_persistent_context(
            p,
            profile_path,
            headless=False,
            channel=channel,
            chrome_profile=chrome_profile,
        )
        page = context.pages[0] if context.pages else context.new_page()

        try:
            logger.info("Navigating to %s", login_url)
            page.goto(login_url, wait_until="domcontentloaded", timeout=45000)

            print("\n" + "═" * 70)
            print("  [ACTION REQUIRED] Reddit Interactive Login")
            print("  1. Complete login in the opened browser window.")
            print("  2. Verify you can see your home feed or profile.")
            print("  3. Come back to this terminal and press [ENTER] to save session.")
            print("═" * 70 + "\n")

            input("Press [ENTER] after you have logged in...")

            # Quick verification navigation
            logger.info("Verifying session...")
            page.goto("https://www.reddit.com", wait_until="domcontentloaded", timeout=30000)
            time.sleep(2)

            is_valid = is_session_valid(context, test_url="https://www.reddit.com")
            if is_valid:
                logger.info("Session login verified successfully! Profile saved to %s", profile_path)
            else:
                logger.warning(
                    "Session validation check returned uncertain status. "
                    "Profile was saved, but please verify login on next scrape."
                )

        except Exception as e:
            logger.error("Error during interactive login: %s", e)
        finally:
            context.close()


def is_session_valid(
    context: BrowserContext,
    test_url: str = "https://www.reddit.com",
) -> bool:
    """
    Check if the current browser session is active and not challenged/blocked.

    Args:
        context: Active persistent BrowserContext.
        test_url: URL to test navigation against.

    Returns:
        bool: True if session appears valid and accessible, False otherwise.
    """
    page = context.new_page()
    try:
        response = page.goto(test_url, wait_until="domcontentloaded", timeout=30000)
        current_url = page.url.lower()

        # Check for login redirection
        if "/login" in current_url or "/register" in current_url:
            logger.warning("Session check: Redirected to login page (%s)", page.url)
            return False

        # Check response status
        if response and response.status in (401, 403, 429):
            logger.warning("Session check: Received HTTP status %d", response.status)
            return False

        # Check for challenge or block text in body
        try:
            body_text = page.evaluate("document.body.innerText || ''").lower()
            block_keywords = [
                "whoa there, takeout",
                "you've been blocked",
                "blocked by security",
                "cloudflare",
                "turn on cookies",
                "access denied",
                "request blocked",
            ]
            for kw in block_keywords:
                if kw in body_text:
                    logger.warning("Session check: Block/challenge detected ('%s')", kw)
                    return False
        except Exception:
            pass

        return True
    except Exception as e:
        logger.warning("Session validity check encountered an error: %s", e)
        return False
    finally:
        page.close()


# ─── URL & JSON Fetching ──────────────────────────────────────────────────────

def build_reddit_json_url(
    target: str,
    listing: Optional[str] = "hot",
    query: Optional[str] = None,
    sort: Optional[str] = None,
    after: Optional[str] = None,
    limit: int = 100,
    time_period: Optional[str] = None,
) -> str:
    """
    Construct a valid Reddit .json endpoint URL supporting feeds and search queries.

    Args:
        target: Subreddit name (e.g. 'OUTFITS', 'r/malefashionadvice') or complete URL.
        listing: Listing type ('hot', 'top', 'new', 'rising') if no search query.
        query: Optional search keyword/phrase.
        sort: Sort mode ('top', 'relevance', 'new', 'comments').
        after: Reddit pagination token.
        limit: Number of items per request (up to 100).
        time_period: Time filter ('all', 'year', 'month', etc.).

    Returns:
        str: Fully formatted .json endpoint URL.
    """
    target = target.strip()

    # 1. If target is already a full URL
    if target.startswith("http://") or target.startswith("https://"):
        parsed = urlparse(target)
        path = parsed.path
        if not path.endswith(".json"):
            path = path.rstrip("/") + ".json"
        query_params = parse_qs(parsed.query)
        query_params["limit"] = [str(limit)]
        if after:
            query_params["after"] = [after]
        if query and "q" not in query_params:
            query_params["q"] = [query]
        if sort and "sort" not in query_params:
            query_params["sort"] = [sort]
        if time_period and "t" not in query_params:
            query_params["t"] = [time_period]

        new_query = urlencode(query_params, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, new_query, parsed.fragment))

    # Strip r/ prefix if present
    sub = target
    if sub.lower().startswith("r/"):
        sub = sub[2:]
    sub = sub.strip("/")

    # 2. Search Query Endpoint (Subreddit search or Global search)
    if query:
        search_params: Dict[str, Any] = {
            "q": query,
            "limit": limit,
            "sort": sort or "top",
            "t": time_period or "all",
        }
        if after:
            search_params["after"] = after

        if sub and sub.lower() not in ("all", "reddit", ""):
            search_params["restrict_sr"] = 1
            return f"https://www.reddit.com/r/{sub}/search.json?{urlencode(search_params)}"
        else:
            search_params["type"] = "link"
            return f"https://www.reddit.com/search.json?{urlencode(search_params)}"

    # 3. Standard Listing Endpoint
    listing_name = listing or "hot"
    query_dict: Dict[str, Any] = {"limit": limit}
    if after:
        query_dict["after"] = after
    if listing_name == "top" and time_period:
        query_dict["t"] = time_period

    url = f"https://www.reddit.com/r/{sub}/{listing_name}.json?{urlencode(query_dict)}"
    return url


def fetch_json(
    url: str,
    page: Page,
    max_retries: int = 4,
    retry_delay: float = 10.0,
) -> Optional[Dict[str, Any]]:
    """
    Navigate to a Reddit .json URL using an authenticated Playwright page and parse JSON.
    Implements intelligent exponential backoff on HTTP 429/403 or challenge pages.

    Args:
        url: The .json URL to fetch.
        page: Playwright Page instance.
        max_retries: Max retry attempts on transient block/failure.
        retry_delay: Base delay in seconds before retrying.

    Returns:
        Optional[Dict[str, Any]]: Parsed JSON dictionary, or None if fetch failed.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=35000)
            status = response.status if response else 0

            # Small human DOM settling delay
            time.sleep(random.uniform(0.3, 0.7))

            if status in (401, 403, 429, 503):
                # Exponential cooldown: 30s -> 60s -> 120s -> 240s
                sleep_time = (30.0 * (2 ** (attempt - 1))) + random.uniform(2.0, 6.0)
                logger.warning(
                    "⚠ [Rate Limit / HTTP %d] received for %s (attempt %d/%d). "
                    "Cooling off for %.1f seconds...",
                    status,
                    url,
                    attempt,
                    max_retries,
                    sleep_time,
                )
                if attempt < max_retries:
                    time.sleep(sleep_time)
                    continue
                return None

            # Extract raw inner text from body
            raw_text = page.evaluate("document.body.innerText || ''").strip()

            # If innerText is wrapped in pre tag, try alternative selector
            if not raw_text or not (raw_text.startswith("{") or raw_text.startswith("[")):
                pre_element = page.query_selector("pre")
                if pre_element:
                    raw_text = (pre_element.inner_text() or "").strip()

            # Validate whether the page returned a challenge or error page instead of JSON
            if not raw_text or not (raw_text.startswith("{") or raw_text.startswith("[")):
                sleep_time = (25.0 * (2 ** (attempt - 1))) + random.uniform(2.0, 5.0)
                logger.warning(
                    "⚠ Non-JSON challenge response received from %s (attempt %d/%d). "
                    "Preview: %s. Backing off for %.1fs...",
                    url,
                    attempt,
                    max_retries,
                    raw_text[:100] if raw_text else "<empty>",
                    sleep_time,
                )
                if attempt < max_retries:
                    time.sleep(sleep_time)
                    continue
                return None

            data = json.loads(raw_text)
            return data

        except (json.JSONDecodeError, PlaywrightTimeoutError, PlaywrightError) as e:
            sleep_time = retry_delay * attempt + random.uniform(1.0, 3.0)
            logger.warning(
                "Error fetching JSON from %s (attempt %d/%d): %s. Pausing %.1fs...",
                url,
                attempt,
                max_retries,
                e,
                sleep_time,
            )
            if attempt < max_retries:
                time.sleep(sleep_time)
            else:
                return None
        except Exception as e:
            logger.error("Unexpected error fetching %s: %s", url, e)
            return None

    return None


# ─── Image URL Extraction & Filtering ─────────────────────────────────────────

def clean_and_unescape_url(raw_url: str) -> str:
    """Unescapes HTML entities (&amp; -> &) and strips whitespace."""
    if not raw_url:
        return ""
    unescaped = html.unescape(raw_url).strip()
    return unescaped


def is_direct_image_url(url: str) -> bool:
    """Check if the URL path ends with a recognized image extension."""
    if not url:
        return False
    parsed = urlparse(url)
    clean_path = parsed.path.lower()
    for ext in IMAGE_EXTENSIONS:
        if clean_path.endswith(ext):
            return True
    return False


def is_video_or_media_embed(data: Dict[str, Any]) -> bool:
    """Check if post is a video, gifv, or non-image embed."""
    if data.get("is_video"):
        return True
    if data.get("post_hint") in ("hosted:video", "rich:video"):
        return True
    domain = (data.get("domain") or "").lower()
    video_domains = {
        "v.redd.it",
        "youtube.com",
        "youtu.be",
        "tiktok.com",
        "streamable.com",
        "redgifs.com",
        "gfycat.com",
    }
    if domain in video_domains:
        return True
    return False


def is_deleted_or_removed(data: Dict[str, Any]) -> bool:
    """Check if post is removed, deleted, or empty."""
    if data.get("removed_by_category") is not None:
        return True
    if data.get("selftext") in ("[removed]", "[deleted]"):
        return True
    if data.get("title") in ("[deleted]", "[removed]"):
        return True
    return False


def extract_image_urls(json_data: Dict[str, Any]) -> List[str]:
    """
    Extract direct image URLs from a Reddit listing JSON response.

    Handles:
      - Direct image URLs (.jpg, .png, .jpeg, .webp, .gif) on `url` / `url_overridden_by_dest`
      - Multi-image gallery posts (`is_gallery` or `media_metadata`)
      - Fallback preview images (`preview.images[0].source.url`)
      - HTML entity unescaping (&amp; -> &)
      - Video / removed / non-image filtering

    Args:
        json_data: Parsed Reddit listing JSON dictionary.

    Returns:
        List[str]: List of extracted image URLs in order of appearance (deduplicated).
    """
    urls: List[str] = []
    seen: Set[str] = set()

    if not isinstance(json_data, dict):
        return urls

    # Standard Reddit listing structure: data -> children -> list of items
    children = json_data.get("data", {}).get("children", [])
    if not isinstance(children, list):
        return urls

    for child in children:
        if not isinstance(child, dict) or child.get("kind") != "t3":
            continue

        data = child.get("data", {})
        if not isinstance(data, dict):
            continue

        # Skip removed/deleted posts and video links
        if is_deleted_or_removed(data) or is_video_or_media_embed(data):
            continue

        post_urls: List[str] = []

        # 1. Gallery Posts (`is_gallery` == True or `media_metadata` present)
        is_gallery = data.get("is_gallery", False)
        media_metadata = data.get("media_metadata", {})
        if (is_gallery or media_metadata) and isinstance(media_metadata, dict):
            gallery_order = [
                item.get("media_id")
                for item in data.get("gallery_data", {}).get("items", [])
                if isinstance(item, dict) and item.get("media_id")
            ]
            keys_to_process = gallery_order if gallery_order else list(media_metadata.keys())

            for media_id in keys_to_process:
                meta = media_metadata.get(media_id)
                if not isinstance(meta, dict):
                    continue
                if meta.get("status") != "valid":
                    continue

                source_obj = meta.get("s", {})
                img_url = source_obj.get("u") or source_obj.get("gif")

                # Fallback to largest preview if source URL not directly provided
                if not img_url and meta.get("p"):
                    previews = meta.get("p", [])
                    if isinstance(previews, list) and len(previews) > 0:
                        img_url = previews[-1].get("u")

                if img_url:
                    clean_url = clean_and_unescape_url(img_url)
                    if clean_url:
                        post_urls.append(clean_url)

        # 2. Direct Image Link on main URL
        main_url = data.get("url_overridden_by_dest") or data.get("url", "")
        if not post_urls and main_url and is_direct_image_url(main_url):
            clean_url = clean_and_unescape_url(main_url)
            if clean_url:
                post_urls.append(clean_url)

        # 3. Fallback: Preview Image
        if not post_urls:
            preview_images = data.get("preview", {}).get("images", [])
            if isinstance(preview_images, list) and len(preview_images) > 0:
                first_preview = preview_images[0]
                if isinstance(first_preview, dict):
                    source_url = first_preview.get("source", {}).get("url")
                    if source_url:
                        clean_url = clean_and_unescape_url(source_url)
                        if clean_url:
                            post_urls.append(clean_url)

        # Add to global list preserving order and deduplicating
        for u in post_urls:
            if u and u not in seen:
                seen.add(u)
                urls.append(u)

    return urls


# ─── Category Scraping Engine with Rate-Limiting Controls ─────────────────────

class RateLimitTracker:
    """Tracks page requests and enforces periodic cooling-off pauses."""

    def __init__(
        self,
        delay_range: Tuple[float, float] = (REDDIT_DELAY_MIN_SEC, REDDIT_DELAY_MAX_SEC),
        batch_size: int = REDDIT_BATCH_SIZE,
        batch_cooldown: float = REDDIT_BATCH_COOLDOWN_SEC,
        category_cooldown: float = REDDIT_CATEGORY_COOLDOWN_SEC,
    ):
        self.delay_range = delay_range
        self.batch_size = batch_size
        self.batch_cooldown = batch_cooldown
        self.category_cooldown = category_cooldown
        self.request_count = 0

    def pause_after_request(self) -> None:
        """Call after each request. Executes randomized delay and batch cooldowns."""
        self.request_count += 1
        if self.batch_size > 0 and self.request_count % self.batch_size == 0:
            pause = self.batch_cooldown + random.uniform(1.0, 4.0)
            logger.info(
                "⏸ [Anti-Rate-Limit] Batch pause: cooling down for %.1fs after %d requests...",
                pause,
                self.request_count,
            )
            time.sleep(pause)
        else:
            delay = random.uniform(self.delay_range[0], self.delay_range[1])
            logger.debug("Sleeping %.2fs before next request (request #%d)...", delay, self.request_count)
            time.sleep(delay)

    def pause_after_category(self, cat_name: str) -> None:
        """Call after finishing a category before moving to the next."""
        pause = self.category_cooldown + random.uniform(0.5, 2.0)
        logger.info("⏸ [Cooldown] Pausing %.1fs before next category (finished '%s')...", pause, cat_name)
        time.sleep(pause)


def scrape_category_target(
    category_spec: Dict[str, Any],
    context: BrowserContext,
    tracker: RateLimitTracker,
    limit: int = 100,
    global_seen_urls: Optional[Set[str]] = None,
) -> List[str]:
    """
    Scrape image URLs for a single category/query specification.

    Args:
        category_spec: Dict containing category metadata:
            - category: unique name
            - subreddit: target subreddit or 'all'
            - query: optional search term
            - sort: optional sort mode ('top', 'relevance', etc.)
            - time: optional time filter ('all', 'year', etc.)
            - listing: optional feed listing ('hot', 'top')
        context: Active Playwright BrowserContext.
        tracker: RateLimitTracker instance.
        limit: Max images to collect for this category.
        global_seen_urls: Set of all previously seen URLs to prevent duplicate scraping.

    Returns:
        List[str]: Collected deduplicated image URLs for this category.
    """
    cat_name = category_spec.get("category", "unnamed_category")
    subreddit = category_spec.get("subreddit", "")
    query = category_spec.get("query")
    sort = category_spec.get("sort", "top")
    time_filter = category_spec.get("time", "all")
    listing = category_spec.get("listing", "hot")
    desc = category_spec.get("description", "")

    query_info = f"query='{query}' in r/{subreddit}" if query else f"r/{subreddit}/{listing}"
    logger.info("▶ Scraping [%s] (%s | goal: %d images)...", cat_name, query_info, limit)
    if desc:
        logger.info("  Description: %s", desc)

    category_urls: List[str] = []
    seen_in_cat: Set[str] = set()

    page = context.new_page()

    try:
        after_token: Optional[str] = None
        page_num = 1
        max_pages = max(1, (limit // 25) + 4)

        while page_num <= max_pages and len(category_urls) < limit:
            json_url = build_reddit_json_url(
                target=subreddit,
                listing=listing,
                query=query,
                sort=sort,
                after=after_token,
                limit=min(100, limit - len(category_urls) + 20),
                time_period=time_filter,
            )

            logger.debug("  [%s] Fetching page %d: %s", cat_name, page_num, json_url)
            json_data = fetch_json(json_url, page)

            if not json_data:
                logger.warning("  [%s] Could not retrieve data for page %d. Moving on.", cat_name, page_num)
                break

            batch_urls = extract_image_urls(json_data)
            new_in_batch = 0
            for u in batch_urls:
                if u not in seen_in_cat and (global_seen_urls is None or u not in global_seen_urls):
                    seen_in_cat.add(u)
                    if global_seen_urls is not None:
                        global_seen_urls.add(u)
                    category_urls.append(u)
                    new_in_batch += 1
                    if len(category_urls) >= limit:
                        break

            logger.info(
                "  [%s] Page %d: Found %d images (+%d new, %d/%d total for category)",
                cat_name,
                page_num,
                len(batch_urls),
                new_in_batch,
                len(category_urls),
                limit,
            )

            # Check pagination after token
            after_token = json_data.get("data", {}).get("after")
            if not after_token:
                logger.info("  [%s] Reached end of listing/search results (no after token).", cat_name)
                break

            page_num += 1
            tracker.pause_after_request()

    except Exception as e:
        logger.error("  [%s] Encountered error during scraping: %s", cat_name, e)
    finally:
        page.close()

    logger.info("✔ [%s] Finished category: %d images collected", cat_name, len(category_urls))
    return category_urls


def normalize_categories(
    raw_input: Union[Dict[str, Any], List[Any], str, Path],
) -> List[Dict[str, Any]]:
    """Normalize various category input formats into standard list of category specifications."""
    # If path to JSON file
    if isinstance(raw_input, (str, Path)):
        p = Path(raw_input)
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                raw_input = json.load(f)

    results: List[Dict[str, Any]] = []

    if isinstance(raw_input, dict):
        for k, v in raw_input.items():
            if isinstance(v, str):
                results.append({"category": k, "subreddit": v})
            elif isinstance(v, dict):
                spec = dict(v)
                spec.setdefault("category", k)
                results.append(spec)
    elif isinstance(raw_input, list):
        for item in raw_input:
            if isinstance(item, dict):
                spec = dict(item)
                folder = spec.get("folder") or spec.get("category") or spec.get("name")
                target = spec.get("subreddit") or spec.get("target") or spec.get("url")
                if folder and not spec.get("category"):
                    spec["category"] = folder
                if target and not spec.get("subreddit"):
                    spec["subreddit"] = target
                results.append(spec)
            elif isinstance(item, str):
                clean = item.replace("r/", "").strip("/")
                results.append({"category": f"reddit_{clean}", "subreddit": item})

    return results


def scrape_categories(
    categories: Union[Dict[str, Any], List[Any], str, Path],
    context: BrowserContext,
    limit: int = 100,
    delay_range: Tuple[float, float] = (REDDIT_DELAY_MIN_SEC, REDDIT_DELAY_MAX_SEC),
    batch_size: int = REDDIT_BATCH_SIZE,
    batch_cooldown: float = REDDIT_BATCH_COOLDOWN_SEC,
    category_cooldown: float = REDDIT_CATEGORY_COOLDOWN_SEC,
    output_json_path: Optional[Union[Path, str]] = None,
    target_total: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    Scrape image URLs across multiple categories or subreddits with rate-limiting controls
    and incremental progress persistence.

    Args:
        categories: Specifications of categories / queries.
        context: Active persistent BrowserContext.
        limit: Max images per category.
        delay_range: Randomized delay range (min_sec, max_sec).
        batch_size: Requests before cooling pause.
        batch_cooldown: Duration of batch cooldown in seconds.
        category_cooldown: Duration of pause between categories in seconds.
        output_json_path: Optional path to save intermediate progress after each category.
        target_total: Optional global image limit across all categories.

    Returns:
        Dict[str, List[str]]: Mapping of category name -> list of image URLs.
    """
    specs = normalize_categories(categories)
    tracker = RateLimitTracker(
        delay_range=delay_range,
        batch_size=batch_size,
        batch_cooldown=batch_cooldown,
        category_cooldown=category_cooldown,
    )

    # Load existing scraped results to allow seamless resuming
    results: Dict[str, List[str]] = {}
    global_seen_urls: Set[str] = set()

    if output_json_path:
        out_p = Path(output_json_path).resolve()
        if out_p.exists():
            try:
                with open(out_p, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                    if isinstance(existing, dict):
                        results = existing
                        for u_list in results.values():
                            if isinstance(u_list, list):
                                global_seen_urls.update(u_list)
                        logger.info("Resuming from existing JSON: %d URLs previously recorded", len(global_seen_urls))
            except Exception as e:
                logger.warning("Could not load existing output JSON: %s", e)

    total_collected = sum(len(urls) for urls in results.values())
    logger.info("Total categories to process: %d (Current total images: %d)", len(specs), total_collected)

    for i, spec in enumerate(specs, start=1):
        if target_total and total_collected >= target_total:
            logger.info("🎯 Reached global target limit of %d total images! Stopping scrape.", target_total)
            break

        cat_name = spec.get("category", f"category_{i}")
        logger.info("\n─── [%d/%d] Category: %s ───", i, len(specs), cat_name)

        existing_urls = results.get(cat_name, [])
        if len(existing_urls) >= limit:
            logger.info("Category '%s' already has %d/%d images. Skipping.", cat_name, len(existing_urls), limit)
            continue

        needed = limit - len(existing_urls)
        new_urls = scrape_category_target(
            category_spec=spec,
            context=context,
            tracker=tracker,
            limit=needed,
            global_seen_urls=global_seen_urls,
        )

        # Merge results
        combined = list(existing_urls)
        for u in new_urls:
            if u not in combined:
                combined.append(u)
        results[cat_name] = combined
        total_collected = sum(len(urls) for urls in results.values())

        # Save progress incrementally
        if output_json_path:
            save_results_to_json(results, output_json_path)

        if i < len(specs) and (not target_total or total_collected < target_total):
            tracker.pause_after_category(cat_name)

    return results


# ─── Results Output & Image Downloader ────────────────────────────────────────

def save_results_to_json(results: Dict[str, List[str]], filepath: Union[Path, str]) -> None:
    """Save category image URL map to a JSON file."""
    path = Path(filepath).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.debug("Saved %d categories of URLs to %s", len(results), path)


def print_scrape_summary(results: Dict[str, List[str]]) -> None:
    """Print a clean summary of scraped image counts per category."""
    total_images = sum(len(urls) for urls in results.values())
    print("\n" + "═" * 70)
    print("  LookMax — Reddit Scraping Summary")
    print("═" * 70)
    for cat, urls in results.items():
        if urls:
            print(f"  📁 {cat:<36} : {len(urls):>5} images")
    print("─" * 70)
    print(f"  🌟 Total Categories Extracted     : {len(results):>5}")
    print(f"  🌟 Total Unique Images Extracted  : {total_images:>5}")
    print("═" * 70 + "\n")


def _download_single_image(url: str, dest_dir: Path, seen_hashes: Set[str]) -> Optional[str]:
    """Download a single image URL and verify hash deduplication."""
    import requests
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.get(url, headers=headers, timeout=25, stream=True)
        if resp.status_code != 200:
            return None
        raw_bytes = resp.content
        if len(raw_bytes) < 10_000:  # Skip tiny/corrupted images
            return None

        sha = hashlib.sha256(raw_bytes).hexdigest()
        if sha in seen_hashes:
            return None
        seen_hashes.add(sha)

        ext = ".jpg"
        out_file = dest_dir / f"{sha[:16]}{ext}"
        out_file.write_bytes(raw_bytes)
        return out_file.name
    except Exception:
        return None


def download_images(
    image_map: Dict[str, List[str]],
    output_dir: Union[Path, str] = RAW_SCRAPES_DIR,
    max_workers: int = DOWNLOAD_WORKERS,
) -> Dict[str, int]:
    """
    Download image URLs to disk organized by category subfolders.

    Args:
        image_map: Dict mapping category name -> list of image URLs.
        output_dir: Root directory to download raw image categories into.
        max_workers: Concurrency thread limit.

    Returns:
        Dict[str, int]: Count of successfully downloaded images per category.
    """
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    base_dir = Path(output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    downloaded_counts: Dict[str, int] = {}
    seen_hashes: Set[str] = set()

    # Pre-populate seen hashes with existing file stems on disk
    for f in base_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            seen_hashes.add(f.stem)

    if seen_hashes:
        logger.info("Found %d pre-existing image files on disk; skipping duplicates.", len(seen_hashes))

    print(f"\nDownloading images to {base_dir} (workers={max_workers})...")

    for cat_name, urls in image_map.items():
        if not urls:
            downloaded_counts[cat_name] = 0
            continue

        cat_dir = base_dir / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        pbar = tqdm(total=len(urls), unit="img", desc=f"  {cat_name[:24]}") if has_tqdm else None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_download_single_image, u, cat_dir, seen_hashes): u
                for u in urls
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    saved += 1
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()

        downloaded_counts[cat_name] = saved
        logger.info("Category '%s': %d/%d images saved to %s", cat_name, saved, len(urls), cat_dir)

    return downloaded_counts


# ─── Main CLI Entrypoint ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LookMax — Playwright Reddit JSON Image Scraper with Anti-Rate-Limit Controls",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--login",
        action="store_true",
        help="Launch interactive browser window to log in to Reddit and persist session.",
    )
    parser.add_argument(
        "--check-session",
        action="store_true",
        help="Check if the existing saved browser profile session is still valid.",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=str(REDDIT_QUERIES_FILE) if REDDIT_QUERIES_FILE.exists() else None,
        help="Path to JSON queries file containing curated search queries and subreddits (110+ items).",
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        help="List of subreddit names to scrape (e.g. --subreddits OUTFITS streetwear).",
    )
    parser.add_argument(
        "--categories-file",
        type=str,
        help="Path to custom JSON file mapping category names to subreddits or query dicts.",
    )
    parser.add_argument(
        "--subreddit",
        type=str,
        help="Target subreddit when using --query search flag.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Custom search query term to execute on target subreddit or globally.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of images to scrape per category / query.",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=None,
        help="Optional global image limit (e.g. 10000). Stops automatically when reached.",
    )
    parser.add_argument(
        "--listings",
        nargs="+",
        default=["hot", "top"],
        choices=["hot", "top", "new", "rising"],
        help="Reddit listings to crawl for feed targets.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REDDIT_SCRAPE_OUTPUT_JSON),
        help="Output path for the extracted image URLs JSON file.",
    )
    parser.add_argument(
        "--profile-dir",
        type=str,
        default=str(REDDIT_PROFILE_DIR),
        help="Directory to store persistent browser cookies and profile.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the scraped images directly to the raw scrapes directory.",
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default=str(RAW_SCRAPES_DIR),
        help="Destination directory when --download is enabled.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run scraping browser with visible window instead of headless mode.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Browser channel to use (e.g. 'chrome', 'msedge', or 'chromium'). Default is auto-detect.",
    )
    parser.add_argument(
        "--chrome-profile",
        type=str,
        default=None,
        help="Directly use an existing Google Chrome profile folder (e.g. 'Default', 'Profile 1'). Note: Chrome must be closed first.",
    )
    parser.add_argument(
        "--list-chrome-profiles",
        action="store_true",
        help="List all existing Google Chrome profiles detected on this machine and exit.",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=REDDIT_DELAY_MIN_SEC,
        help="Minimum randomized delay between requests in seconds.",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=REDDIT_DELAY_MAX_SEC,
        help="Maximum randomized delay between requests in seconds.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=REDDIT_BATCH_SIZE,
        help="Number of page requests before triggering a cooling-off pause.",
    )
    parser.add_argument(
        "--batch-cooldown",
        type=float,
        default=REDDIT_BATCH_COOLDOWN_SEC,
        help="Duration of cooling-off pause between request batches in seconds.",
    )
    parser.add_argument(
        "--category-cooldown",
        type=float,
        default=REDDIT_CATEGORY_COOLDOWN_SEC,
        help="Duration of pause between switching categories in seconds.",
    )

    args = parser.parse_args()

    # 0. List Chrome Profiles Mode
    if args.list_chrome_profiles:
        print_available_chrome_profiles()
        sys.exit(0)

    # 1. Interactive Login Mode
    if args.login:
        login_and_save_session(
            user_data_dir=args.profile_dir,
            channel=args.channel,
            chrome_profile=args.chrome_profile,
        )
        sys.exit(0)

    # 2. Check Session Validity Mode
    if args.check_session:
        _ensure_playwright()
        with sync_playwright() as p:
            context = create_persistent_context(
                p,
                args.profile_dir,
                headless=not args.no_headless,
                channel=args.channel,
                chrome_profile=args.chrome_profile,
            )
            try:
                valid = is_session_valid(context)
                if valid:
                    print("\n✅ Saved Reddit session is VALID and ready for scraping.\n")
                else:
                    print("\n⚠ Saved session is INVALID or EXPIRED. Run with --login to refresh.\n")
            finally:
                context.close()
        sys.exit(0)

    # 3. Determine Categories / Subreddits / Queries to Scrape
    categories_input: Union[Dict[str, Any], List[Any], str, Path]
    if args.query:
        sub = args.subreddit or "all"
        clean_name = re.sub(r"[^a-zA-Z0-9_]+", "_", args.query).strip("_").lower()
        categories_input = [{
            "category": f"search_{sub}_{clean_name}",
            "subreddit": sub,
            "query": args.query,
            "sort": "top",
            "time": "all",
        }]
    elif args.subreddits:
        categories_input = args.subreddits
    elif args.categories_file:
        categories_input = args.categories_file
    elif args.queries_file:
        categories_input = args.queries_file
    else:
        # Fallback to default REDDIT_SOURCES
        categories_input = REDDIT_SOURCES

    _ensure_playwright()

    print(f"\n{'═' * 68}")
    print("  LookMax — Reddit Playwright JSON Scraper (Rate-Limit Protected)")
    print(f"{'═' * 68}")
    if args.chrome_profile:
        print(f"  Chrome Prof  : {args.chrome_profile}")
    else:
        print(f"  Profile Dir  : {args.profile_dir}")
    print(f"  Headless     : {not args.no_headless}")
    print(f"  Limit/cat    : {args.limit}")
    if args.target_total:
        print(f"  Target Total : {args.target_total} images")
    print(f"  Delays       : {args.delay_min}s - {args.delay_max}s randomized per request")
    print(f"  Batch Pause  : {args.batch_cooldown}s every {args.batch_size} requests")
    print(f"  Cat Pause    : {args.category_cooldown}s between categories")
    print(f"  Output JSON  : {args.output}")
    print(f"  Download     : {args.download}")
    if args.download:
        print(f"  Download Dir : {args.download_dir}")
    print(f"{'═' * 68}\n")

    with sync_playwright() as p:
        context = create_persistent_context(
            p,
            args.profile_dir,
            headless=not args.no_headless,
            channel=args.channel,
            chrome_profile=args.chrome_profile,
        )
        try:
            # Check session validity before proceeding
            valid = is_session_valid(context)
            if not valid:
                logger.warning(
                    "Reddit session is not authenticated or challenge encountered. "
                    "Scraping will proceed as public/guest session. "
                    "Run with --login if you experience rate limits."
                )

            # Perform Scraping
            results = scrape_categories(
                categories=categories_input,
                context=context,
                limit=args.limit,
                delay_range=(args.delay_min, args.delay_max),
                batch_size=args.batch_size,
                batch_cooldown=args.batch_cooldown,
                category_cooldown=args.category_cooldown,
                output_json_path=args.output,
                target_total=args.target_total,
            )

            # Save Final Results to JSON & Print Summary
            save_results_to_json(results, args.output)
            print_scrape_summary(results)

            # Optional Image Download
            if args.download:
                download_images(
                    image_map=results,
                    output_dir=args.download_dir,
                    max_workers=DOWNLOAD_WORKERS,
                )

        finally:
            context.close()


if __name__ == "__main__":
    main()
