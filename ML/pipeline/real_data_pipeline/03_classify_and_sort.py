"""
LookMax ML Pipeline — Phase 3
==============================
03_classify_and_sort.py

Two-stage image qualification, AI classification, and auto-sorting engine:

  STEP A — Fast Heuristic Quality Filter (< 15ms/image):
    • Rejects corrupted or unreadable images
    • Rejects low-resolution thumbnails (< 480px shortest edge)
    • Rejects extreme aspect ratio crops (> 2.5:1)
    • Rejects excessively blurry images (Laplacian variance threshold)
    • Rejections organized into categorized subfolders under 2_VLM_Processing/filtered_rejected/

  STEP B — Local Vision-Language Model (VLM) Batch Engine:
    • Supported backends: Ollama (llava:7b / llama3.2-vision), MLX-VLM (M-series native), Gemini
    • Strict JSON schema qualification:
        - is_valid_human_look: filters out flat-lays, shoe closeups, memes, text, scenery
        - gender: male / female
        - age_bracket: under_35 / 35_to_50 / over_50
        - aesthetic_tier: 1_Needs_Improvement / 2_Average / 3_Polished
        - posture_score, fit_score, overall_score, shot_type
    • Auto-routes each qualified image directly into:
        ML/data/3_CoreML_Training_Data/{Demographic}/{Aesthetic_Tier}/
    • Appends structured JSON annotations to metadata_logs/dataset_annotations.jsonl
    • Resumable: automatically skips previously processed images

Usage Examples:
    # 1. Quick dry-run test (25 images)
    python3 ML/pipeline/03_classify_and_sort.py --sample 25 --dry-run

    # 2. Process batch using local Ollama model with 4 parallel workers
    python3 ML/pipeline/03_classify_and_sort.py --engine ollama --model llava:7b --workers 4

    # 3. Process with native Apple Silicon MLX-VLM engine
    python3 ML/pipeline/03_classify_and_sort.py --engine mlx_vlm

    # 4. View dataset balance and distribution without processing
    python3 ML/pipeline/03_classify_and_sort.py --stats-only
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import shutil
import socket
import sys
import time

# Set global socket read timeout to prevent indefinite network hangs on TCP sockets
socket.setdefaulttimeout(45.0)
from collections import defaultdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add pipeline directory to import config
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # config.py now lives one level up
from config import (
    AESTHETIC_TIERS,
    ANNOTATIONS_FILE,
    BLUR_LAPLACIAN_THRESHOLD,
    DEMOGRAPHICS,
    GEMINI_MODEL,
    IMAGE_EXTENSIONS,
    MAX_ASPECT_RATIO,
    METADATA_LOGS_DIR,
    MIN_IMAGE_DIMENSION_PX,
    MLX_MODEL_ID,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    RAW_SCRAPES_DIR,
    REJECTED_DIR,
    TRAINING_DATA_DIR,
    VLM_ENGINE,
    VLM_PARALLEL_WORKERS,
    VLM_REQUEST_TIMEOUT_SEC,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("classify_and_sort")

# Conditional imports
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ─── VLM Prompt & Qualification Schema ───────────────────────────────────────
VLM_SYSTEM_PROMPT = """\
You are an expert fashion, outfit, facial aesthetics, and grooming analyst for the LookMax style coaching platform.
Analyze the image carefully and respond ONLY with a raw JSON object (no markdown, no prose, no code fences).

Required JSON schema:
{
  "is_valid_human_look": <true or false>,
  "gender": <"male" or "female">,
  "age_bracket": <"under_35", "35_to_50", or "over_50">,
  "focus_type": <"outfit" or "face_grooming">,
  "shot_type": <"full_body", "upper_body", "portrait", or "other">,
  "posture_score": <float 1.0 to 10.0>,
  "fit_score": <float 1.0 to 10.0>,
  "overall_score": <float 1.0 to 10.0>,
  "aesthetic_tier": <"1_Needs_Improvement", "2_Average", or "3_Polished">,
  "confidence": <float 0.0 to 1.0>,
  "summary_critique": "<one clear sentence explaining the rating>"
}

CRITICAL RULES:
1. is_valid_human_look:
   - Set FALSE if: clothing items on hangers/racks, flat-lays on bed/floor, spreadsheets/text tables/screenshots, restaurant/cafe/room scenery, shoe-only closeups, fabric detail/stain shots, memes, or any image without an active person wearing the outfit or showing their face/hair.
   - Set TRUE ONLY IF a real person is actively wearing an outfit, showing a haircut/grooming look, or posing for a fit check/posture evaluation.
2. focus_type:
   - "face_grooming" if the image focuses on the head, haircut, beard, skin, facial aesthetics, or upper-chest portrait / selfie.
   - "outfit" if the image showcases full-body or 3/4-body clothing silhouette, coordination, and styling.
3. aesthetic_tier & overall_score calibration (DO NOT default to 3_Polished):
   - FOR OUTFITS:
     • "1_Needs_Improvement" (1.0 to 4.9): Ill-fitting/wrinkled/stained clothes, excessive bagginess/sagging, sloppy execution, poor slouching posture, or mismatched garments.
     • "2_Average" (5.0 to 7.4): Standard daily casual wear, basic tee & jeans, ordinary unstyled hoodies, typical office casual, normal posture. MOST daily outfits belong here.
     • "3_Polished" (7.5 to 10.0): Exceptionally sharp tailoring, cohesive color harmony, sophisticated layering, confident upright posture, high-end modern streetwear, or formal elegance.
   - FOR FACE & GROOMING:
     • "1_Needs_Improvement" (1.0 to 4.9): Messy unstyled bedhead hair, overgrown/patchy untrimmed beard, severe redness/tired eyes, unflattering camera angle/lighting.
     • "2_Average" (5.0 to 7.4): Clean daily grooming baseline, standard neat haircut, natural everyday shave, neutral daily lighting.
     • "3_Polished" (7.5 to 10.0): Crisp styled hair/fade, sharp beard lines, glowing skin clarity, harmonious facial presentation.
4. age_bracket:
   - "under_35" for young adults, college students, 20s to early 30s.
   - "35_to_50" for 30s to 40s.
   - "over_50" for mature adults and seniors.
"""


# ──────────────────────────────────────────────────────────────────────────────
# STEP A — Fast Quality Heuristic Filter (< 15ms/image)
# ──────────────────────────────────────────────────────────────────────────────
def heuristic_filter(image_path: Path) -> Tuple[bool, str]:
    """
    Evaluates basic physical quality constraints using fast OpenCV/PIL operations.

    Returns:
        (passed: bool, reason: str)
    """
    if not image_path.exists() or image_path.stat().st_size == 0:
        return False, "empty_file_0bytes"

    # Fast PIL dimension check if OpenCV not available
    if not HAS_CV2:
        if HAS_PIL:
            try:
                with PILImage.open(image_path) as img:
                    w, h = img.size
                    if min(w, h) < MIN_IMAGE_DIMENSION_PX:
                        return False, f"too_small_{min(w, h)}px"
                    ratio = max(w, h) / max(min(w, h), 1)
                    if ratio > MAX_ASPECT_RATIO:
                        return False, f"bad_aspect_ratio_{ratio:.2f}"
                return True, "ok"
            except Exception as e:
                return False, f"unreadable_pil_{str(e)[:20]}"
        return True, "skipped_heuristics"

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "unreadable_cv2"

        h, w = img.shape[:2]

        # 1. Minimum dimension check
        if min(h, w) < MIN_IMAGE_DIMENSION_PX:
            return False, f"too_small_{min(h, w)}px"

        # 2. Aspect ratio check
        ratio = max(h, w) / max(min(h, w), 1)
        if ratio > MAX_ASPECT_RATIO:
            return False, f"bad_aspect_ratio_{ratio:.2f}"

        # 3. Blur detection via Laplacian variance
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < BLUR_LAPLACIAN_THRESHOLD:
            return False, f"blurry_laplacian_{lap_var:.1f}"

        return True, "ok"

    except Exception as e:
        return False, f"heuristic_error_{str(e)[:30]}"


# ──────────────────────────────────────────────────────────────────────────────
# STEP B — VLM Classification Backends
# ──────────────────────────────────────────────────────────────────────────────
def image_to_base64(image_path: Path) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_vlm_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON parser for VLM outputs, handling code blocks, whitespace,
    and single-quoted keys/values.
    """
    if not raw:
        return None

    cleaned = raw.strip()
    # Strip markdown code blocks
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()

    # Extract outermost JSON structure
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None

    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # Fallback cleanup for trailing commas or single quotes
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        return json.loads(fixed)
    except Exception:
        return None


def classify_with_ollama(
    image_path: Path,
    model: str = OLLAMA_MODEL,
    host: str = OLLAMA_HOST,
    timeout: int = VLM_REQUEST_TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    """Classify an image using Ollama's local vision API."""
    import requests

    try:
        b64 = image_to_base64(image_path)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{VLM_SYSTEM_PROMPT}\n\nAnalyze this image and return the JSON object:",
                    "images": [b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 450},
        }

        resp = requests.post(f"{host}/api/chat", json=payload, timeout=timeout)
        if resp.status_code != 200:
            logger.debug("Ollama returned status %d for %s", resp.status_code, image_path.name)
            return None

        content = resp.json().get("message", {}).get("content", "")
        return parse_vlm_json(content)

    except Exception as e:
        logger.debug("Ollama classification exception for %s: %s", image_path.name, e)
        return None


import threading

_THREAD_LOCAL = threading.local()


class PacedRateLimiter:
    """
    Thread-safe paced rate limiter that spaces requests evenly across all workers
    and supports global backoff when a 429 is encountered by any worker.
    """
    def __init__(self, requests_per_minute: float = 14.0, min_interval_sec: float = 4.3):
        self.interval = max(60.0 / requests_per_minute, min_interval_sec)
        self.lock = threading.Lock()
        self.next_allowed_time = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed_time:
                wait_time = self.next_allowed_time - now
                self.next_allowed_time += self.interval
            else:
                wait_time = 0.0
                self.next_allowed_time = now + self.interval

        if wait_time > 0:
            time.sleep(wait_time)

    def trigger_backoff(self, seconds: float = 10.0):
        """When any worker gets a 429, push back the next allowed request time for ALL workers."""
        with self.lock:
            now = time.time()
            self.next_allowed_time = max(self.next_allowed_time, now + seconds)


_GEMINI_RATE_LIMITER = PacedRateLimiter(requests_per_minute=14.0, min_interval_sec=4.3)


def _get_gemini_client(api_key: Optional[str] = None):
    """Retrieve thread-local Gemini/Gemma client instance to avoid cross-thread socket contention."""
    if hasattr(_THREAD_LOCAL, "gemini_client") and _THREAD_LOCAL.gemini_client is not None:
        return _THREAD_LOCAL.gemini_client

    from google import genai
    from google.genai import types

    resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not resolved_key:
        return None
    client = genai.Client(
        api_key=resolved_key,
        http_options=types.HttpOptions(timeout=45000),
    )
    _THREAD_LOCAL.gemini_client = client
    return client


def classify_with_gemini(
    image_path: Path,
    model: str = GEMINI_MODEL,
    api_key: Optional[str] = None,
    max_retries: int = 5,
) -> Optional[Dict[str, Any]]:
    """Classify an image using Google Gemini/Gemma API with thread-safe client, rate-limiting and retry backoff."""
    try:
        from google.genai import types as genai_types

        client = _get_gemini_client(api_key)
        if not client:
            return None

        # Resize image in memory to max 800px to optimize network upload and token efficiency
        try:
            with Image.open(image_path) as img:
                img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                img_bytes = buf.getvalue()
                mime = "image/jpeg"
        except Exception:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            mime = "image/jpeg"

        is_gemma = "gemma" in str(model).lower()

        for attempt in range(max_retries):
            try:
                _GEMINI_RATE_LIMITER.wait()

                gen_config_kwargs: Dict[str, Any] = {"temperature": 0.1}
                if not is_gemma:
                    gen_config_kwargs["response_mime_type"] = "application/json"

                response = client.models.generate_content(
                    model=model,
                    contents=[
                        genai_types.Part.from_bytes(data=img_bytes, mime_type=mime),
                        f"{VLM_SYSTEM_PROMPT}\n\nAnalyze this image and return the required JSON object:",
                    ],
                    config=genai_types.GenerateContentConfig(**gen_config_kwargs),
                )

                parsed = parse_vlm_json(response.text)
                if parsed is not None:
                    return parsed
                else:
                    logger.debug("Failed parsing JSON from %s output: %s", model, response.text[:100] if response.text else "empty")
                    if attempt < max_retries - 1:
                        time.sleep(1.0 + random.uniform(0.1, 0.5))
                        continue
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5 + random.uniform(2.0, 5.0)
                    _GEMINI_RATE_LIMITER.trigger_backoff(wait_time)
                    logger.warning("Rate limited on %s, backing off globally for %.1fs...", image_path.name, wait_time)
                    time.sleep(wait_time)
                    continue
                if ("timeout" in err_str.lower() or "read operation timed out" in err_str.lower()) and attempt < max_retries - 1:
                    logger.warning("Socket timeout on %s (attempt %d/%d). Retrying...", image_path.name, attempt + 1, max_retries)
                    time.sleep(2.0 + random.uniform(0.5, 1.5))
                    continue
                logger.debug("Gemini/Gemma classification exception for %s: %s", image_path.name, e)
                return None

        return None

    except Exception as e:
        logger.debug("Gemini top-level exception for %s: %s", image_path.name, e)
        return None


def classify_with_mlx_vlm(
    image_path: Path,
    mlx_processor: Any,
    mlx_model: Any,
    mlx_config: Any,
) -> Optional[Dict[str, Any]]:
    """Classify an image using local MLX-VLM on Apple Silicon."""
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        user_content = f"{VLM_SYSTEM_PROMPT}\n\nAnalyze this image and return the JSON object:"
        formatted_prompt = apply_chat_template(
            mlx_processor,
            mlx_config,
            user_content,
            num_images=1,
        )
        result = generate(
            mlx_model,
            mlx_processor,
            image=str(image_path),
            prompt=formatted_prompt,
            max_tokens=400,
            temperature=0.1,
            verbose=False,
        )
        text = result.text if hasattr(result, "text") else str(result)
        return parse_vlm_json(text)

    except Exception as e:
        logger.debug("MLX-VLM classification exception: %s", e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# STEP C — Routing and Directory Resolution
# ──────────────────────────────────────────────────────────────────────────────
def resolve_demographic(gender: str, age_bracket: str) -> Optional[str]:
    """Resolve normalized demographic directory name."""
    g_str = str(gender).strip().lower()
    a_str = str(age_bracket).strip().lower()

    if "female" in g_str or "woman" in g_str or "women" in g_str:
        prefix = "Women"
    else:
        prefix = "Men"

    if "under" in a_str or "u35" in a_str or "young" in a_str or "20" in a_str:
        suffix = "Under_35"
    elif "over" in a_str or "o50" in a_str or "senior" in a_str or "mature" in a_str or "60" in a_str:
        suffix = "Over_50"
    else:
        suffix = "35_to_50"

    resolved = f"{prefix}_{suffix}"
    return resolved if resolved in DEMOGRAPHICS else None


def resolve_tier(aesthetic_tier: str, overall_score: Optional[float] = None) -> Optional[str]:
    """Resolve normalized aesthetic tier directory name."""
    t_str = str(aesthetic_tier).strip().lower().replace(" ", "_")

    if "1" in t_str or "needs" in t_str or "improve" in t_str or "low" in t_str or "poor" in t_str or "bad" in t_str:
        return "1_Needs_Improvement"
    elif "3" in t_str or "polish" in t_str or "high" in t_str or "sartorial" in t_str or "good" in t_str or "great" in t_str or "elegan" in t_str:
        return "3_Polished"
    elif "2" in t_str or "average" in t_str or "mid" in t_str or "med" in t_str or "decent" in t_str:
        return "2_Average"

    # Fallback to score mapping if string unmapped
    if overall_score is not None:
        if overall_score >= 7.5:
            return "3_Polished"
        elif overall_score < 5.0:
            return "1_Needs_Improvement"
        else:
            return "2_Average"

    return "2_Average"


def resolve_stream(focus_type: Optional[str], shot_type: Optional[str], image_path: Path) -> str:
    """
    Determine whether the image belongs to 'Face_Grooming' or 'Outfit' training stream.
    Combines VLM focus_type, shot_type, and scraper category keywords.
    """
    f_str = str(focus_type or "").lower().strip()
    s_str = str(shot_type or "").lower().strip()
    p_str = str(image_path).lower()

    # 1. Strong folder prior from scraper categories
    facial_keywords = [
        "face", "hair", "grooming", "beard", "skincare", "amiugly",
        "eyebrow", "facial", "glowup", "makeup", "headshot", "portrait"
    ]
    if any(kw in p_str for kw in facial_keywords):
        return "Face_Grooming"

    # 2. VLM decision
    if "face" in f_str or "groom" in f_str:
        return "Face_Grooming"
    if "portrait" in s_str or "close" in s_str or "head" in s_str:
        return "Face_Grooming"

    return "Outfit"


# ──────────────────────────────────────────────────────────────────────────────
# Per-Image Processing Workflow
# ──────────────────────────────────────────────────────────────────────────────
def process_single_image(
    image_path: Path,
    engine: str,
    model_name: str,
    mlx_model: Any = None,
    mlx_processor: Any = None,
    mlx_config: Any = None,
    dry_run: bool = False,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Executes end-to-end qualification, VLM inference, and routing for one image.
    """
    record: Dict[str, Any] = {
        "file": str(image_path),
        "filename": image_path.name,
        "source_category": image_path.parent.name,
        "heuristic_passed": False,
        "heuristic_reason": "",
        "vlm_result": None,
        "destination": None,
        "status": "pending",
        "timestamp": time.time(),
    }

    # Step 1: Heuristic Quality Filter
    passed_heuristics, reason = heuristic_filter(image_path)
    record["heuristic_passed"] = passed_heuristics
    record["heuristic_reason"] = reason

    if not passed_heuristics:
        record["status"] = f"rejected_heuristic_{reason}"
        dest_dir = REJECTED_DIR / "blurry_or_low_res"
        record["destination"] = str(dest_dir / image_path.name)
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(image_path), str(dest_dir / image_path.name))
            except Exception as e:
                logger.debug("Failed moving rejected file: %s", e)
        return record

    # Step 2: VLM Classification
    annotation: Optional[Dict[str, Any]] = None
    if engine == "ollama":
        annotation = classify_with_ollama(image_path, model=model_name)
    elif engine == "gemini":
        annotation = classify_with_gemini(image_path, model=model_name, api_key=api_key)
    elif engine == "mlx_vlm" and mlx_model is not None:
        annotation = classify_with_mlx_vlm(image_path, mlx_processor, mlx_model, mlx_config)

    record["vlm_result"] = annotation

    if not annotation:
        record["status"] = "rejected_vlm_failed"
        dest_dir = REJECTED_DIR / "vlm_failed"
        record["destination"] = str(dest_dir / image_path.name)
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(image_path), str(dest_dir / image_path.name))
            except Exception as e:
                logger.debug("Failed moving vlm-failed file: %s", e)
        return record

    # Step 3: Human look qualification
    is_valid_human = annotation.get("is_valid_human_look", False)
    if not is_valid_human:
        record["status"] = "rejected_not_human"
        dest_dir = REJECTED_DIR / "non_human_or_flatlay"
        record["destination"] = str(dest_dir / image_path.name)
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(image_path), str(dest_dir / image_path.name))
            except Exception as e:
                logger.debug("Failed moving non-human file: %s", e)
        return record

    # Step 4: Demographic & Tier Routing
    gender = annotation.get("gender", "")
    age_bracket = annotation.get("age_bracket", "")
    tier_raw = annotation.get("aesthetic_tier", "")
    score_raw = annotation.get("overall_score")
    score_val = float(score_raw) if score_raw is not None else None

    demographic = resolve_demographic(gender, age_bracket)
    tier = resolve_tier(tier_raw, score_val)

    if not demographic or not tier:
        record["status"] = "rejected_routing_failed"
        dest_dir = REJECTED_DIR / "routing_failed"
        record["destination"] = str(dest_dir / image_path.name)
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(image_path), str(dest_dir / image_path.name))
            except Exception as e:
                logger.debug("Failed moving unroutable file: %s", e)
        return record

    # Step 5: Dual-Stream Resolution (Outfit vs Face_Grooming)
    focus_type = annotation.get("focus_type")
    shot_type = annotation.get("shot_type")
    stream = resolve_stream(focus_type, shot_type, image_path)

    # Step 6: Route to Training Directory (TRAINING_DATA_DIR / stream / demographic / tier)
    dest_dir = TRAINING_DATA_DIR / stream / demographic / tier
    dest_path = dest_dir / image_path.name
    record["destination"] = str(dest_path)
    record["stream"] = stream
    record["demographic"] = demographic
    record["tier"] = tier
    record["status"] = "sorted"

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(image_path), str(dest_path))
        except Exception as e:
            logger.error("Failed moving sorted file to %s: %s", dest_path, e)

    return record


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing & Distribution Metrics
# ──────────────────────────────────────────────────────────────────────────────
def load_processed_files(annotations_path: Path) -> Set[str]:
    """
    Load set of already processed file names to allow automatic resumption:
      1. Reads all filenames from dataset_annotations.jsonl
      2. Scans 3_CoreML_Training_Data/ and filtered_rejected/ for existing images
    """
    processed = set()

    # 1. Check annotation logs
    if annotations_path.exists():
        try:
            with open(annotations_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            f_path = entry.get("file") or entry.get("filename")
                            if f_path:
                                processed.add(Path(f_path).name)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning("Could not read existing annotations: %s", e)

    # 2. Check destination directories on disk
    for root_dir in [TRAINING_DATA_DIR, REJECTED_DIR]:
        if root_dir.exists():
            for f in root_dir.rglob("*"):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
                    processed.add(f.name)

    return processed


def append_annotation_record(record: Dict[str, Any], annotations_path: Path) -> None:
    """Safely append single annotation record to JSONL."""
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with open(annotations_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_dataset_distribution() -> None:
    """Inspect and print the current distribution of images across both training streams."""
    print("\n" + "═" * 74)
    print("  LookMax — CoreML Training Dataset Distribution (Dual-Stream)")
    print("═" * 74)

    grand_total = 0

    for stream in ["Outfit", "Face_Grooming"]:
        stream_dir = TRAINING_DATA_DIR / stream
        stream_total = 0
        print(f"\n  ══════════════════ 🌟 STREAM: {stream.upper()} ══════════════════")

        for demo in DEMOGRAPHICS:
            print(f"\n  📁 {demo}")
            for tier in AESTHETIC_TIERS:
                tier_dir = stream_dir / demo / tier
                count = 0
                if tier_dir.exists():
                    count = sum(
                        1
                        for f in tier_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith(".")
                    )
                stream_total += count
                grand_total += count
                bar = "█" * min(25, count // 10)
                print(f"     ├── {tier:<20} : {count:>5} images  {bar}")

        print(f"\n  📊 Subtotal for {stream}: {stream_total:>5} images")

    print("\n" + "─" * 74)
    print(f"  🌟 Grand Total Qualified Images in 3_CoreML_Training_Data: {grand_total:>6}")
    print("═" * 74 + "\n")


def rollback_sorting() -> None:
    """
    Rolls back all sorted and rejected images back to their original 1_Raw_Scrapes/ paths,
    and archives the dataset_annotations.jsonl log.
    """
    print("\n" + "═" * 70)
    print("  LookMax — Rolling Back Sorted & Rejected Images to 1_Raw_Scrapes")
    print("═" * 70)

    restored_count = 0
    file_map: Dict[str, Path] = {}

    # 1. Parse annotations log to map filenames back to their original scrape folder
    if ANNOTATIONS_FILE.exists():
        try:
            with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            dest = entry.get("destination")
                            orig = entry.get("file")
                            if dest and orig:
                                file_map[Path(dest).name] = Path(orig)
                        except Exception:
                            pass
        except Exception as e:
            logger.warning("Could not read annotations file: %s", e)

    # 2. Move files back from 3_CoreML_Training_Data
    if TRAINING_DATA_DIR.exists():
        for f in list(TRAINING_DATA_DIR.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
                orig_path = file_map.get(f.name)
                if orig_path:
                    orig_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(orig_path))
                    restored_count += 1
                else:
                    fallback_dir = RAW_SCRAPES_DIR / "recovered_untracked"
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(fallback_dir / f.name))
                    restored_count += 1

    # 3. Move files back from filtered_rejected
    if REJECTED_DIR.exists():
        for f in list(REJECTED_DIR.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
                orig_path = file_map.get(f.name)
                if orig_path:
                    orig_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(orig_path))
                    restored_count += 1
                else:
                    fallback_dir = RAW_SCRAPES_DIR / "recovered_untracked"
                    fallback_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(fallback_dir / f.name))
                    restored_count += 1

    # 4. Archive annotations log
    if ANNOTATIONS_FILE.exists():
        backup_path = ANNOTATIONS_FILE.with_suffix(f".bak_{int(time.time())}.jsonl")
        shutil.move(str(ANNOTATIONS_FILE), str(backup_path))
        print(f"  📦 Archived annotations log → {backup_path.name}")

    print(f"  ✅ Successfully restored {restored_count} images back to {RAW_SCRAPES_DIR}")
    print("═" * 70 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="LookMax Phase 3 — High-Throughput AI Dataset Qualification & Sorting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--engine",
        choices=["ollama", "mlx_vlm", "gemini"],
        default=VLM_ENGINE,
        help="Vision-Language Model backend to use.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Custom model name for the selected engine (e.g. 'llava:7b', 'gemini-2.0-flash').",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Process only first N images (smoke testing mode).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=VLM_PARALLEL_WORKERS,
        help="Number of concurrent worker threads.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Classify and print decisions without moving files on disk.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Print current training dataset distribution and exit.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Undo all classification: move sorted & rejected images back to 1_Raw_Scrapes and archive annotations log.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY"),
        help="Gemini API Key (or set GEMINI_API_KEY environment variable).",
    )

    args = parser.parse_args()

    if args.reset:
        rollback_sorting()
        sys.exit(0)

    if args.stats_only:
        print_dataset_distribution()
        sys.exit(0)

    # Resolve active model name
    if args.model:
        model_name = args.model
    elif args.engine == "ollama":
        model_name = OLLAMA_MODEL
    elif args.engine == "gemini":
        model_name = GEMINI_MODEL
    else:
        model_name = MLX_MODEL_ID

    print(f"\n{'═' * 68}")
    print("  LookMax ML Pipeline — Phase 3: AI Qualification & Auto-Sorting")
    print(f"{'═' * 68}")
    print(f"  Engine      : {args.engine} (Model: {model_name})")
    print(f"  Workers     : {args.workers}")
    print(f"  Dry Run     : {args.dry_run}")
    print(f"  Raw Scrapes : {RAW_SCRAPES_DIR}")
    print(f"  Target Dir  : {TRAINING_DATA_DIR}")
    print(f"  Reject Dir  : {REJECTED_DIR}")
    print(f"  Annotations : {ANNOTATIONS_FILE}")
    print(f"{'═' * 68}\n")

    # Step 1: Collect raw images
    all_images: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(RAW_SCRAPES_DIR.rglob(f"*{ext}"))
        all_images.extend(RAW_SCRAPES_DIR.rglob(f"*{ext.upper()}"))

    all_images = [p for p in all_images if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    # Load already processed files for resumption
    processed_names = load_processed_files(ANNOTATIONS_FILE)
    if processed_names:
        initial_count = len(all_images)
        all_images = [p for p in all_images if p.name not in processed_names]
        skipped = initial_count - len(all_images)
        if skipped > 0:
            logger.info("Resuming: skipping %d previously processed images.", skipped)

    if args.sample:
        all_images = all_images[: args.sample]

    print(f"Found {len(all_images)} images ready to classify.\n")

    if not all_images:
        print("✅ No pending images found to process. All images have been sorted or filtered.")
        print_dataset_distribution()
        return

    # Step 2: Initialize MLX model if requested
    mlx_model = mlx_processor = mlx_config = None
    if args.engine == "mlx_vlm":
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config

            logger.info("Loading MLX-VLM model: %s on Apple Silicon...", model_name)
            mlx_model, mlx_processor = load(model_name)
            mlx_config = load_config(model_name)
            logger.info("MLX-VLM model loaded successfully.")
        except Exception as e:
            logger.error("Failed loading MLX-VLM (%s). Falling back to Ollama.", e)
            args.engine = "ollama"
            model_name = OLLAMA_MODEL

    # Step 3: Execute Classification
    stats: Dict[str, int] = defaultdict(int)
    sorted_by_demo: Dict[str, int] = defaultdict(int)
    sorted_by_tier: Dict[str, int] = defaultdict(int)

    start_time = time.time()
    use_threading = args.engine != "mlx_vlm" and args.workers > 1

    if use_threading:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_img = {
                executor.submit(
                    process_single_image,
                    img_path,
                    args.engine,
                    model_name,
                    None,
                    None,
                    None,
                    args.dry_run,
                    args.api_key,
                ): img_path
                for img_path in all_images
            }

            pbar = tqdm(total=len(all_images), unit="img", desc="Classifying") if HAS_TQDM else None

            for future in as_completed(future_to_img):
                record = future.result()
                status = record.get("status", "unknown")
                stats[status] += 1

                if status == "sorted":
                    sorted_by_demo[record.get("demographic", "unknown")] += 1
                    sorted_by_tier[record.get("tier", "unknown")] += 1

                if not args.dry_run:
                    append_annotation_record(record, ANNOTATIONS_FILE)

                if pbar:
                    pbar.update(1)
                else:
                    logger.info(
                        "[%s] -> %s (%s / %s)",
                        record["filename"],
                        status,
                        record.get("demographic", "-"),
                        record.get("tier", "-"),
                    )

            if pbar:
                pbar.close()
    else:
        # Sequential processing (MLX-VLM or 1 worker)
        iterator = tqdm(all_images, unit="img", desc="Classifying") if HAS_TQDM else all_images
        for img_path in iterator:
            record = process_single_image(
                img_path,
                args.engine,
                model_name,
                mlx_model,
                mlx_processor,
                mlx_config,
                args.dry_run,
                args.api_key,
            )
            status = record.get("status", "unknown")
            stats[status] += 1

            if status == "sorted":
                sorted_by_demo[record.get("demographic", "unknown")] += 1
                sorted_by_tier[record.get("tier", "unknown")] += 1

            if not args.dry_run:
                append_annotation_record(record, ANNOTATIONS_FILE)

            if not HAS_TQDM:
                logger.info(
                    "[%s] -> %s (%s / %s)",
                    record["filename"],
                    status,
                    record.get("demographic", "-"),
                    record.get("tier", "-"),
                )

    elapsed = time.time() - start_time
    throughput = len(all_images) / max(elapsed, 0.001)

    # Step 4: Summary Telemetry
    print("\n" + "═" * 68)
    print("  LookMax — Classification & Sorting Complete")
    print("═" * 68)
    print(f"  ⏱  Time Elapsed             : {elapsed:.1f}s ({throughput:.2f} images/sec)")
    print(f"  ✅ Qualified & Sorted       : {stats['sorted']}")
    print(f"  🚫 Rejected (Non-human)     : {stats['rejected_not_human']}")
    print(
        f"  🗑  Rejected (Heuristic/Blur): {sum(v for k, v in stats.items() if k.startswith('rejected_heuristic'))}"
    )
    print(f"  ⚠  VLM Inference Failures  : {stats['rejected_vlm_failed']}")
    print(f"  ❓ Routing Failures         : {stats['rejected_routing_failed']}")
    print("─" * 68)

    if sorted_by_demo:
        print("  Demographic Breakdown:")
        for demo, count in sorted(sorted_by_demo.items()):
            print(f"    • {demo:<20} : {count:>5} images")

    if sorted_by_tier:
        print("  Aesthetic Tier Breakdown:")
        for tier, count in sorted(sorted_by_tier.items()):
            print(f"    • {tier:<20} : {count:>5} images")

    print("═" * 68 + "\n")

    if not args.dry_run:
        print_dataset_distribution()


if __name__ == "__main__":
    main()
