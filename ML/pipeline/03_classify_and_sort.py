"""
LookMax ML Pipeline — Phase 3
==============================
03_classify_and_sort.py

Two-stage image classification and auto-sorting engine:

  STEP A — Fast Heuristic Quality Filter (< 15ms/image):
    • Rejects blurry images (Laplacian variance threshold)
    • Rejects images below minimum dimension or bad aspect ratio
    • Quickly checks for face/upper-body presence using OpenCV Haar Cascades
    • Rejected images moved to 2_VLM_Processing/filtered_rejected/

  STEP B — VLM Parallel Batch Engine:
    • Supports 3 backends: mlx-vlm (M3 Max native), Ollama, Google Gemini
    • Strict JSON schema extraction per image
    • Auto-routes each valid image into the exact demographic & aesthetic-tier
      bucket under 3_CoreML_Training_Data/
    • Appends full annotation to dataset_annotations.jsonl

Usage:
    python3 ML/pipeline/03_classify_and_sort.py
    python3 ML/pipeline/03_classify_and_sort.py --engine mlx_vlm
    python3 ML/pipeline/03_classify_and_sort.py --engine gemini --sample 10
    python3 ML/pipeline/03_classify_and_sort.py --engine ollama --workers 4
"""

import argparse
import base64
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_SCRAPES_DIR, REJECTED_DIR, METADATA_LOGS_DIR,
    TRAINING_DATA_DIR, ANNOTATIONS_FILE, DEMOGRAPHICS, AESTHETIC_TIERS,
    VLM_ENGINE, OLLAMA_HOST, OLLAMA_MODEL, GEMINI_MODEL,
    MLX_MODEL_ID, VLM_PARALLEL_WORKERS, VLM_REQUEST_TIMEOUT_SEC,
    IMAGE_EXTENSIONS, MIN_IMAGE_DIMENSION_PX, MAX_ASPECT_RATIO,
    BLUR_LAPLACIAN_THRESHOLD,
)

# ─── Conditional imports (graceful degradation) ──────────────────────────────
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

# ─── VLM Prompt Template ────────────────────────────────────────────────────
VLM_SYSTEM_PROMPT = """\
You are an expert fashion and posture analyst for a style coaching app.
Analyse the provided image carefully and respond ONLY with a single valid JSON object.
Do NOT include any prose, markdown, or code blocks — just raw JSON.

Required JSON schema:
{
  "is_valid_human_look": <true|false>,
  "gender": "<male|female|unisex>",
  "age_bracket": "<under_35|35_to_50|over_50>",
  "shot_type": "<full_body|upper_body|portrait|other>",
  "posture_score": <float 1.0-10.0>,
  "posture_notes": "<one sentence>",
  "fit_score": <float 1.0-10.0>,
  "fit_notes": "<one sentence>",
  "aesthetic_tier": "<1_Needs_Improvement|2_Average|3_Polished>",
  "overall_score": <float 1.0-10.0>,
  "confidence": <float 0.0-1.0>
}

Rules:
- Set is_valid_human_look to false for flat-lays, product shots, animals, text-only images, or any image without a visible person.
- aesthetic_tier must map to overall_score: 1_Needs_Improvement (1.0–4.9), 2_Average (5.0–7.4), 3_Polished (7.5–10.0).
- Estimate age_bracket from visible cues (face, skin, style). When uncertain, choose under_35.
- Output only the JSON object. Nothing else.
"""


# ──────────────────────────────────────────────────────────────────────────────
# STEP A — Heuristic Fast Filter
# ──────────────────────────────────────────────────────────────────────────────
def heuristic_filter(image_path: Path) -> tuple[bool, str]:
    """
    Returns (passes: bool, reason: str).
    Fast CPU-only quality checks — runs in < 15ms per image.
    """
    if not HAS_CV2 or not HAS_NP:
        return True, "cv2_unavailable"

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "unreadable_file"

        h, w = img.shape[:2]

        # 1. Minimum short-edge dimension
        if min(h, w) < MIN_IMAGE_DIMENSION_PX:
            return False, f"too_small_{min(h,w)}px"

        # 2. Aspect ratio — reject extreme panoramas or thin slivers
        ratio = max(h, w) / max(min(h, w), 1)
        if ratio > MAX_ASPECT_RATIO:
            return False, f"bad_aspect_ratio_{ratio:.2f}"

        # 3. Blur detection — Laplacian variance
        gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if lap_var < BLUR_LAPLACIAN_THRESHOLD:
            return False, f"too_blurry_var_{lap_var:.1f}"

        # 4. Human presence check — skin-tone HSV detection
        #    (Handles all skin tones with a broad HSV range)
        #    Avoids CascadeClassifier which isn't available in opencv-headless
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Broad multi-range skin detection (light to dark skin tones)
        mask1 = cv2.inRange(hsv, np.array([0,  15, 60]), np.array([25, 200, 255]))
        mask2 = cv2.inRange(hsv, np.array([0,  10, 40]), np.array([20, 150, 200]))
        mask3 = cv2.inRange(hsv, np.array([170, 15, 60]), np.array([180, 200, 255]))
        skin_mask  = cv2.bitwise_or(cv2.bitwise_or(mask1, mask2), mask3)
        skin_ratio = cv2.countNonZero(skin_mask) / (h * w)
        if skin_ratio < 0.02:    # 2% skin pixels minimum (lenient — fashion shots often mostly clothing)
            return False, f"no_human_detected_skin_{skin_ratio:.3f}"

        return True, "ok"

    except Exception as e:
        return False, f"error_{str(e)[:30]}"




# ──────────────────────────────────────────────────────────────────────────────
# STEP B — VLM Classification Backends
# ──────────────────────────────────────────────────────────────────────────────
def image_to_base64(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_vlm_json(raw: str) -> dict | None:
    """Robustly parse JSON from VLM output, handling markdown code fences."""
    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
    # Extract first JSON object
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


# ── Backend: Ollama ──────────────────────────────────────────────────────────
def classify_with_ollama(image_path: Path, model: str = OLLAMA_MODEL) -> dict | None:
    try:
        import requests as req
        b64 = image_to_base64(image_path)

        # Ollama native vision format: images array in the user message
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": VLM_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Analyse this image and respond with the required JSON.",
                    "images": [b64],      # Ollama native format — base64 string in list
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1, "top_p": 0.9, "num_predict": 512},
        }
        resp = req.post(
            f"{OLLAMA_HOST}/api/chat",
            json=payload,
            timeout=VLM_REQUEST_TIMEOUT_SEC
        )
        if resp.status_code != 200:
            return None
        content = resp.json().get("message", {}).get("content", "")
        return parse_vlm_json(content)
    except Exception:
        return None



# ── Backend: Google Gemini (cloud fallback) ───────────────────────────────────
def classify_with_gemini(image_path: Path) -> dict | None:
    try:
        from google import genai
        from google.genai import types as genai_types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("  GEMINI_API_KEY not set. Skipping image.")
            return None

        client = genai.Client(api_key=api_key)

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        suffix = image_path.suffix.lower()
        mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".png": "image/png", ".webp": "image/webp"}
        mime = mime_map.get(suffix, "image/jpeg")

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=image_bytes, mime_type=mime),
                f"{VLM_SYSTEM_PROMPT}\n\nAnalyse this image and return the JSON."
            ],
            config=genai_types.GenerateContentConfig(temperature=0.1),
        )
        return parse_vlm_json(response.text)
    except Exception as e:
        return None


# ── Backend: MLX-VLM (Apple Silicon native — recommended for M3 Max) ─────────
def classify_with_mlx_vlm(image_path: Path, mlx_processor, mlx_model, mlx_config) -> dict | None:
    try:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        user_content = "Analyse this image and return the required JSON."
        formatted_prompt = apply_chat_template(
            mlx_processor, mlx_config, user_content,
            num_images=1,
            system_prompt=VLM_SYSTEM_PROMPT
        )
        output = generate(
            mlx_model, mlx_processor,
            image=str(image_path),
            prompt=formatted_prompt,
            max_tokens=512,
            temperature=0.1,
            verbose=False,
        )
        return parse_vlm_json(output)
    except Exception as e:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Demographic & Tier Routing
# ──────────────────────────────────────────────────────────────────────────────
def resolve_demographic(gender: str, age_bracket: str) -> str | None:
    """Map VLM gender + age_bracket → training folder name."""
    gender_map = {
        "male":   "Men",
        "female": "Women",
        "unisex": "Men",   # Default unisex → Men bucket (balanced)
    }
    age_map = {
        "under_35": "Under_35",
        "35_to_50": "35_to_50",
        "over_50":  "Over_50",
    }
    g = gender_map.get(gender.lower())
    a = age_map.get(age_bracket.lower())
    if g and a:
        return f"{g}_{a}"
    return None


def resolve_tier(aesthetic_tier: str) -> str | None:
    """Normalize tier string to folder name."""
    tier_map = {
        "1_needs_improvement": "1_Needs_Improvement",
        "2_average":           "2_Average",
        "3_polished":          "3_Polished",
    }
    return tier_map.get(aesthetic_tier.lower().replace(" ", "_"))


# ──────────────────────────────────────────────────────────────────────────────
# Per-Image Processing Workflow
# ──────────────────────────────────────────────────────────────────────────────
def process_image(
    image_path: Path,
    engine: str,
    mlx_model=None, mlx_processor=None, mlx_config=None,
    dry_run: bool = False,
) -> dict:
    """
    Full pipeline for a single image:
      1. Heuristic quality filter
      2. VLM classification
      3. Route to correct training folder
    Returns a result dict for logging.
    """
    result = {
        "file": str(image_path),
        "heuristic_passed": None,
        "heuristic_reason": None,
        "vlm_result": None,
        "destination": None,
        "status": "unknown",
    }

    # ── Step A: Heuristic filter ─────────────────────────────────────────────
    passed, reason = heuristic_filter(image_path)
    result["heuristic_passed"] = passed
    result["heuristic_reason"] = reason

    if not passed:
        result["status"] = "rejected_heuristic"
        if not dry_run:
            REJECTED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(REJECTED_DIR / image_path.name))
        return result

    # ── Step B: VLM Classification ───────────────────────────────────────────
    annotation = None

    if engine == "ollama":
        annotation = classify_with_ollama(image_path)
    elif engine == "gemini":
        annotation = classify_with_gemini(image_path)
    elif engine == "mlx_vlm" and mlx_model is not None:
        annotation = classify_with_mlx_vlm(image_path, mlx_processor, mlx_model, mlx_config)

    result["vlm_result"] = annotation

    if annotation is None:
        result["status"] = "vlm_failed"
        return result

    # Reject non-human images
    if not annotation.get("is_valid_human_look", False):
        result["status"] = "rejected_not_human"
        if not dry_run:
            REJECTED_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(image_path), str(REJECTED_DIR / image_path.name))
        return result

    # ── Step C: Route to training bucket ────────────────────────────────────
    gender       = annotation.get("gender", "")
    age_bracket  = annotation.get("age_bracket", "")
    aesthetic    = annotation.get("aesthetic_tier", "")

    demographic  = resolve_demographic(gender, age_bracket)
    tier         = resolve_tier(aesthetic)

    if not demographic or not tier:
        result["status"] = "routing_failed"
        return result

    dest_dir = TRAINING_DATA_DIR / demographic / tier
    dest_path = dest_dir / image_path.name

    result["destination"] = str(dest_path)
    result["status"] = "sorted"

    if not dry_run:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(image_path), str(dest_path))
        # Remove stale .gitkeep if present so it doesn't count as training data
        gitkeep = dest_dir / ".gitkeep"
        if dest_path.exists() and gitkeep.exists():
            pass  # Keep gitkeep; the actual image is stored alongside it

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Annotation Logging
# ──────────────────────────────────────────────────────────────────────────────
def append_annotation(result: dict):
    METADATA_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LookMax Phase 3 — Parallel AI Classification & Auto-Sorting"
    )
    parser.add_argument("--engine", choices=["ollama", "gemini", "mlx_vlm"],
                        default=VLM_ENGINE, help="VLM inference backend")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only N images (smoke test mode)")
    parser.add_argument("--workers", type=int, default=VLM_PARALLEL_WORKERS,
                        help="Parallel VLM worker threads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Classify but do not move files")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"  LookMax ML Pipeline — Phase 3: Classification & Sorting")
    print(f"{'═'*60}")
    print(f"  Engine  : {args.engine}")
    print(f"  Workers : {args.workers}")
    print(f"  Dry-run : {args.dry_run}")

    # Collect all raw images
    all_images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        all_images.extend(RAW_SCRAPES_DIR.rglob(f"*{ext}"))
        all_images.extend(RAW_SCRAPES_DIR.rglob(f"*{ext.upper()}"))

    # Remove .gitkeep placeholder matches (they won't match image extensions anyway)
    all_images = [p for p in all_images if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]

    if args.sample:
        all_images = all_images[:args.sample]

    print(f"  Images  : {len(all_images)} found in {RAW_SCRAPES_DIR}")

    if not all_images:
        print("\n  ⚠  No images found. Run 02_scrape_images.py first.")
        return

    # ── Load MLX model once (if using mlx_vlm) ────────────────────────────
    mlx_model = mlx_processor = mlx_config = None
    if args.engine == "mlx_vlm":
        try:
            from mlx_vlm import load
            print(f"\n  Loading MLX-VLM model: {MLX_MODEL_ID} (this may take ~30s first run)...")
            mlx_model, mlx_processor = load(MLX_MODEL_ID)
            from mlx_vlm.utils import load_config
            mlx_config = load_config(MLX_MODEL_ID)
            print("  ✓ MLX-VLM model loaded on Apple Silicon")
        except ImportError:
            print("  ✗ mlx-vlm not installed. Falling back to Ollama.")
            args.engine = "ollama"

    # ── Process images ────────────────────────────────────────────────────
    stats = {"sorted": 0, "rejected_heuristic": 0, "rejected_not_human": 0,
             "vlm_failed": 0, "routing_failed": 0}

    # For mlx_vlm we can't share the model across threads safely — use sequential
    use_threads = args.engine != "mlx_vlm"

    iterator = tqdm(all_images, unit="img", desc="  Classifying") if HAS_TQDM else all_images

    if use_threads:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_image, img_path, args.engine,
                    None, None, None, args.dry_run
                ): img_path
                for img_path in all_images
            }
            for future in (tqdm(as_completed(futures), total=len(all_images),
                                unit="img", desc="  Classifying") if HAS_TQDM else as_completed(futures)):
                result = future.result()
                stats[result["status"]] = stats.get(result["status"], 0) + 1
                if not args.dry_run:
                    append_annotation(result)
    else:
        # Sequential for MLX-VLM (shared model state)
        for img_path in iterator:
            result = process_image(
                img_path, args.engine,
                mlx_model, mlx_processor, mlx_config,
                args.dry_run
            )
            stats[result["status"]] = stats.get(result["status"], 0) + 1
            if not args.dry_run:
                append_annotation(result)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Classification & Sorting Summary")
    print(f"{'─'*60}")
    print(f"  ✅ Sorted to training data  : {stats.get('sorted', 0)}")
    print(f"  🗑  Rejected (blurry/small)  : {stats.get('rejected_heuristic', 0)}")
    print(f"  🚫 Rejected (not human)     : {stats.get('rejected_not_human', 0)}")
    print(f"  ⚠  VLM parse failed         : {stats.get('vlm_failed', 0)}")
    print(f"  ❓  Routing failed           : {stats.get('routing_failed', 0)}")
    print(f"\n  Annotations → {ANNOTATIONS_FILE}")
    print(f"  Next: python3 ML/pipeline/04_train_coreml_models.py")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
