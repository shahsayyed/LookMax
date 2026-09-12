#!/usr/bin/env python3
"""
vertex_batch_classify.py — High-Throughput Cloud VLM Qualification, Auditing & Attribute Extraction.
====================================================================================================
Submits all real images (both existing sorted in 3_CoreML_Training_Data and pending
raw web scrapes in 1_Raw_Scrapes) to Vertex AI Batch Prediction using Gemini 3.8 / 2.5 Flash.

Key Capabilities:
  1. Human Look Qualification & Rejection:
     - Automatically flags and isolates flat-lays, hanger-only shots, memes, scenery, and shoe closeups.
     - Routes raw web images to 3_CoreML_Training_Data if valid, or filtered_rejected/ if invalid.
     - Quarantines any invalid images found in existing training data.
  2. Continuous Calibration & Demographics:
     - Calibrated continuous overall_score (1.0 to 10.0), fit_score, posture_score, and aesthetic_tier.
     - Resolves demographic (Men/Women × Under_35/35_to_50/Over_50) and stream (Outfit vs Face_Grooming).
  3. Master Styling Taxonomy:
     - Core garment classes: upper_type, mid_type, lower_type, footwear_type, formality.
     - Actionable styling execution: tuck_style, pant_break, sleeve_styling, outerwear_buttoning,
       layering_count, color_harmony, accessories.
     - Grooming precision: hair_styled, hair_length, hair_texture, fade_or_taper_quality,
       facial_hair_style, beard_neckline_sharpness, eyebrow_grooming, makeup_style.
  4. Resumable Batch Pipeline:
     - Chunks images (default 500/job) and submits via sliding-window concurrency (--max-concurrent 8).
     - Native GCS upload/download via google.cloud.storage.
     - Per-chunk local cache (.vertex_batch_cache/) so interrupted runs resume instantly without re-billing.

Usage:
  # Dry run (checks image paths, builds chunks without cloud submission)
  python3 ML/vision/dataset_real/vertex_batch_classify.py --dry-run

  # Smoke test (processes first 50 images in a single batch)
  python3 ML/vision/dataset_real/vertex_batch_classify.py --sample 50

  # Full dataset run
  python3 ML/vision/dataset_real/vertex_batch_classify.py
"""
import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TRAINING_DATA_DIR = REPO_ROOT / "ML" / "data" / "vision_real" / "3_CoreML_Training_Data"
RAW_SCRAPES_DIR = REPO_ROOT / "ML" / "data" / "vision_real" / "1_Raw_Scrapes"
REJECTED_DIR = REPO_ROOT / "ML" / "data" / "vision_real" / "2_VLM_Processing" / "filtered_rejected"
ANNOTATIONS_FILE = REPO_ROOT / "ML" / "data" / "vision_real" / "2_VLM_Processing" / "metadata_logs" / "dataset_annotations.jsonl"
CREDENTIALS_FILE = REPO_ROOT / "lookmax-generation-513e3f9ab69e.json"
CHUNKS_CACHE_DIR = SCRIPT_DIR / ".vertex_batch_cache"

GCS_BUCKET = "lookmax-generation-batch-staging"
DEFAULT_MODEL = "gemini-3.8-flash"
LOCATION = "global"

DEMOGRAPHICS = [
    "Men_Under_35", "Men_35_to_50", "Men_Over_50",
    "Women_Under_35", "Women_35_to_50", "Women_Over_50",
]
AESTHETIC_TIERS = ["1_Needs_Improvement", "2_Average", "3_Polished"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".avif"}

# ─── Master LookMax VLM Prompt ───────────────────────────────────────────────
MASTER_VLM_SYSTEM_PROMPT = """\
You are an expert fashion, outfit, facial aesthetics, and grooming analyst for the LookMax style coaching platform.
Analyze the image carefully and respond ONLY with a raw JSON object (no markdown, no prose, no code fences).

Required JSON schema:
{
  "is_valid_human_look": <true or false>,
  "rejection_reason": <"flat_lay" | "hanger_only" | "scenery_no_human" | "shoe_closeup" | "meme_or_text" | null>,
  "gender": <"male" or "female">,
  "age_bracket": <"under_35", "35_to_50", or "over_50">,
  "focus_type": <"outfit" or "face_grooming">,
  "shot_type": <"full_body", "upper_body", "portrait", or "other">,
  "posture_score": <float 1.0 to 10.0>,
  "fit_score": <float 1.0 to 10.0>,
  "overall_score": <float 1.0 to 10.0>,
  "aesthetic_tier": <"1_Needs_Improvement", "2_Average", or "3_Polished">,
  "confidence": <float 0.0 to 1.0>,
  "summary_critique": "<one clear sentence explaining the rating>",

  "upper_type": <string or null>,
  "mid_type": <string or null>,
  "lower_type": <string or null>,
  "footwear_type": <string or null>,
  "formality": <string or null>,

  "tuck_style": <"untucked" | "full_tuck" | "french_half_tuck" | "cropped_not_tucked" | "sloppy_half_out" | null>,
  "pant_break": <"no_break_tailored" | "slight_break" | "medium_break" | "full_pooling_dragging" | "cropped_ankle" | null>,
  "sleeve_styling": <"cuffed_rolled" | "straight_down" | "sleeveless" | "too_long_covering_hands" | null>,
  "outerwear_buttoning": <"unbuttoned" | "top_only_correct" | "all_buttoned_incorrect" | "zipped" | "na" | null>,
  "layering_count": <integer 1 to 4 or null>,
  "color_harmony": <"monochrome" | "neutral_minimalist" | "complementary_contrast" | "earth_tones" | "clashing_mismatch" | null>,
  "accessories": <list of zero or more from ["watch", "belt", "necklace_chain", "rings", "sunglasses_glasses", "hat_cap", "bag"] or []>,

  "hair_styled": <"0" or "1" or null>,
  "hair_length": <"buzz_cut" | "short" | "medium" | "long" | null>,
  "hair_texture": <"straight" | "wavy" | "curly" | "coily" | "bald_buzzed" | null>,
  "fade_or_taper_quality": <"crisp_skin_fade" | "clean_taper" | "natural_scissor_blend" | "overgrown_no_fade" | "na" | null>,
  "facial_hair_style": <"clean_shaven" | "stubble" | "short_beard" | "full_beard" | "moustache" | null>,
  "beard_neckline_sharpness": <"crisp_lineup" | "natural_clean" | "patchy_unkempt_neckbeard" | "na" | null>,
  "eyebrow_grooming": <"natural_neat" | "defined_arched" | "unibrow_unkempt" | "overplucked_thin" | null>,
  "makeup_style": <"none" | "minimal" | "everyday" | "full" | null>,

  "inferred_occasion": <"casual_street" | "gym_athletic" | "smart_casual_office" | "business_formal" | "evening_date" | "lounge_home" | null>
}

CRITICAL RULES:
1. is_valid_human_look:
   - Set FALSE if: clothing items on hangers/racks, flat-lays on bed/floor, spreadsheets/text tables/screenshots, restaurant/cafe/room scenery, shoe-only closeups, fabric detail/stain shots, memes, or any image without an active person wearing the outfit or showing their face/hair. Set rejection_reason accordingly.
   - Set TRUE ONLY IF a real person is actively wearing an outfit, showing a haircut/grooming look, or posing for a fit check/posture evaluation.
2. focus_type:
   - "face_grooming" if the image focuses on the head, haircut, beard, skin, facial aesthetics, or upper-chest portrait / selfie.
   - "outfit" if the image showcases full-body or 3/4-body clothing silhouette, coordination, and styling.
3. aesthetic_tier & overall_score calibration (DO NOT default to 3_Polished):
   - FOR OUTFITS:
     • "1_Needs_Improvement" (1.0 to 4.9): Ill-fitting/wrinkled/stained clothes, excessive bagginess/sagging, dragging hems, sloppy execution, poor slouching posture, or mismatched garments.
     • "2_Average" (5.0 to 7.4): Standard daily casual wear, basic tee & jeans, ordinary unstyled hoodies, typical office casual, normal posture. MOST daily outfits belong here.
     • "3_Polished" (7.5 to 10.0): Exceptionally sharp tailoring, cohesive color harmony, sophisticated layering, confident upright posture, high-end modern streetwear, or formal elegance.
   - FOR FACE & GROOMING:
     • "1_Needs_Improvement" (1.0 to 4.9): Messy unstyled bedhead hair, overgrown/patchy untrimmed beard, severe redness/tired eyes, unflattering camera angle/lighting.
     • "2_Average" (5.0 to 7.4): Clean daily grooming baseline, standard neat haircut, natural everyday shave, neutral daily lighting.
     • "3_Polished" (7.5 to 10.0): Crisp styled hair/fade, sharp beard lines, glowing skin clarity, harmonious facial presentation.
4. CORE RATING PHILOSOPHY (Effort & Execution ONLY):
   - Rate styling, fit, and grooming execution ONLY.
   - Strictly FORBIDDEN to penalize or grade based on: body weight/size, facial symmetry/features, age, or medical skin conditions (acne).
5. AVOID LENIENCY BIAS (Crucial for dataset balance):
   - Do NOT default to "2_Average" or "3_Polished" out of politeness.
   - If an outfit has poorly proportioned silhouettes, dragging hems, wrinkled/sloppy fabric, bunching or pulling seams, or uncoordinated garments, you MUST assign "1_Needs_Improvement" (1.0 to 4.9).
6. age_bracket:
   - "under_35" for young adults, college students, 20s to early 30s.
   - "35_to_50" for 30s to 40s.
   - "over_50" for mature adults and seniors.
7. EXACT TAXONOMY MATCHES FOR GARMENTS:
   - upper_type: pick best match from ["tank_top", "graphic_tee", "plain_crewneck_tee", "casual_camisole", "henley", "polo_shirt", "flannel_shirt", "casual_blouse", "wrap_top", "silk_blouse", "fitted_sweater", "crewneck_sweater", "turtleneck_sweater", "oxford_button_down", "tailored_blouse", "dress_shirt", "dress"] (or null if obscured).
   - mid_type: pick best match from ["none", "hoodie", "denim_jacket", "bomber_jacket", "cardigan", "blazer", "cropped_jacket"].
   - lower_type: pick best match from ["sweatpants", "athletic_shorts", "cargo_shorts", "leggings", "denim_shorts", "denim_jeans", "joggers", "cargo_pants", "chino_pants", "corduroy_pants", "casual_skirt", "wide_leg_trousers", "midi_skirt", "tailored_trousers", "pencil_skirt", "dress_pants", "none"].
   - footwear_type: pick best match from ["slides", "flip_flops", "canvas_sneakers", "running_shoes", "leather_sneakers", "sneakers", "ballet_flats", "ankle_boots", "block_heels", "pointed_flats", "heeled_pumps", "suede_desert_boots", "leather_dress_shoes", "oxford_shoes"] (or null if not visible).
   - formality: pick from ["casual", "smart_casual", "business_casual", "formal"].
"""


def normalize_vlm_attributes(data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize VLM attribute outputs to match LookMax schema class vocabularies."""
    if not isinstance(data, dict):
        return data

    upper = str(data.get("upper_type") or "").lower().strip()
    if upper in ["suit_jacket", "blazer", "suit"]:
        if not data.get("mid_type") or data.get("mid_type") == "none":
            data["mid_type"] = "blazer"
        data["upper_type"] = "dress_shirt"
    elif upper in ["t-shirt", "tee", "tshirt"]:
        data["upper_type"] = "plain_crewneck_tee"
    elif upper in ["button_down", "button_up", "oxford"]:
        data["upper_type"] = "oxford_button_down"
    elif upper in ["blouse"]:
        data["upper_type"] = "casual_blouse"

    mid = str(data.get("mid_type") or "").lower().strip()
    if mid in ["suit_pants", "pants", "trousers"]:
        if not data.get("lower_type"):
            data["lower_type"] = "tailored_trousers"
        data["mid_type"] = "none"
    elif mid in ["suit_jacket", "suit_coat", "sport_coat"]:
        data["mid_type"] = "blazer"

    lower = str(data.get("lower_type") or "").lower().strip()
    if lower in ["suit_pants", "dress_pants", "slacks", "tailored_pants", "trouser", "trousers", "suit"]:
        data["lower_type"] = "tailored_trousers"
    elif lower in ["jeans", "blue_jeans"]:
        data["lower_type"] = "denim_jeans"
    elif lower in ["shorts"]:
        data["lower_type"] = "athletic_shorts"
    elif lower in ["chinos", "khakis"]:
        data["lower_type"] = "chino_pants"
    elif lower in ["skirt"]:
        data["lower_type"] = "casual_skirt"

    footwear = str(data.get("footwear_type") or "").lower().strip()
    if footwear in ["dress_shoes", "loafers", "monk_straps", "derbies", "oxfords"]:
        data["footwear_type"] = "leather_dress_shoes"
    elif footwear in ["sneakers", "tennis_shoes", "trainers"]:
        data["footwear_type"] = "canvas_sneakers"
    elif footwear in ["heels", "pumps", "high_heels"]:
        data["footwear_type"] = "heeled_pumps"
    elif footwear in ["boots"]:
        data["footwear_type"] = "ankle_boots"
    elif footwear in ["flats"]:
        data["footwear_type"] = "ballet_flats"

    styled = data.get("hair_styled")
    if styled is True or styled == "true" or styled == "1" or styled == 1:
        data["hair_styled"] = "1"
    elif styled is False or styled == "false" or styled == "0" or styled == 0:
        data["hair_styled"] = "0"

    facial = str(data.get("facial_hair_style") or "").lower().strip()
    if "shaven" in facial or "clean" in facial or facial == "none":
        data["facial_hair_style"] = "clean_shaven"
    elif "stubble" in facial:
        data["facial_hair_style"] = "stubble"
    elif "beard" in facial:
        data["facial_hair_style"] = "short_beard" if "short" in facial else "full_beard"

    return data


def resolve_demographic(gender: str, age_bracket: str) -> Optional[str]:
    g_str = str(gender).strip().lower()
    a_str = str(age_bracket).strip().lower()
    prefix = "Women" if ("female" in g_str or "woman" in g_str or "women" in g_str) else "Men"
    if "under" in a_str or "u35" in a_str or "young" in a_str or "20" in a_str:
        suffix = "Under_35"
    elif "over" in a_str or "o50" in a_str or "senior" in a_str or "mature" in a_str or "60" in a_str:
        suffix = "Over_50"
    else:
        suffix = "35_to_50"
    resolved = f"{prefix}_{suffix}"
    return resolved if resolved in DEMOGRAPHICS else None


def resolve_tier(aesthetic_tier: str, overall_score: Optional[float] = None) -> str:
    t_str = str(aesthetic_tier).strip().lower().replace(" ", "_")
    if any(k in t_str for k in ["1", "needs", "improve", "low", "poor", "bad"]):
        return "1_Needs_Improvement"
    elif any(k in t_str for k in ["3", "polish", "high", "sartorial", "good", "great", "elegan"]):
        return "3_Polished"
    elif overall_score is not None:
        if overall_score >= 7.5:
            return "3_Polished"
        elif overall_score < 5.0:
            return "1_Needs_Improvement"
    return "2_Average"


def resolve_stream(focus_type: Optional[str], shot_type: Optional[str], path_str: str) -> str:
    p_lower = path_str.lower()
    facial_keywords = ["face", "hair", "grooming", "beard", "skincare", "amiugly", "eyebrow", "facial", "makeup", "portrait"]
    if any(kw in p_lower for kw in facial_keywords):
        return "Face_Grooming"
    f_str = str(focus_type or "").lower()
    s_str = str(shot_type or "").lower()
    if "face" in f_str or "groom" in f_str or "portrait" in s_str or "head" in s_str:
        return "Face_Grooming"
    return "Outfit"


def get_clients() -> Tuple[Any, Any, str]:
    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(f"Missing service account credentials: {CREDENTIALS_FILE}")
    with open(CREDENTIALS_FILE) as f:
        project_id = json.load(f)["project_id"]
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_FILE.resolve())

    from google import genai
    from google.genai import types
    from google.cloud import storage

    client = genai.Client(vertexai=True, project=project_id, location=LOCATION,
                           http_options=types.HttpOptions(timeout=45000))
    storage_client = storage.Client.from_service_account_json(str(CREDENTIALS_FILE))
    return client, storage_client, project_id


def build_request_line(key: str, image_path: Path) -> dict:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((768, 768), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "key": key,
        "request": {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": img_b64}},
                    {"text": f"{MASTER_VLM_SYSTEM_PROMPT}\n\nAnalyze this image and return the required JSON object:"},
                ],
            }],
            "generation_config": {"temperature": 0.1, "response_mime_type": "application/json"},
        },
    }


def parse_vlm_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            return normalize_vlm_attributes(data)
    except Exception:
        try:
            fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
            data = json.loads(fixed)
            if isinstance(data, dict):
                return normalize_vlm_attributes(data)
        except Exception:
            return None
    return None


def collect_chunk_results(job, storage_client: Any, chunk_label: str) -> List[Dict[str, Any]]:
    if "SUCCEEDED" not in str(job.state):
        raise RuntimeError(f"[{chunk_label}] job did not succeed: state={job.state}")

    dest_uri = job.dest.gcs_uri if job.dest else ""
    if not dest_uri:
        raise RuntimeError(f"[{chunk_label}] missing dest GCS URI")

    bucket_name = dest_uri.replace("gs://", "").split("/")[0]
    prefix = "/".join(dest_uri.replace("gs://", "").split("/")[1:])
    bucket = storage_client.bucket(bucket_name)

    results = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith(".jsonl"):
            lines = blob.download_as_text().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = rec.get("key")
                try:
                    text = rec["response"]["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = parse_vlm_json(text)
                except Exception:
                    parsed = None
                results.append({"filename": key, "vlm_result": parsed, "raw_error": None if parsed else "parse_failure"})

    return results


def main():
    parser = argparse.ArgumentParser(description="Vertex AI Batch Prediction classifier & auditor for LookMax real images.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Number of images per batch job (default 500).")
    parser.add_argument("--max-concurrent", type=int, default=8, help="Max concurrent batch jobs in flight (default 8).")
    parser.add_argument("--poll-interval", type=int, default=25, help="Polling interval in seconds (default 25).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model ID (default gemini-3.8-flash).")
    parser.add_argument("--sample", type=int, default=None, help="Process first N images only (smoke test).")
    parser.add_argument("--dry-run", action="store_true", help="Build and validate chunks without calling Vertex AI.")
    parser.add_argument("--label", type=str, default="real_audit", help="Run label for staging and caching.")
    args = parser.parse_args()

    print("\n" + "═" * 74)
    print("  LookMax ML Pipeline — Vertex AI Batch Prediction Classifier & Auditor")
    print("═" * 74)
    print(f"  Model           : {args.model}")
    print(f"  GCS Bucket      : gs://{GCS_BUCKET}")
    print(f"  Chunk Size      : {args.chunk_size} images/job")
    print(f"  Max Concurrent  : {args.max_concurrent}")
    print(f"  Dry Run         : {args.dry_run}")
    print("─" * 74)

    # 1. Discover existing processed records to allow resumption
    existing_annotated: Set[str] = set()
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        v = d.get("vlm_result") or {}
                        if v.get("upper_type") or v.get("hair_length") or v.get("tuck_style") or v.get("hair_texture"):
                            existing_annotated.add(d.get("filename"))
                    except Exception:
                        pass
        print(f"  Existing annotations with master attributes: {len(existing_annotated)}")

    # 2. Collect images to process
    raw_images: List[Tuple[str, Path, str]] = []
    for f in RAW_SCRAPES_DIR.glob("web_*/*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
            if f.name not in existing_annotated:
                raw_images.append((f.name, f, "raw_scrape"))

    sorted_images: List[Tuple[str, Path, str]] = []
    for f in TRAINING_DATA_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS and not f.name.startswith("."):
            if f.name not in existing_annotated:
                sorted_images.append((f.name, f, "existing_sorted"))

    all_items = raw_images + sorted_images
    if args.sample:
        all_items = all_items[: args.sample]

    print(f"  Pending Raw Scrapes to Qualify & Route : {len(raw_images):>6}")
    print(f"  Existing Sorted Images to Enrich/Audit  : {len(sorted_images):>6}")
    print(f"  Total Images to Process in this Run    : {len(all_items):>6}")
    print("═" * 74 + "\n")

    if not all_items:
        print("✅ All images are already annotated with master styling attributes. Nothing to do.")
        return

    chunks = [all_items[i : i + args.chunk_size] for i in range(0, len(all_items), args.chunk_size)]
    print(f"Divided into {len(chunks)} chunk(s) of up to {args.chunk_size} images each.")

    if args.dry_run:
        print("\n[DRY-RUN] Sample request payload verification:")
        fn, p, st = all_items[0]
        sample_req = build_request_line(fn, p)
        print(f"  • Key: {sample_req['key']}")
        print(f"  • Parts: {len(sample_req['request']['contents'][0]['parts'])} (image + prompt)")
        print(f"  • Inline image payload length: {len(sample_req['request']['contents'][0]['parts'][0]['inline_data']['data'])} chars")
        print("\n[DRY-RUN] Chunks ready for submission. Exiting without Vertex API calls.")
        return

    # 3. Initialize Cloud Clients
    client, storage_client, project_id = get_clients()
    bucket = storage_client.bucket(GCS_BUCKET)
    CHUNKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    pending_idxs: List[int] = []

    for i in range(len(chunks)):
        cache_path = CHUNKS_CACHE_DIR / f"{args.label}_chunk{i:04d}.json"
        if cache_path.exists():
            chunk_res = json.loads(cache_path.read_text())
            print(f"  [Chunk {i+1}/{len(chunks)}] Already cached ({len(chunk_res)} results) — skipping submission.")
            all_results.extend(chunk_res)
        else:
            pending_idxs.append(i)

    def submit_chunk(chunk_idx: int) -> Any:
        from google.genai import types

        chunk = chunks[chunk_idx]
        chunk_label = f"{args.label}_c{chunk_idx:04d}"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        gcs_prefix = f"batch_classify/{chunk_label}_{ts}"
        local_jsonl = SCRIPT_DIR / f".batch_in_{chunk_label}_{ts}.jsonl"

        with open(local_jsonl, "w", encoding="utf-8") as f:
            for fn, p, st in chunk:
                try:
                    f.write(json.dumps(build_request_line(fn, p)) + "\n")
                except Exception as e:
                    print(f"    ⚠ Skipped unreadable image {fn}: {e}")

        # Upload to GCS via google.cloud.storage
        blob = bucket.blob(f"{gcs_prefix}/input.jsonl")
        blob.upload_from_filename(str(local_jsonl))
        local_jsonl.unlink(missing_ok=True)

        job = client.batches.create(
            model=args.model,
            src=f"gs://{GCS_BUCKET}/{gcs_prefix}/input.jsonl",
            config=types.CreateBatchJobConfig(dest=f"gs://{GCS_BUCKET}/{gcs_prefix}/output/"),
        )
        print(f"  🚀 [Chunk {chunk_idx+1}/{len(chunks)}] Job {job.name.split('/')[-1]} submitted ({len(chunk)} images)", flush=True)
        return job

    in_flight: Dict[int, Any] = {}
    next_pending_pos = 0

    def top_up():
        nonlocal next_pending_pos
        while len(in_flight) < args.max_concurrent and next_pending_pos < len(pending_idxs):
            idx = pending_idxs[next_pending_pos]
            next_pending_pos += 1
            in_flight[idx] = submit_chunk(idx)

    path_lookup = {fn: (p, st) for fn, p, st in all_items}

    def ingest_results(results: List[Dict[str, Any]]):
        ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ANNOTATIONS_FILE, "a", encoding="utf-8") as annot_f:
            for item in results:
                fn = item["filename"]
                vlm = item.get("vlm_result")
                if not vlm or fn not in path_lookup:
                    continue

                curr_path, source_type = path_lookup[fn]
                is_valid = vlm.get("is_valid_human_look", False)
                score = vlm.get("overall_score")
                gender = vlm.get("gender")
                age = vlm.get("age_bracket")
                focus = vlm.get("focus_type")
                shot = vlm.get("shot_type")

                final_dest = curr_path

                if source_type == "raw_scrape":
                    if not is_valid:
                        dest_dir = REJECTED_DIR / "non_human_or_flatlay"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        final_dest = dest_dir / fn
                        if curr_path.exists():
                            shutil.move(str(curr_path), str(final_dest))
                    else:
                        demo = resolve_demographic(gender, age)
                        tier = resolve_tier(vlm.get("aesthetic_tier"), score)
                        stream = resolve_stream(focus, shot, str(curr_path))
                        if demo and tier:
                            dest_dir = TRAINING_DATA_DIR / stream / demo / tier
                            dest_dir.mkdir(parents=True, exist_ok=True)
                            final_dest = dest_dir / fn
                            if curr_path.exists():
                                shutil.move(str(curr_path), str(final_dest))
                elif source_type == "existing_sorted":
                    if not is_valid:
                        dest_dir = REJECTED_DIR / "quarantined_from_training"
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        final_dest = dest_dir / fn
                        if curr_path.exists():
                            shutil.move(str(curr_path), str(final_dest))

                record = {
                    "file": str(final_dest),
                    "filename": fn,
                    "source_category": curr_path.parent.name,
                    "heuristic_passed": True,
                    "vlm_result": vlm,
                    "destination": str(final_dest),
                    "status": "sorted" if is_valid else "rejected",
                    "timestamp": time.time(),
                }
                annot_f.write(json.dumps(record) + "\n")

    top_up()
    done_count = len(chunks) - len(pending_idxs)

    while in_flight:
        time.sleep(args.poll_interval)
        for idx in list(in_flight.keys()):
            try:
                job = client.batches.get(name=in_flight[idx].name)
            except Exception as e:
                print(f"    ⚠ Poll retry for chunk {idx+1}: {e}")
                continue

            state_str = str(job.state)
            if not state_str.endswith(("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED", "PARTIALLY_SUCCEEDED")):
                in_flight[idx] = job
                continue

            del in_flight[idx]
            if "SUCCEEDED" in state_str:
                chunk_results = collect_chunk_results(job, storage_client, f"{args.label}_c{idx:04d}")
                cache_path = CHUNKS_CACHE_DIR / f"{args.label}_chunk{idx:04d}.json"
                cache_path.write_text(json.dumps(chunk_results, indent=2))
                all_results.extend(chunk_results)
                ingest_results(chunk_results)
                done_count += 1
                print(f"  ✓ [Chunk {idx+1}/{len(chunks)}] Done ({len(chunk_results)} images). Total finished: {done_count}/{len(chunks)}", flush=True)
            else:
                print(f"  ❌ [Chunk {idx+1}/{len(chunks)}] Failed with state: {state_str}")

            top_up()

    print("\n" + "═" * 74)
    print("  Vertex AI Batch Processing Complete")
    print("═" * 74)
    print(f"  Total Processed Images: {len(all_results)}")
    print(f"  Updated Annotations   : {ANNOTATIONS_FILE}")
    print("═" * 74 + "\n")


if __name__ == "__main__":
    main()
