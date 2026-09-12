#!/usr/bin/env python3
"""
LookMax Vision Model Predictor
==============================
Runs inference on an input image using the trained CoreML model (.mlpackage).
Outputs calibrated continuous score (1.0 - 10.0), aesthetic tier, and all decoded attribute heads.

Usage:
    python3 ML/vision/predict.py --image /path/to/photo.jpg --category Women_Outfit
    python3 ML/vision/predict.py --image /path/to/face.jpg --category Men_Grooming
"""

import argparse
import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Ensure root ML directory in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIES, get_image_size, MODELS_DIR

try:
    import coremltools as ct
    HAS_CT = True
except ImportError:
    HAS_CT = False


def predict_image(image_path: Path | str, category: str = "Women_Outfit") -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    mlpackage_path = MODELS_DIR / f"LookMax_{category}.mlpackage"
    if not mlpackage_path.exists():
        raise FileNotFoundError(f"CoreML model not found at: {mlpackage_path}")

    # Load CoreML Model
    model = ct.models.MLModel(str(mlpackage_path))

    # Read label schema from metadata
    meta = model.user_defined_metadata or {}
    schema_str = meta.get("label_schema", "[]")
    schema = json.loads(schema_str)

    # Resolution handling
    img_size = get_image_size(category)  # e.g. (512, 384) for Outfit, (384, 384) for Grooming
    if isinstance(img_size, tuple):
        h, w = img_size
    else:
        h, w = img_size, img_size

    # Non-destructive aspect preservation & resize
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)

    # Run CoreML Model Prediction
    preds = model.predict({"image": img})

    # Extract score
    score_raw = preds.get("score", 5.0)
    if hasattr(score_raw, "item"):
        score = float(score_raw.item())
    elif isinstance(score_raw, (np.ndarray, list)):
        score = float(score_raw[0])
    else:
        score = float(score_raw)

    score = round(max(1.0, min(10.0, score)), 2)

    # Decode attribute heads
    attributes = {}
    for f in schema:
        name = f["name"]
        if name == "score":
            continue

        raw_val = preds.get(name)
        if raw_val is None:
            continue

        probs = np.array(raw_val).squeeze()
        classes = f.get("classes") or []

        if len(probs.shape) == 0:
            attributes[name] = float(probs)
            continue

        top_idx = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        is_low_conf = confidence < 0.45

        runner_up = None
        if len(probs) > 1:
            sorted_indices = np.argsort(probs)[::-1]
            r_idx = int(sorted_indices[1])
            runner_up_class = classes[r_idx] if (classes and r_idx < len(classes)) else r_idx
            runner_up = {
                "prediction": runner_up_class,
                "confidence": round(float(probs[r_idx]), 3),
            }

        if classes and top_idx < len(classes):
            top_class = classes[top_idx]
            # Formatted key probabilities
            top_probs = {classes[i]: round(float(probs[i]), 3) for i in range(min(len(classes), len(probs)))}
            attributes[name] = {
                "prediction": top_class,
                "confidence": round(confidence, 3),
                "is_low_confidence": is_low_conf,
                "runner_up": runner_up,
                "distribution": top_probs,
            }
        else:
            attributes[name] = {
                "top_index": top_idx,
                "confidence": round(confidence, 3),
                "is_low_confidence": is_low_conf,
                "runner_up": runner_up,
            }

    # Flaw-Consistency Score Calibration:
    # An active flaw or un-groomed attribute prevents placing in the "Polished & Sharp" tier (>= 7.5).
    has_active_defect = False
    if "Grooming" in category:
        fh_style = attributes.get("facial_hair_style", {}).get("prediction", "clean_shaven")
        fh_groomed = attributes.get("facial_hair_groomed", {}).get("prediction")
        if fh_style not in ["clean_shaven", "none"] and fh_groomed == "0":
            has_active_defect = True
        if attributes.get("hair_styled", {}).get("prediction") == "0":
            has_active_defect = True
        if attributes.get("hair_untidy", {}).get("top_index", 0) > 0:
            has_active_defect = True
        if attributes.get("skin_neglected", {}).get("top_index", 0) > 0:
            has_active_defect = True
        if attributes.get("eyebrows_unkempt", {}).get("top_index", 0) > 0:
            has_active_defect = True
        if attributes.get("makeup_uneven", {}).get("top_index", 0) > 0:
            has_active_defect = True
    else:
        if attributes.get("fabric_wrinkled", {}).get("top_index", 0) > 0:
            has_active_defect = True
        if attributes.get("styling_sloppy", {}).get("top_index", 0) > 0:
            has_active_defect = True
        if attributes.get("footwear_worn", {}).get("top_index", 0) > 0:
            has_active_defect = True
        if attributes.get("fit_baggy", {}).get("top_index", 0) > 0:
            has_active_defect = True

    if has_active_defect and score >= 7.5:
        score = 7.2

    # Determine Tier
    if score >= 7.5:
        tier = "3_Polished"
        tier_label = "Polished & Sharp"
    elif score >= 5.0:
        tier = "2_Average"
        tier_label = "Average / Baseline Everyday"
    else:
        tier = "1_Needs_Improvement"
        tier_label = "Needs Improvement / Poor Execution"

    return {
        "file": str(image_path),
        "filename": image_path.name,
        "category": category,
        "score": score,
        "tier": tier,
        "tier_label": tier_label,
        "attributes": attributes,
    }


def format_card(result: dict) -> str:
    lines = []
    lines.append(f"══════════════════════════════════════════════════════════")
    lines.append(f"  LookMax Vision Analysis: {result['category']}")
    lines.append(f"══════════════════════════════════════════════════════════")
    lines.append(f"  📷 Image      : {result['filename']}")
    lines.append(f"  ⭐ Score      : {result['score']} / 10.0")
    lines.append(f"  🏷  Tier       : {result['tier_label']} ({result['tier']})")
    lines.append(f"──────────────────────────────────────────────────────────")
    lines.append(f"  Detected Wardrobe & Styling Attributes:")

    attrs = result.get("attributes", {})
    # Core garments
    garments = ["upper_type", "mid_type", "lower_type", "footwear_type"]
    for g in garments:
        if g in attrs:
            val = attrs[g].get("prediction")
            conf = attrs[g].get("confidence", 0) * 100
            lines.append(f"    • {g:<15}: {val:<20} ({conf:.1f}% confidence)")

    # Formality & Fit
    lines.append(f"\n  Fit & Execution Signals:")
    signals = [
        ("formality", "Formality"),
        ("fit_tailored", "Tailored Fit"),
        ("fit_baggy", "Baggy Silhouette"),
        ("fit_tight", "Tight Fit"),
        ("fabric_crisp", "Fabric Crispness"),
        ("fabric_wrinkled", "Fabric Wrinkles"),
        ("styling_sharp", "Styling Sharpness"),
        ("styling_sloppy", "Sloppy Execution"),
        ("footwear_polished", "Footwear Polished"),
        ("footwear_worn", "Footwear Scuffed/Worn"),
    ]
    for key, label in signals:
        if key in attrs:
            val_data = attrs[key]
            if "prediction" in val_data:
                pred = val_data["prediction"]
                if str(pred) in ["0", "1"]:
                    pred = "Yes / Sharp" if str(pred) == "1" else "No"
            else:
                idx = val_data.get("top_index", 0)
                if key == "fit_tailored" or key.endswith("_sharp") or key.endswith("_crisp") or key.endswith("_polished"):
                    pred = "Yes / Sharp" if idx == 1 else "No"
                else:
                    pred = ["Absent", "Mild / Present", "Severe"][min(idx, 2)]

            conf = val_data.get("confidence", 0) * 100
            lines.append(f"    • {label:<22}: {pred:<16} ({conf:.1f}% confidence)")

    lines.append(f"══════════════════════════════════════════════════════════\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Test LookMax CoreML Model on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to test image file")
    parser.add_argument("--category", type=str, default="Women_Outfit", choices=CATEGORIES)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    try:
        res = predict_image(args.image, category=args.category)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print(format_card(res))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
