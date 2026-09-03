"""
extract_measured_labels.py -- Bucket C (see taxonomy.py's module docstring):
labels measured FROM PIXELS after generation, never put in the prompt.
Run this after full_run.py (or smoke_test.py --per-tier / merge_shards.py)
has produced images + a labels CSV.

What it adds per row, without deleting any row (a QA gate that silently
drops rows can skew the score distribution more than the bad images it
removes -- see the warning printed below):
  - measured_upper_color / measured_lower_color   (outfit only, snapped to
    taxonomy.COLOR_ANCHORS_RGB in CIELab, not RGB -- navy and black are
    close in RGB but obviously different to a human)
  - color_match_upper / color_match_lower          (1 if the measured
    colour equals the requested colour family, else 0 -- this is the
    "colour binding" signal validation_sweep.py --check-binding reports
    in aggregate before the full run)
  - color_harmony                                  ("harmonious",
    "neutral_pair", or "clashing" -- rule-based, no model)
  - qa_pass, qa_reasons                            (face detected, exactly
    one face found, blur (variance-of-Laplacian) >= MIN_BLUR_VARIANCE,
    brightness in range, and for Outfit rows a full-body/cropped-feet
    proxy check)

Usage:
    python3 extract_measured_labels.py <labels_csv> <images_dir> [--output out.csv]

Dependencies (already in ML/vision/requirements.txt): opencv-python-headless,
numpy, scikit-learn.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tx

# --------------------------------------------------------------------------
# QA gate thresholds
# --------------------------------------------------------------------------
MIN_BLUR_VARIANCE = 40.0          # variance-of-Laplacian; below this = too blurry
BRIGHTNESS_MIN = 35               # mean grayscale value; below = too dark to judge
BRIGHTNESS_MAX = 225              # above = blown out / overexposed
FEET_BAND_STD_MIN = 8.0           # bottom-band std-dev below this -> likely flat
                                   # floor/background, i.e. feet probably cropped
                                   # off (a crude proxy, not a real feet detector --
                                   # documented as such per the build spec)
QA_PASS_RATE_WARN_THRESHOLD = 0.85

# Crop bands (fraction of image width/height), per the build spec.
TORSO_BAND = (0.36, 0.26, 0.64, 0.45)   # x0, y0, x1, y1
LEGS_BAND = (0.40, 0.58, 0.60, 0.78)

NEUTRALS = {"black", "white", "grey", "beige"}
# Small, defensible clashing set -- classic complementary-primary clash.
# Deliberately kept small per the build spec ("flag a small clashing set")
# rather than trying to encode general colour theory here.
CLASHING_PAIRS = {frozenset({"red", "green"})}


def _lazy_imports():
    import cv2
    import numpy as np
    return cv2, np


# --------------------------------------------------------------------------
# Colour: crop -> skin-mask -> k-means -> CIELab snap to taxonomy palette
# --------------------------------------------------------------------------
def _rgb_to_lab(rgb, np):
    """Standard sRGB (0-255) -> CIELab (D65) conversion, no extra deps."""
    rgb_arr = np.array(rgb, dtype=np.float64) / 255.0
    rgb_arr = np.where(rgb_arr > 0.04045, ((rgb_arr + 0.055) / 1.055) ** 2.4, rgb_arr / 12.92)
    r, g, b = rgb_arr
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    x, y, z = x / 0.95047, y / 1.00000, z / 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), (7.787 * t) + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b_ = 200 * (fy - fz)
    return (L, a, b_)


def _palette_lab(np):
    return {name: _rgb_to_lab(rgb, np) for name, rgb in tx.COLOR_ANCHORS_RGB.items()}


def snap_to_palette(rgb, np, palette_lab=None):
    palette_lab = palette_lab or _palette_lab(np)
    lab = _rgb_to_lab(rgb, np)
    best_name, best_dist = None, float("inf")
    for name, plab in palette_lab.items():
        dist = sum((a - b) ** 2 for a, b in zip(lab, plab))
        if dist < best_dist:
            best_name, best_dist = name, dist
    return best_name


def _skin_mask_ycrcb(crop_bgr, cv2, np):
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    # Standard YCrCb skin-tone heuristic band.
    skin = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
    return skin


def _crop_fraction(img, band, cv2, np):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = band
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def measured_dominant_color(img_bgr, band, cv2, np):
    """Crops the given fractional band, masks out skin-toned pixels, k-means
    clusters the remainder in RGB, and returns the RGB centroid of the
    largest cluster. Falls back to the unmasked crop's mean colour if
    masking removes everything (e.g. a very skin-heavy crop)."""
    crop = _crop_fraction(img_bgr, band, cv2, np)
    if crop.size == 0:
        return None
    skin = _skin_mask_ycrcb(crop, cv2, np)
    non_skin = crop[~skin]
    pixels = non_skin if non_skin.shape[0] >= 50 else crop.reshape(-1, 3)
    if pixels.shape[0] == 0:
        return None

    # RGB (cv2 loads BGR)
    pixels_rgb = pixels[:, ::-1].astype(np.float64)
    try:
        from sklearn.cluster import KMeans
        k = min(3, max(1, pixels_rgb.shape[0] // 20))
        km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(pixels_rgb)
        counts = np.bincount(km.labels_)
        dominant = km.cluster_centers_[counts.argmax()]
    except ImportError:
        dominant = pixels_rgb.mean(axis=0)
    return tuple(float(v) for v in dominant)


def color_harmony(color_a, color_b):
    if color_a is None or color_b is None:
        return "unknown"
    if color_a == color_b:
        return "harmonious"
    if color_a in NEUTRALS or color_b in NEUTRALS:
        return "neutral_pair"
    if frozenset({color_a, color_b}) in CLASHING_PAIRS:
        return "clashing"
    return "harmonious"


# --------------------------------------------------------------------------
# QA gates
# --------------------------------------------------------------------------
def qa_checks(img_bgr, category, cv2, np, face_cascade):
    reasons = []
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur
    blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_var < MIN_BLUR_VARIANCE:
        reasons.append(f"blurry(var={blur_var:.1f})")

    # Brightness
    mean_brightness = float(gray.mean())
    if not (BRIGHTNESS_MIN <= mean_brightness <= BRIGHTNESS_MAX):
        reasons.append(f"brightness_out_of_range({mean_brightness:.0f})")

    # Face detection / person count proxy
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) == 0:
            reasons.append("no_face_detected")
        elif len(faces) > 1:
            reasons.append(f"multiple_faces({len(faces)})")

    # Full-body / cropped-feet proxy (outfit only)
    if tx.CATEGORY_KIND.get(category) == "outfit":
        h = img_bgr.shape[0]
        bottom_band = gray[int(h * 0.92):, :]
        if bottom_band.size and bottom_band.std() < FEET_BAND_STD_MIN:
            reasons.append(f"possible_cropped_feet(std={bottom_band.std():.1f})")

    return (len(reasons) == 0), reasons


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def _load_face_cascade(cv2):
    """Best-effort: some opencv builds/environments ship without the
    objdetect module or its bundled Haar cascade data wired up (seen on
    at least one dev machine here). Face-detected/person-count QA is then
    skipped rather than crashing the whole pass -- every other measured
    column (colour, blur, brightness, cropped-feet proxy) still runs."""
    try:
        cascade_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_file)
        if cascade.empty():
            raise RuntimeError("cascade loaded empty")
        return cascade
    except Exception as e:
        print(f"WARNING: could not load OpenCV's face cascade ({e}) -- face_detected/multiple_faces "
              f"QA checks will be skipped for this run. All other QA/colour checks still apply.")
        return None


def process(labels_csv, images_dir, output_csv):
    cv2, np = _lazy_imports()
    palette_lab = _palette_lab(np)
    face_cascade = _load_face_cascade(cv2)

    images_dir = Path(images_dir)
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    extra_fields = ["measured_upper_color", "measured_lower_color",
                     "color_match_upper", "color_match_lower",
                     "color_harmony", "qa_pass", "qa_reasons"]
    out_fieldnames = fieldnames + [f for f in extra_fields if f not in fieldnames]

    pass_count = 0
    missing = 0
    for row in rows:
        img_path = images_dir / row["filename"]
        if not img_path.exists():
            missing += 1
            for f in extra_fields:
                row.setdefault(f, "")
            row["qa_pass"] = "0"
            row["qa_reasons"] = "image_file_missing"
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            for f in extra_fields:
                row.setdefault(f, "")
            row["qa_pass"] = "0"
            row["qa_reasons"] = "unreadable_image"
            continue

        category = row.get("category", "")
        if tx.CATEGORY_KIND.get(category) == "outfit":
            upper_rgb = measured_dominant_color(img_bgr, TORSO_BAND, cv2, np)
            lower_rgb = measured_dominant_color(img_bgr, LEGS_BAND, cv2, np)
            measured_upper = snap_to_palette(upper_rgb, np, palette_lab) if upper_rgb else ""
            measured_lower = snap_to_palette(lower_rgb, np, palette_lab) if lower_rgb else ""
            requested_upper = row.get("requested_upper_color", "")
            requested_lower = row.get("requested_lower_color", "")
            row["measured_upper_color"] = measured_upper
            row["measured_lower_color"] = measured_lower
            row["color_match_upper"] = int(bool(measured_upper) and measured_upper == requested_upper)
            row["color_match_lower"] = int(
                bool(measured_lower) and requested_lower not in ("", "none") and measured_lower == requested_lower
            )
            row["color_harmony"] = color_harmony(measured_upper or None, measured_lower or None)
        else:
            row["measured_upper_color"] = ""
            row["measured_lower_color"] = ""
            row["color_match_upper"] = ""
            row["color_match_lower"] = ""
            row["color_harmony"] = ""

        qa_pass, reasons = qa_checks(img_bgr, category, cv2, np, face_cascade)
        row["qa_pass"] = int(qa_pass)
        row["qa_reasons"] = ";".join(reasons)
        if qa_pass:
            pass_count += 1

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    total = len(rows)
    scored = total - missing
    pass_rate = pass_count / scored if scored else 0.0
    print(f"Processed {total} rows ({missing} missing image files, skipped for QA scoring).")
    print(f"QA pass rate: {pass_count}/{scored} = {pass_rate:.1%}")
    if pass_rate < QA_PASS_RATE_WARN_THRESHOLD:
        print(f"WARNING: QA pass rate is below {QA_PASS_RATE_WARN_THRESHOLD:.0%}. Rows are NOT deleted -- "
              f"but an over-strict gate here could silently skew the trained score distribution more than "
              f"the images it's flagging would. Inspect qa_reasons before discarding anything downstream.")
    print(f"Wrote {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Bucket C: measure colour/QA labels from generated pixels.")
    parser.add_argument("labels_csv")
    parser.add_argument("images_dir")
    parser.add_argument("--output", default=None, help="Defaults to <labels_csv>_measured.csv")
    args = parser.parse_args()

    output_csv = args.output or str(Path(args.labels_csv).with_name(Path(args.labels_csv).stem + "_measured.csv"))
    process(args.labels_csv, args.images_dir, output_csv)


if __name__ == "__main__":
    main()
