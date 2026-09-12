import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
from ML.vision.predict import predict_image

img_dir = Path("/Users/sayyed/Downloads/Test Images")

routing = [
    ("Asian man polo jeans.jpeg", "Men_Outfit"),
    ("Man Portrait sharp.webp", "Men_Grooming"),
    ("Man needs improvement.avif", "Men_Grooming"),
    ("Man portrain needs improvement.avif", "Men_Grooming"),
    ("Woman average.webp", "Women_Grooming"),
    ("Woman face sharp.jpg", "Women_Grooming"),
    ("Woman full body sharp.jpg", "Women_Outfit"),
    ("bald man good.webp", "Men_Outfit"),
    ("needs improvement woman.jpeg", "Women_Grooming"),
    ("woman bad hair.webp", "Women_Grooming"),
]

print("\n" + "=" * 80)
print("  LOOKMAX VISION COREML MODEL PREDICTIONS ON DOWNLOADS FOLDER")
print("=" * 80)

for filename, cat in routing:
    p = img_dir / filename
    res = predict_image(p, category=cat)
    score = res["score"]
    tier = res["tier"]
    tier_label = res["tier_label"]
    attrs = res.get("attributes", {})

    print(f"\n📷 Image   : {filename}")
    print(f"🎯 Category: {cat}")
    print(f"⭐ Score   : {score:.2f} / 10.0  [{tier_label}]")
    print("   Detected Features:")

    if "Outfit" in cat:
        for k in ["upper_type", "lower_type", "footwear_type", "formality", "fit_tailored", "fit_baggy", "styling_sharp"]:
            if k in attrs:
                pred = attrs[k].get("prediction", attrs[k].get("top_index"))
                conf = attrs[k].get("confidence", 0) * 100
                print(f"     • {k:<16}: {str(pred):<18} ({conf:.1f}% conf)")
    else:
        for k in ["hair_styled", "hair_length", "beard_styled", "skin_healthy", "makeup_style"]:
            if k in attrs:
                pred = attrs[k].get("prediction", attrs[k].get("top_index"))
                conf = attrs[k].get("confidence", 0) * 100
                print(f"     • {k:<16}: {str(pred):<18} ({conf:.1f}% conf)")

print("\n" + "=" * 80 + "\n")
