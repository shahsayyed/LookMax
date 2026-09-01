# LookMax ML Pipeline

Machine learning pipeline for LookMax — an iOS app that rates a user's styling and grooming **effort** (not genetics) on a 1–10 scale and returns an actionable improvement checklist.

---

## Architecture Philosophy

> **We rate effort and execution, not beauty.**

A plus-size user with an asymmetrical face who has perfectly styled hair, crisp fitting clothes, and polished grooming should score **10/10**. A conventionally attractive person in a wrinkled, stained baggy t-shirt scores **2/10**. The models are trained on purely effort-based signals.

---

## Model Architecture

We train **4 separate models** (split by gender and stream) to keep each model narrowly focused:

| Model Name | Input | Outputs |
|---|---|---|
| `Men_Grooming` | Head-and-shoulders portrait | Score (1–10) + attribute checklist |
| `Women_Grooming` | Head-and-shoulders portrait | Score (1–10) + attribute checklist |
| `Men_Outfit` | Full-body portrait | Score (1–10) + attribute checklist |
| `Women_Outfit` | Full-body portrait | Score (1–10) + attribute checklist |

Each model uses a **Two-Headed multi-task architecture**:
- **Head 1:** Regression output → single float `score` (1.0–10.0)
- **Head 2:** Multi-label binary classification → binary attribute checklist (e.g., `hair_messy=1`, `clothes_wrinkled=0`)

Models are trained in PyTorch and exported to both:
- `.mlpackage` (CoreML) for iPhone inference
- `.tflite` for future Android support

---

## Pipeline Phases

```
ML/
├── pipeline/
│   ├── dataset_generator/           ← Phase 1: Synthetic data via FLUX.1 [dev]
│   │   ├── generate_flux_dataset.py ← Master 24,000-image generation script
│   │   ├── test_flux_prompts.py     ← Quick 4-image sanity check
│   │   ├── test_flux_variations.py  ← Full 48-image demographic/level validation
│   │   ├── requirements.txt         ← Server dependencies (no torch pinning)
│   │   └── PLAN.md                  ← Full server setup & execution guide
│   ├── 04_train_coreml_models.py    ← Phase 2: PyTorch training + CoreML export
│   └── 05_finetune_real_world.py    ← Phase 3 (TODO): Fine-tune on real-world images
├── models/                          ← Exported .mlpackage artifacts for Xcode
└── data/
    ├── Remote AI Image Generations/ ← Test outputs from FLUX.1 [dev]
    ├── Remote AI Image Generations New/ ← Test outputs using taxonomy v2
    └── 3_CoreML_Training_Data/      ← Final curated training data
```

---

## Phase 1: Synthetic Dataset Generation (FLUX.1 [dev])

Because labeling 24,000 real-world photos by hand is impractical, we use **FLUX.1 [dev]** — a state-of-the-art open-source text-to-image model by Black Forest Labs — to procedurally generate a precisely labeled synthetic dataset.

### Why FLUX.1 [dev] and not [schnell]?

We tested both. **FLUX.1 [schnell]** (4-step distilled) runs at ~2 seconds/image but suffers from "prompt collapse" — it ignores complex negative flaw descriptors and gravitates toward making everyone look attractive regardless of the prompt. **FLUX.1 [dev]** (28 steps, 12B parameters) is the only model that reliably generates obvious grooming flaws (greasy hair, smeared makeup, sloppy clothing).

| Model | Steps | Speed on RTX 5090 | Flaw Adherence |
|---|---|---|---|
| FLUX.1 [schnell] | 4 | ~2 sec/image | ❌ Fails on complex flaws |
| FLUX.1 [dev] | 28 | ~19 sec/image | ✅ Accurate |

### Prompt Taxonomy (v2)

All prompts follow this structure:
```
A {alignment_guide} of a {identity}, {variation_description}. Photorealistic, ultra detailed, 85mm lens.
```

**Alignment guides** lock the camera framing per model type:
- **Grooming:** `"front-facing head-and-shoulders portrait, perfectly centered face, straight-on eye-level camera angle, looking directly at the camera, bright even studio lighting"`
- **Outfit:** `"front-facing full-body portrait, perfectly centered, standing straight, straight-on eye-level camera angle, looking directly at the camera, head to toe visible, bright even studio lighting"`

**Identities** combine age + ethnicity + body type + face shape to ensure diversity and prevent genetic bias. Example:
- `"28-year-old Caucasian man, heavy set, round face, double chin"`
- `"26-year-old East Asian woman, athletic, heart-shaped face"`

**Variation Descriptions (Taxonomy v2 — Effort-Based Only):**

> **Critical design decision:** Flaws are **purely effort-based**. We never penalise biological traits. Acne, face shape, and body type do NOT appear in flaw descriptions. If a prompt contains "acne" it will teach the model to penalise users who cannot control their skin condition.

| Level | Score | Men Grooming Example |
|---|---|---|
| Flaw | 2–3 | `"severely overgrown greasy unwashed hair, patchy unmaintained neckbeard stubble, dry flaky skin, unkempt bushy unibrow"` |
| Average | 4–6 | `"basic short haircut with no styling product, natural flat unstyled hair, standard trimmed facial hair, bare natural skin"` |
| Polished | 9–10 | `"meticulously styled hair with high volume and visible styling pomade, extremely sharp razor-edge beard lineup, deeply hydrated glowing skin"` |

> **Note on Polished Outfits:** "Polished" does NOT mean suits or formal wear. A perfectly fitted streetwear outfit (crisp t-shirt + sleek jacket) scores 10/10. The model rates **fit, fabric crispness, and color harmony**, not formality level.

### Dataset Scale
- **Total images:** 24,000
- **Split:** 6,000 per model (Men_Grooming, Women_Grooming, Men_Outfit, Women_Outfit)
- **Image size:** 1024×1024 PNG
- **Estimated disk space:** ~65 GB
- **Output:** `dataset_output/images/` + `dataset_output/labels.csv`

---

## Phase 2: Training (PyTorch → CoreML)

Script: `pipeline/04_train_coreml_models.py`

- Base model: EfficientNet-B3 (pretrained ImageNet)
- Fine-tuned with two output heads (score regression + attribute multi-label)
- Exported via `coremltools` to `.mlpackage`
- Compatible with iOS 17+ CoreML runtime

---

## Phase 3: Real-World Fine-Tuning (TODO)

Script: `pipeline/05_finetune_real_world.py` (to be created)

After Phase 1 pre-training, we will fine-tune on the existing classified real-world image datasets to adapt the model from synthetic to real photography conditions.

---

## iOS Integration

- Models run fully **on-device** (no internet dependency)
- Camera framing enforced by AVFoundation overlay (oval for grooming, rectangle for outfit)
- Lighting check via `CMSampleBuffer` EXIF `brightnessValue` before inference
- Contextual improvement advice is generated by a **native Swift rules engine** (not an LLM), mapping binary attribute outputs like `hair_messy=1` to human-readable suggestions
