# LookMax ML Pipeline

Machine learning pipeline for LookMax — an iOS app that rates a user's styling and grooming **effort** (not genetics) on a 1–10 scale and returns an actionable improvement checklist.

---

## Architecture Philosophy

> **We rate effort and execution, not beauty.**

A plus-size user with an asymmetrical face who has perfectly styled hair, crisp fitting clothes, and polished grooming should score **10/10**. A conventionally attractive person in a wrinkled, stained baggy t-shirt scores **2/10**. The models are trained on purely effort-based signals.

**Concretely, the score and checklist may only reflect things a user can change within minutes to hours** — hairstyle, product use, outfit choice, fit, color coordination, makeup. They must never reflect things a user cannot change on that timescale: body weight/size, face shape, skin conditions like acne, age. If a synthetic training image or a generated checklist item implies "lose weight" or "clear up your skin" as the path to a higher score, that's a bug in the dataset or the prompt, not a valid signal — see the taxonomy v4 note below for a concrete instance of this (body-shape drift) that was caught and fixed in the synthetic data pipeline.

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

We tested both. **FLUX.1 [schnell]** (4-step distilled) runs at ~2 seconds/image but suffers from "prompt collapse" — it ignores complex negative flaw descriptors and gravitates toward making everyone look attractive regardless of the prompt. **FLUX.1 [dev]** (28 steps, 12B parameters) is significantly better at following flaw descriptors than [schnell], but is not immune to the same collapse — see the note below.

> **Update (taxonomy v3 → v4):** Visual review of `test_variations_comprehensive/` (taxonomy v2, `guidance_scale=3.5`) showed [dev] still softens abstract flaw adjectives ("severely overgrown greasy unwashed hair," "heavily stained baggy t-shirt") into merely-slightly-messy versions of an otherwise conventionally attractive render — Flaw-tier images were often hard to distinguish from Average or even Polished. Taxonomy v3 rewrote flaw descriptions with concrete sensory detail (grease sheen, flaking, stains, fold lines) instead of intensity adjectives, raised `guidance_scale` to 5.0, and fixed one random seed per identity across its Flaw/Average/Polished triplet.
>
> Reviewing the resulting v3 batch (`test_variations_comprehensive_v3/`) surfaced two further problems that go directly against the "effort, not genetics" philosophy above:
> 1. **Body-shape drift:** "Polished" outfit language ("flawlessly tailored proportions that flatter the body") was visibly slimming heavy-set/plus-size/curvy identities relative to their own Flaw-tier render — the fixed seed pins pose/composition but not body proportions, which are still reshaped through cross-attention on the styling text. This would have silently taught the model that losing weight is part of "improvement," which is exactly what the app must never do.
> 2. **Focus drift:** "Average"-tier images were visibly blurrier than Flaw/Polished for the same identity — traced to self-referential closers like "an unremarkable average appearance" in the v3 Average descriptions, which appears to read as a cue for soft/amateur-snapshot rendering rather than just low grooming/styling effort.
>
> Taxonomy v4 fixed both broadly (confirmed across a full 48-image batch), but two things didn't fully hold: grooming flaw visibility was identity-dependent — clear for Caucasian/East Asian test identities, nearly invisible for Black, South Asian, and Hispanic ones — and one outfit triplet with a moderate ("athletic") build still drifted heavier in Flaw than Polished despite the body clause. Taxonomy v5 reinforced every Flaw description with a second restatement of its defect plus a blunt evaluative closer, and moved the body-preservation clause to the front of the prompt.
>
> **While v5 was generating on the remote server, its own console output revealed the actual root cause**: `"CLIP can only handle sequences up to 77 tokens"`, truncating on every image. FLUX.1 uses two text encoders — CLIP (hard 77-token cap, contributes a global conditioning vector) and T5-XXL (`max_sequence_length=512`, drives most fine-grained detail). Every taxonomy round since v3 kept adding reinforcement text, pushing prompts to 180-220+ tokens; T5 saw all of it, but CLIP was silently losing most or all of the flaw description. Verified with the real tokenizer: a v5 Outfit prompt's CLIP view cut off *before the outfit description even started*. This is a better explanation for the identity-dependent weakness than an aesthetic-prior theory — each round of "make the flaw language stronger" was fighting a token-budget problem underneath it, and made that problem worse, not better.
>
> Taxonomy v6 fixes this directly: every prompt now leads with a short opener (core flaw/effort keywords, plus body preservation for outfits) verified with the real CLIP tokenizer to fit completely inside the 77-token window, even for the longest identity combinations. **Validate v6 on the remote server before running `generate_flux_dataset.py`** — the master script has already been updated to match, but its `guidance_scale` is intentionally left at 3.5 pending visual confirmation. If grooming flaw visibility is still weak after v6 — with the CLIP truncation actually fixed — that's real evidence to stop iterating on wording and add a post-generation VLM QA pass instead (score each image against its intended label, discard/flag mismatches) — see the recommendation further down.

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

| Level | Score | Men Grooming Example (v3) |
|---|---|---|
| Flaw | 2–3 | `"visibly oily greasy hair matted flat with individual strands clumped together and a visible grease sheen catching the light, unwashed for over a week, thin patchy neckbeard growing in uneven blotchy patches with bare skin gaps, dry flaking dead skin visibly peeling on the forehead"` |
| Average | 4–6 | `"plain unstyled short hair with no product lying flat with no defined shape, ordinary trimmed facial hair with no sharp lineup, plain skin with no visible skincare routine"` |
| Polished | 9–10 | `"meticulously styled textured voluminous hair with a visible pomade sheen and sharp defined part, razor-sharp lined-up short beard with crisp clean edges, flawless clear glowing hydrated skin"` |

v3's flaw descriptions favor concrete, sensory detail (grease sheen, flaking, blotchy patches) over abstract intensity words ("severely," "unkempt") — the abstract phrasing in v2 was being softened by FLUX's aesthetic prior. See the taxonomy note above.

> **Note on Polished Outfits:** "Polished" does NOT mean suits or formal wear. A perfectly fitted streetwear outfit (crisp t-shirt + sleek jacket) scores 10/10. The model rates **fit, fabric crispness, and color harmony**, not formality level.

### Dataset Scale
- **Total images:** 24,000
- **Split:** 6,000 per model (Men_Grooming, Women_Grooming, Men_Outfit, Women_Outfit)
- **Image size:** 1024×1024 PNG
- **Estimated disk space:** ~65 GB
- **Output:** `dataset_output/images/` + `dataset_output/labels.csv`

> **Recommended next step — automated label-adherence QA:** Because prompt collapse can silently mislabel images (a Flaw-tier prompt rendering as a Polished-looking photo), manually spot-checking 24,000 images isn't feasible. Before or shortly after the full run, consider reusing the VLM filtering pattern already built for `2_VLM_Processing/` (see `filtered_rejected/`) to score each generated image against its intended level/attributes and flag or regenerate mismatches, rather than trusting the prompt's label unconditionally.

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
