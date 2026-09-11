# LookMax ML Pipeline

Machine learning pipeline for LookMax — an iOS app that rates a user's styling and grooming **effort** (not genetics) on a 1–10 scale and returns an actionable improvement checklist.

Two separate on-device models, two separate training pipelines:
1. **Vision Models** (scores the photo, outputs attribute tags): `ML/vision/`
2. **On-Device Stylist LLM** (turns those tags into one specific, actionable fix): `ML/stylist_llm/`

They are fully isolated in code, data, and checkpoints, connected only by one deliberate, one-directional interface contract (`tag_vocabulary.py` reading `taxonomy.py`).

---

## Architecture Philosophy

> **We rate effort and execution, not beauty.**

A plus-size user with an asymmetrical face who has perfectly styled hair, crisp fitting clothes, and polished grooming should score **10/10**. A conventionally attractive person in a wrinkled, stained baggy t-shirt scores **2/10**.

**Concretely, the score and checklist may only reflect things a user can change within minutes to hours** — hairstyle, product use, outfit choice, fit, color coordination, makeup. They must never reflect things a user cannot change on that timescale: body weight/size, face shape, skin conditions like acne, age. If a synthetic training image or a generated checklist item implies "lose weight" or "clear up your skin" as the path to a higher score, that is a bug in the dataset or the prompt, not a valid signal.

---

## Directory Structure

```
ML/
├── vision/                           ← Vision Model Pipeline
│   ├── dataset_synthetic/            ← Qwen-Image-2512 synthetic image generation
│   │   ├── taxonomy.py               ← Single source of truth for variation matrix & label schema
│   │   ├── prompt_builder.py         ← Composes prompt + ground truth labels
│   │   ├── qwen_pipeline.py          ← GPU model loading & inference
│   │   ├── smoke_test.py             ← Quick verification (--dry-run / --per-tier)
│   │   ├── quick_prompt_test.py      ← First GPU confirmation test (6 hardcoded prompts)
│   │   ├── variation_test.py         ← Diversity & quality test within tiers
│   │   ├── validation_sweep.py       ← Coverage simulation & binding gate
│   │   ├── full_run.py               ← 28,000-image production generator
│   │   ├── merge_shards.py           ← Merges shard CSVs
│   │   ├── extract_measured_labels.py← Pixel-measured color & QA filters — final qa_pass gate
│   │   ├── rebuild_label_csvs.py     ← Deterministic label-CSV reconstruction (recovers lost/missing rows)
│   │   ├── verify_dataset_vertex.py  ← Synchronous VLM semantic audit (Gemini via Vertex)
│   │   ├── verify_batch.py           ← Same audit, Vertex Batch Prediction (hundreds+ images)
│   │   ├── prompt_experiment.py      ← A/B prompt-wording test harness (isolated GPU box)
│   │   ├── relabel_batch.py          ← Classify-then-correct labels in place, or salvage+copy before regen
│   │   ├── regenerate_targeted.py    ← Regenerates an explicit filename whitelist only
│   │   ├── pull_regenerated.py       ← Downloads regenerated images back, checksum-verified
│   │   ├── fleet_monitor.py          ← Multi-machine sync/health monitoring during generation
│   │   ├── install.sh                ← Remote GPU setup script
│   │   ├── PLAN.md                   ← Execution guide for image generation
│   │   └── QA_AND_RELABELING.md      ← Post-generation audit/fix pipeline — read before touching label CSVs
│   ├── dataset_real/                 ← Real photos: scraping & VLM classification
│   │   ├── 01_setup_environment.py   ← Verifies environment & dependencies
│   │   ├── 02_scrape_images.py       ← Scrapes Unsplash, Pexels, Pixabay
│   │   ├── 03_classify_and_sort.py   ← VLM classification into aesthetic tiers (14,079 images)
│   │   ├── reddit_scraper.py         ← Reddit Playwright scraper (rate-limited, human-like delays)
│   │   ├── generate_face_queries.py  ← Search-query generation (good-face buckets)
│   │   ├── generate_bad_face_queries.py ← Search-query generation ("needs improvement" buckets)
│   │   ├── load_celeba_dataset.py    ← CelebA-HQ ingestion
│   │   ├── load_fairface_dataset.py  ← FairFace demographic ingestion
│   │   ├── load_unsplash_dataset.py  ← Unsplash research dataset ingestion
│   │   └── reddit_*.json             ← Search query catalogs (face/outfit/bad-face/general)
│   ├── training/                     ← Multi-Head PyTorch Training & CoreML Export
│   │   ├── pretrain_synthetic.py     ← Phase A: Multi-head synthetic pretraining
│   │   ├── finetune_real_world.py    ← Phase B: Real-world fine-tuning + synthetic replay + CoreML export
│   │   ├── multihead_common.py       ← Shared MultiHeadModel architecture, dataset, & CoreML export
│   │   └── inspect_phaseA.py         ← Post-training diagnostics: confusion patterns, score-tier calibration
│   ├── config.py                     ← Centralized vision settings & paths
│   └── requirements.txt              ← Vision pipeline dependencies
├── stylist_llm/                      ← On-Device Stylist LLM (SmolLM2-135M)
│   ├── config.py                     ← Self-contained LLM hyperparameters & paths
│   ├── tag_vocabulary.py             ← Interface contract: translates vision taxonomy to tag prompt
│   ├── generate_synthetic_dataset.py ← Generates advice pairs via Gemini
│   ├── qa_review.py                  ← Quality gate (word count, meta-chatter, effort-vs-genetics)
│   ├── prune_vocabulary.py           ← Prunes vocabulary to domain terms (~106M params)
│   ├── remap_tokenizer.py            ← Tokenizer wrapper with pruned token IDs
│   ├── finetune.py                   ← Full SFT fine-tuning with prompt loss masking
│   ├── export_coreml.py              ← Stateless (no KV-cache) FP16 CoreML export for iOS 18+ ANE — see PLAN.md "Current Status"
│   ├── smoke_test.py                 ← Checkpoint & .mlpackage verification
│   ├── install.sh                    ← Setup script
│   ├── requirements.txt              ← LLM dependencies
│   └── PLAN.md                       ← Detailed setup & execution guide
├── data/                             ← Training & Evaluation Datasets
│   ├── vision_real/                  ← Real photo pipeline data
│   │   ├── 1_Raw_Scrapes/            ← Raw unclassified scraped images
│   │   ├── 2_VLM_Processing/         ← VLM metadata logs & rejected images
│   │   └── 3_CoreML_Training_Data/   ← 14,079 curated images sorted by Stream/Demographic/Tier
│   ├── vision_synthetic/             ← Synthetic Qwen pipeline data
│   │   ├── raw_generated/            ← Generated image shards & raw CSVs
│   │   └── qa_processed/             ← Post-QA images with measured labels
│   └── stylist_llm/                  ← Stylist LLM training data
│       ├── raw_generated/            ← Raw instruction pairs (local Ollama by default, or Gemini)
│       ├── qa_reviewed/              ← Filtered dataset passing all QA gates
│       └── pruned_vocab/             ← Tokenizer mapping artifacts
├── models/                           ← Exported CoreML .mlpackage artifacts for Xcode
│                                        (all 5 populated as of 2026-09-11: 4x LookMax_<Category>.mlpackage
│                                        + StylistEngine.mlpackage/StylistEngine_INT8.mlpackage, plus
│                                        the .pt checkpoints and *_metrics.json each phase produced)
└── archive/                          ← Deprecated iterations (FLUX v7, etc.)
```

---

## The Two Training Pipelines

### 1. Vision Model Pipeline (`ML/vision/`)

The vision pipeline trains **4 models** (consolidated across age brackets to maximize sample depth):
* `LookMax_Men_Grooming.mlpackage`
* `LookMax_Women_Grooming.mlpackage`
* `LookMax_Men_Outfit.mlpackage`
* `LookMax_Women_Outfit.mlpackage`

**Status (2026-09-11): both phases complete, all 4 `.mlpackage` files exported to `ML/models/`.**
28,000 synthetic images generated, audited, corrected, and grown to 29,436
training-ready images post-QA (see `ML/vision/dataset_synthetic/QA_AND_RELABELING.md`
for the full audit/relabel/regeneration story — this is not a small footnote,
it's most of what made the dataset actually trainable). Phase A ran
12-19 epochs per category (early-stopped, 25 max) with 90.6-97.4% average
per-head validation accuracy. Phase B ran the full 10 epochs per category
with no forgetting of Phase A's attribute heads (96-98.2% retained). One
known, measured issue: `Women_Outfit`'s real "needs improvement" bucket is
thin (137 of 14,079 real photos) and the fine-tuned model is measurably
too generous on real bad outfits in that category (predicted mean 6.34 vs.
true annotated mean 4.09 on that bucket) — flagged as a real-data scraping
gap, not a bug, not yet fixed. See `ML/models/LookMax_<Category>_metrics.json`
for full per-category numbers and `inspect_phaseA.py` for the diagnostic
methodology (score-tier calibration, per-class confusion — not just
aggregate accuracy) used to find it.

#### Model Architecture
Backbone: `mobilenet_v3_large` (or `fastvit_t8`).
Multi-head architecture dynamically constructed from `label_schema_<Category>.json`:
* **Primary Head**: Continuous `score` regression (1.0 to 10.0 scale) trained with `SmoothL1Loss` (weight 1.0).
* **Auxiliary Heads**: Attribute classification heads (`upper_type`, `formality`, `fabric_wrinkled`, `fit_baggy`, `hair_styled`, etc.) trained with cross-entropy (weight 0.3, or 0.5 for formality).
* **Category-Specific Input Resolutions & Transforms**:
  - **Grooming (`LookMax_Men/Women_Grooming`)**: $384 \times 384$ Square. iOS uses Apple Vision `face.boundingBox` expanded with a ~35% margin to crop the head, hair, and neckline tightly into a square. This ensures consistent scale and maximizes facial feature density regardless of distance.
  - **Outfit (`LookMax_Men/Women_Outfit`)**: $384 \times 512$ Portrait (3:4 aspect ratio). Directly resized from the camera's 3:4 portrait view without destructive center-cropping, preserving the entire body from head/collar down to hemline/shoes.
  - *Legacy $224 \times 224$ and `CenterCrop` were removed to prevent amputating head/shoes and washing out fine fabric micro-wrinkles.*

#### Two-Phase Training Strategy
* **Phase A: Synthetic Pretraining (`pretrain_synthetic.py`)**:
  Trains all heads with full supervision on the synthetic Qwen-Image-2512 dataset (`data/vision_synthetic/qa_processed/`). Saves PyTorch checkpoints (`LookMax_<Category>_phaseA.pt`).
* **Phase B: Real-World Fine-Tuning (`finetune_real_world.py`)**:
  Loads Phase A checkpoint. Fine-tunes on real phone/candid photos (`data/vision_real/3_CoreML_Training_Data/`) to adapt to real camera optics and lighting. Because real photos only have tier labels, attribute targets are masked out during loss calculation. Each batch mixes a 30% synthetic replay fraction (`--replay-ratio 0.3`) to prevent catastrophic forgetting of attribute heads. Exports final `.mlpackage` artifacts to `ML/models/`.

---

### 2. Stylist LLM Pipeline (`ML/stylist_llm/`)

Produces `StylistEngine.mlpackage` — a compact, fast language model that takes detected vision tags and occasion, outputting a concise (<50 words), actionable "5-minute fix". **Status: dataset generated, model fine-tuned, exported, and quality-verified once (see `ML/stylist_llm/PLAN.md`'s "Current Status" section for the full account, real numbers, and open issues — this summary is necessarily abbreviated).**

* **Base Model**: `HuggingFaceTB/SmolLM2-135M-Instruct`.
* **Training data**: generated locally via Ollama (`qwen2.5:14b-instruct`) by default — no cloud API needed; a Gemini backend is also available. 4,994 examples generated, 4,985 passed QA.
* **Vocabulary Pruning (`prune_vocabulary.py`)**: Embedding and LM head are pruned to retain only domain words (garments, fit descriptors, occasions) plus ChatML special tokens. Measured: 134.5M → 107.83M parameters.
* **Fine-Tuning (`finetune.py`)**: SFT training with supervised tokens masked so only the assistant advice is trained. Measured: 18m22s on Apple Silicon (MPS).
* **CoreML Export (`export_coreml.py`)**: **Stateless** (no KV-cache — a stateful `StaticCache`/`ct.StateType` design was attempted and abandoned after hitting confirmed upstream PyTorch/coremltools bugs) conversion targeting **iOS 18+** Apple Neural Engine (ANE). Currently shipped at **FP16 (207MB)**, not INT4 — a real side-by-side quality test found INT4 quantization caused response truncation and repetition-loop failures that FP16 doesn't have. INT8 is a smaller, not-yet-quality-tested middle ground.
* **Interface Contract (`tag_vocabulary.py`)**: Reads the vision model's real `taxonomy.py` schema to format tags identically between pipelines. Known gap: its outfit branch compresses multiple simultaneous defects into a single `priority_defect` field, losing real information that its grooming branch doesn't lose (see PLAN.md).

---

## iOS Integration

1. **Camera Capture**: `AVFoundation` with real-time biometric tracking (neck-to-root spine alignment axis, face contour landmarks) and lighting check.
2. **Vision Inference**: Runs `LookMax_<Category>.mlpackage` on-device to produce effort score (1–10) and attribute tags.
3. **Stylist Advice**: Passes detected tags + user-selected occasion to `StylistEngine.mlpackage` on-device for instant, offline 5-minute fix advice.
4. **Privacy & Offline**: Zero network dependency for core styling evaluations; no cloud costs. Cloud Gemini VLM (`GeminiVisionService.swift`) remains available as a high-fidelity reference and development baseline.
