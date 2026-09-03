# LookMax ML Pipeline

Machine learning pipeline for LookMax — an iOS app that rates a user's styling and grooming **effort** (not genetics) on a 1–10 scale and returns an actionable improvement checklist.

Two separate on-device models, two separate training pipelines: a **vision model** (scores the photo, outputs tags — `ML/pipeline/dataset_generator/` + `ML/pipeline/real_data_pipeline/`) and an **on-device stylist LLM** (turns those tags into one specific, actionable fix — `ML/pipeline/stylist_llm/`). They are fully isolated in code/data/checkpoints, connected only by one deliberate, one-directional interface contract — see "On-device stylist LLM" below.

---

## Architecture Philosophy

> **We rate effort and execution, not beauty.**

A plus-size user with an asymmetrical face who has perfectly styled hair, crisp fitting clothes, and polished grooming should score **10/10**. A conventionally attractive person in a wrinkled, stained baggy t-shirt scores **2/10**.

**Concretely, the score and checklist may only reflect things a user can change within minutes to hours** — hairstyle, product use, outfit choice, fit, color coordination, makeup. They must never reflect things a user cannot change on that timescale: body weight/size, face shape, skin conditions like acne, age. If a synthetic training image or a generated checklist item implies "lose weight" or "clear up your skin" as the path to a higher score, that's a bug in the dataset or the prompt, not a valid signal. This has been a real, caught-and-fixed bug more than once — see `ML/pipeline/dataset_generator/taxonomy.py`'s module docstring and `PLAN.md` for the specific instances (body-shape drift under "polished" styling language; a `skin_acne` label that was hardcoded to always-zero in an earlier taxonomy, i.e. present as a column but never actually meaningful — dropped entirely in the current one rather than carried forward as dead weight).

---

## Two data sources, kept deliberately separate

This pipeline gets labeled training images two independent ways. They live in separate code folders and separate data folders so they can never be silently mixed or mistaken for each other:

| | Real photos | Synthetic (AI-generated) photos |
|---|---|---|
| Code | `ML/pipeline/real_data_pipeline/` | `ML/pipeline/dataset_generator/` |
| Data | `ML/data/1_Raw_Scrapes/` → `2_VLM_Processing/` → `3_CoreML_Training_Data/` | `ML/data/4_Synthetic_Qwen/raw_generated/` → `qa_processed/` |
| Source | Reddit scraping + VLM classification | Qwen-Image-2512, procedurally prompted |
| Status | Populated — 14,079 classified images | Pipeline built; dataset generation not yet run |

## Pipeline layout

```
ML/
├── pipeline/
│   ├── real_data_pipeline/           ← real photos: scrape + classify
│   │   ├── 01_setup_environment.py
│   │   ├── 02_scrape_images.py       ← Reddit scraper (Playwright)
│   │   ├── 03_classify_and_sort.py   ← VLM classification into aesthetic tiers
│   │   ├── reddit_scraper.py, load_celeba_dataset.py, load_fairface_dataset.py,
│   │   │   load_unsplash_dataset.py, generate_face_queries.py,
│   │   │   generate_bad_face_queries.py, reddit_*.json, reddit_profile/
│   │   └── test_reddit_scraper.py
│   ├── dataset_generator/            ← synthetic photos: Qwen-Image-2512 pipeline
│   │   ├── taxonomy.py               ← single source of truth for every variation matrix
│   │   ├── prompt_builder.py         ← composes prompt + label row from taxonomy.py
│   │   ├── qwen_pipeline.py          ← GPU-aware model loading + generate()
│   │   ├── smoke_test.py             ← --dry-run / --per-tier quick check, no GPU needed for --dry-run
│   │   ├── validation_sweep.py       ← --coverage-only (no GPU) / --check-binding gate
│   │   ├── full_run.py               ← the 28,000-image production run, resumable, shardable
│   │   ├── merge_shards.py
│   │   ├── extract_measured_labels.py← Bucket-C: pixel-measured colour, QA gates
│   │   ├── install.sh
│   │   └── PLAN.md                   ← full setup & execution guide — read this before running anything
│   ├── config.py                     ← shared paths/constants (both pipelines + the trainer)
│   ├── 04_train_coreml_models.py     ← Phase 2: PyTorch training + CoreML export
│   └── 05_finetune_real_world.py     ← Phase 3 (TODO)
├── models/                           ← exported .mlpackage artifacts for Xcode
└── data/
    ├── 1_Raw_Scrapes/                ← real, Phase 1 output
    ├── 2_VLM_Processing/             ← real, Phase 1 output
    ├── 3_CoreML_Training_Data/       ← real, the trainer's actual current input (see below)
    └── 4_Synthetic_Qwen/             ← synthetic, Qwen pipeline output
        ├── raw_generated/            ← copied back from the remote GPU box after a run
        └── qa_processed/             ← after extract_measured_labels.py
```

`4_Synthetic_Qwen` is deliberately numbered `4`, not folded into `1`-`3` — those are real photos, this is generated. Keeping them as visibly distinct siblings means a training run can never accidentally blend the two without that being an explicit, visible decision.

---

## Phase 1a: Real-world data (`real_data_pipeline/`)

Scrapes candidate photos from Reddit (subreddits and settings in `config.py`'s `REDDIT_SOURCES`) into `1_Raw_Scrapes/`, classifies them with a VLM (`config.py`'s `VLM_ENGINE`: `mlx_vlm` on Apple Silicon, `ollama`, or `gemini`) into `2_VLM_Processing/`, then sorts the kept ones into `3_CoreML_Training_Data/{Outfit,Face_Grooming}/{demographic}/{1_Needs_Improvement,2_Average,3_Polished}/` — a plain folder-per-class layout. `config.py`'s `DEMOGRAPHICS` (six age/gender buckets) and `AESTHETIC_TIERS` (the three folder names) define this contract exactly; the trainer (below) expects this structure precisely, not anything derived from it.

## Phase 1b: Synthetic data (`dataset_generator/`) — Qwen-Image-2512

Because labeling tens of thousands of real photos by hand doesn't scale, the second path procedurally generates a precisely-labeled synthetic dataset with Qwen-Image-2512.

**Why Qwen and not FLUX.1 [dev]** (the original choice, now archived at `ML/archive/dataset_generator_v7/`): both were tested head-to-head across a full 48-prompt comparison (also against Google's Nano Banana Pro). FLUX reliably collapsed "flaw" and "average" tier prompts toward its own aesthetic prior — it would render a person asked to look "greasy and unwashed" as merely a little tousled — which meant a meaningful share of any FLUX-generated flaw-tier data likely didn't visually match its own label. Qwen held up across all severity tiers with no collapse; Nano Banana Pro matched it on descriptor quality but had a ~19% identity-collapse rate (reusing one identity's face/age/ethnicity for a completely different requested one), which is disqualifying for automated label generation specifically.

The current taxonomy (v8, in `taxonomy.py`) fixes two further problems found by reviewing FLUX/Qwen output directly: flaw severity used to be one maximally-bad description (making "needs improvement" read as costume-level distress rather than a realistic bad day), and "polished" outfits were a visual jacket monoculture (only one archetype, always involving outerwear, contradicting the effort principle above — fit, crispness, and colour harmony matter, not formality). See `taxonomy.py`'s module docstring and `PLAN.md` for the full design: independent sampling axes (effort condition vs. garment identity vs. colour vs. environment — background/lighting/framing are sampled *independently of effort tier*, specifically so the model can't learn "polished = studio-lit" as a shortcut), coherent (not uniform-random) outfit-formality sampling, ordinal 0/1/2 effort flags, and a generated `label_schema_<Category>.json` per category as the CSV's contract.

Image budget: 28,000 total (Men/Women_Grooming 6,000 each, Men/Women_Outfit 8,000 each — outfit needs more so all 12 upper-garment types clear 250+ examples across all three tiers). **Full setup and run instructions are in `ML/pipeline/dataset_generator/PLAN.md`** — don't skip `smoke_test.py --dry-run` and `validation_sweep.py --coverage-only` before spending any GPU time.

**Before the full 28,000-image run, three escalating checks confirm the output is what's expected**, each cheaper to fix a problem in than the next: `quick_prompt_test.py` (6 hardcoded prompts, not built from the taxonomy — the fastest possible "does the model even load and generate on this box" check, and a stable reference for manually comparing Qwen against another model), `smoke_test.py --per-tier` (16 images, one per category x tier, checking the severity gradient reads as distinct), and `variation_test.py` (64 images, several samples per category x tier — checks diversity *within* a tier: different identities, garments, colours, environments, not just the gradient across tiers). All three write a manifest/images to a local, gitignored output folder for side-by-side review. See `PLAN.md`'s numbered run order for exactly when to run each one.

---

## ⚠ Current gap: the trainer does not read the synthetic pipeline's output yet

`04_train_coreml_models.py` is a plain `torchvision.datasets.ImageFolder` classifier — it reads training labels **from subfolder names only** (`1_Needs_Improvement`/`2_Average`/`3_Polished`), one 3-class softmax head per `(stream, demographic)`. It does not open any CSV. This means:

- `label_schema_<Category>.json` and the per-category CSVs that `dataset_generator/` produces are **the intended interface for a future trainer rewrite** — a continuous score-regression head plus multi-label attribute heads, one per schema entry, `SmoothL1` on score at weight 1.0 and cross-entropy on the rest at 0.3–0.5 (score must stay dominant, or the auxiliary heads drown the regression, which is the actual product). That rewrite has not been done.
- Generating the full 28,000-image synthetic dataset today would **not**, by itself, feed the current trainer. Either the trainer needs rewriting to consume the schema/CSV contract, or a bridging step needs to convert synthetic output + its tier label into the same `ImageFolder`-per-class layout `3_CoreML_Training_Data/` already uses (lossy — it would throw away the richer attribute/formality/colour labels down to just the 3-class tier).

This is a known, deliberate gap, not an oversight — decide which path (trainer rewrite vs. bridging script) before running the full 28,000-image generation, since the answer changes what's worth generating.

---

## Phase 2: Training (current state)

Script: `pipeline/04_train_coreml_models.py`

- `torchvision.datasets.ImageFolder` input, one model per `(stream, demographic)` — 2 streams (`Outfit`, `Face_Grooming`) × 6 age/gender demographics = 12 models, not 4
- Single-head 3-class softmax (`1_Needs_Improvement`/`2_Average`/`3_Polished`), `CrossEntropyLoss` — not the two-headed score-regression + multi-label design the synthetic pipeline's schema is built for (see the gap noted above)
- Exported via `coremltools` to `.mlpackage`, compatible with iOS 17+

## Phase 3: Real-World Fine-Tuning (TODO)

Script: `pipeline/05_finetune_real_world.py` (to be created). Fine-tune on real-world images to adapt from synthetic/scraped training conditions to real phone-camera photos.

---

## Phase 4: On-device stylist LLM (`stylist_llm/`)

A second, fully isolated model and pipeline — **not** the vision model above. The vision model scores a photo and outputs tags; this one takes those tags plus a user-selected occasion and generates one specific, single-shot "5-minute fix" in under 50 words, entirely on-device (no cloud call, no multi-turn chat). Base model `HuggingFaceTB/SmolLM2-135M-Instruct`, vocabulary-pruned, fully fine-tuned (not LoRA), exported as a stateful INT4 CoreML package targeting **iOS 18+** (this bumps the app's minimum deployment target beyond the vision model's iOS17 — a deliberate, confirmed product decision, not an oversight).

Isolation is real: separate code (`ML/pipeline/stylist_llm/`), separate config (its own `config.py`, not the shared one above), separate data folder (`ML/data/5_Stylist_LLM/`), separate checkpoints — with exactly one deliberate, one-directional exception: `tag_vocabulary.py` reads (never writes) `dataset_generator/taxonomy.py`'s real label schema, so the LLM is trained on the tag format the vision model actually produces rather than an invented one. Full setup, run order, and a "corrections from the original brief" log (several claims in the original spec — parameter count, model size, the input-token bound, the quantization API — didn't hold up against this codebase or real testing) are in **`ML/pipeline/stylist_llm/PLAN.md`**.

---

## iOS Integration

- Models run fully **on-device** (no internet dependency)
- Camera framing enforced by AVFoundation overlay (oval for grooming, rectangle for outfit)
- Lighting check via `CMSampleBuffer` EXIF `brightnessValue` before inference
- Contextual improvement advice is generated by the **on-device stylist LLM** (`stylist_llm/`, above) from the vision model's tags — not a static native Swift rules engine and not a cloud LLM call
