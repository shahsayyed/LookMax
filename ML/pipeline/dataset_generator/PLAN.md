# Dataset Generator: FLUX.1 [dev] Synthetic Image Pipeline

This folder contains all scripts for generating the **24,000-image synthetic training dataset** for LookMax using FLUX.1 [dev].

---

## Scripts

| Script | Purpose |
|---|---|
| `generate_flux_dataset.py` | Master generation script — generates all 24,000 images and writes `labels.csv` |
| `test_flux_prompts.py` | Quick 4-image sanity check (one per model category) |
| `test_flux_variations.py` | Full 48-image demographic + level validation test |
| `requirements.txt` | Python dependencies for the remote GPU server |

---

## Remote Server Setup (Vast.ai)

We run generation on a **rented GPU server** via [Vast.ai](https://vast.ai) because FLUX.1 [dev] requires ~24 GB VRAM and takes ~19 seconds per image at 1024×1024 resolution.

### Hardware Used
- **GPU:** RTX 5090 (32 GB VRAM)
- **VRAM Note:** The full FLUX.1 [dev] pipeline requires ~33 GB when the T5 text encoder is included. We use `pipe.enable_model_cpu_offload()` to keep the image generator on GPU (24 GB) and offload the text encoder to system RAM. This is the maximum safe configuration for a 32 GB card.
- **Disk:** 150 GB allocated (IMPORTANT: see disk quirk below)
- **OS/Image:** `vastai/pytorch:cuda-12.8.1-auto` (Jupyter interface)

### Critical Disk Quirk on Vast.ai
The host mounts the large disk to `/` (root), but maps `/workspace` to a tiny 10 GB loop device. **Do NOT use `/workspace` for model weights or outputs.** Use `/data` instead:

```bash
# Create the working directory on the large drive
mkdir /data
cp -r /workspace/dataset_generator /data/
cd /data/dataset_generator
```

### Required Environment Variables
```bash
export HF_TOKEN="hf_your_token_here"         # HuggingFace access token (FLUX.1 [dev] is gated)
export HF_HOME="/data/huggingface_cache"      # Redirect 33GB model download to large drive
export HF_XET_HIGH_PERFORMANCE=1              # Enable Xet high-speed multi-threaded download
```

> **Note on HF_TRANSFER:** The older `HF_HUB_ENABLE_HF_TRANSFER=1` env var is deprecated. Use `HF_XET_HIGH_PERFORMANCE=1` which uses the newer Xet downloader achieving 40–80 MB/s vs the default ~2 MB/s single-stream.

### Install Dependencies
```bash
pip install -r requirements.txt
```
> `requirements.txt` intentionally does NOT pin a PyTorch version. The Vast.ai image ships with an optimized CUDA build of PyTorch — overwriting it breaks the GPU drivers. We install only the missing HuggingFace libraries on top.

---

## Running the Generation

### Step 1: Test First (Recommended)
Always validate prompts before committing to the full 24,000-image run:

```bash
# Quick 4-image test (one per model category)
python3 test_flux_prompts.py

# Full 48-image demographic + level validation
python3 test_flux_variations.py
```

### Step 2: Run inside tmux (Critical)
The full generation takes ~130 hours. Run inside `tmux` so the process survives browser disconnections:

```bash
tmux new -s fluxgen

# Inside tmux:
export HF_TOKEN="hf_your_token_here"
export HF_HOME="/data/huggingface_cache"
export HF_XET_HIGH_PERFORMANCE=1
python3 generate_flux_dataset.py

# Detach safely (leaves process running):
# Press Ctrl+B, then D

# Reattach later:
tmux attach -t fluxgen
```

### Step 3: Monitor Progress
The script prints `[idx/24000] Generating: filename...` for every image. Check the `dataset_output/` folder size periodically.

---

## Prompt Taxonomy (v2 — Effort-Based)

### Design Principles
1. **Effort only, no biology:** Flaws never include biological traits (acne, face shape, body type). We only penalise choices the user can control: greasy/unstyled hair, sloppy clothing, unblended makeup.
2. **Polished ≠ Formal:** A polished score does NOT require a suit. A perfectly fitted streetwear outfit with pristine fabrics scores 10/10. The model rates **fit, crispness, and color harmony**.
3. **Average = no effort:** Average is defined as "clean but zero styling" — no product, flat hair, standard jeans. This creates a clear visual gap between Average and Polished.

### Identity Matrix
Identities combine **age × ethnicity × body type × face shape** to cover diverse real-world users:

```
Ages:        22, 28, 35, 45, 55
Ethnicities: Caucasian, Black, East Asian, South Asian, Hispanic, Middle Eastern
Body types:  slim, athletic, average build, heavy set, muscular (men)
             slim, curvy, athletic, average build, plus size (women)
Face shapes: explicitly included to prevent model from learning genetic bias
```

### Score Scale
| Score | Label | Description |
|---|---|---|
| 1–3 | Flaw | Zero grooming effort, sloppy clothing, obvious neglect |
| 4–6 | Average | Clean but completely unstyled, no products, basic clothes |
| 7–10 | Polished | High-effort styling, product use, excellent fit and color coordination |

---

## Output Structure

```
dataset_output/
├── images/
│   ├── 00001_Men_Grooming.png
│   ├── 00002_Women_Outfit.png
│   └── ...
└── labels.csv      ← Auto-written during generation; safe to resume after interruption
```

`labels.csv` columns: `filename, category, score, [binary attribute flags]`

---

## Timeline & Cost

| Metric | Value |
|---|---|
| Time per image (RTX 5090) | ~19 seconds |
| Total images | 24,000 |
| **Total time** | **~130 hours** |
| Server cost (Vast.ai RTX 5090) | ~\$0.41/hr |
| **Estimated total cost** | **~\$53** |

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Tried `pipe.to("cuda")` — model exceeds 32GB VRAM | Use `pipe.enable_model_cpu_offload()` |
| `No space left on device` | HuggingFace cache defaulted to `/root/.cache` on 10GB loop drive | Set `HF_HOME="/data/huggingface_cache"` |
| Slow download (~2 MB/s) | Single-stream TCP throttling by HuggingFace CDN | Set `HF_XET_HIGH_PERFORMANCE=1` |
| `HF_HUB_ENABLE_HF_TRANSFER deprecated` | Old env var replaced by Xet system | Use `HF_XET_HIGH_PERFORMANCE=1` instead |
| `File reconstruction error: IO Error` | Corrupted partial download from force-killed process | `rm -rf /data/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev` and re-run |
| Schnell ignoring flaw prompts | "Prompt collapse" in 4-step distilled models | Must use FLUX.1 [dev] (28 steps) |

---

## Cleanup Commands

```bash
# Delete old model weights to free space
rm -rf /data/huggingface_cache/hub/models--black-forest-labs--FLUX.1-dev
rm -rf /data/huggingface_cache/hub/models--black-forest-labs--FLUX.1-schnell

# Wipe test image folders for a clean re-run
rm -rf /data/dataset_generator/test_images/*
rm -rf /data/dataset_generator/test_variations_comprehensive/*

# Wipe full dataset output for a fresh generation run
rm -rf /data/dataset_generator/dataset_output/*

# Remove old workspace files from the 10GB drive
rm -rf /workspace/dataset_generator
rm -rf ~/.cache/huggingface
```
