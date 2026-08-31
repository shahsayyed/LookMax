# LookMax Dataset Generation Plan (FLUX.1 [dev])

This folder contains the scripts needed to procedurally generate ~24,000 synthetic images for your LookMax multi-task AI models. It uses **FLUX.1 [dev]** and writes all images and labels directly to a single directory structure.

## Overview of the Taxonomy (Prompt Matrix)
The python script `generate_flux_dataset.py` contains the comprehensive prompt matrix. It generates a diverse pool of identities (combining age, ethnicity, and body type) and maps them to varying execution levels (Scores 2, 3, 5, 10). 

It covers all 4 CoreML target models:
1. `Men_Grooming`
2. `Women_Grooming`
3. `Men_Outfit`
4. `Women_Outfit`

As it generates the images, it writes the exact score (1-10) and binary checklist attributes (e.g. `hair_messy`, `clothes_wrinkled`) to a `labels.csv` file automatically.

## Infrastructure Setup (Vast.ai or Similar)

Since you are renting a machine (e.g., RTX 4090 or 5090 with 24GB+ VRAM), follow these steps:

### 1. Provision the Machine
1. Rent an Ubuntu instance with an RTX 4090 (24GB VRAM).
2. Ensure the instance has at least **100GB of free disk space** (24,000 PNGs at 1024x1024 will take ~60-80GB).
3. Use an image that has PyTorch and CUDA pre-installed (e.g., `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`).

### 2. Transfer the Files
Transfer this directory to your rented machine. From your local Mac terminal:
```bash
scp -r /Users/sayyed/development/repos/LookMax/ML/pipeline/dataset_generator root@<SERVER_IP>:/workspace/
```

### 3. Install Dependencies on Server
SSH into the server and run:
```bash
cd /workspace/dataset_generator
pip install -r requirements.txt
```

### 4. HuggingFace Authentication
FLUX.1 [dev] is an open-weights model, but it is "gated" by its creators (Black Forest Labs).
1. Go to [HuggingFace FLUX.1 [dev]](https://huggingface.co/black-forest-labs/FLUX.1-dev) and click "Agree and access repository".
2. Go to your HuggingFace Settings -> Access Tokens and create a "Read" token.
3. Export the token on your server:
```bash
export HF_TOKEN="hf_your_token_here"
```

### 5. Run the Generation Script
We highly recommend running this inside a `tmux` session or using `nohup` so that if your SSH connection drops, the generation continues!

```bash
# Start a tmux session
tmux new -s fluxgen

# Run the script
python3 generate_flux_dataset.py
```
*(To detach from tmux and leave it running, press `Ctrl+B`, then press `D`.)*

## Timeline & Cost Expectation
* **Time per image:** ~10 to 14 seconds on an RTX 4090.
* **Total Time:** 24,000 images × 12 seconds = 288,000 seconds = **~80 hours** (3.3 days) of continuous generation.
* **Cost:** An RTX 4090 on Vast.ai costs about $0.25 - $0.40/hour. 80 hours will cost roughly **$20 to $32 total**.

## Downloading the Result
Once the script completes, the `dataset_output` folder will contain the `images/` directory and your perfectly labeled `labels.csv`.
Download it back to your Mac:
```bash
scp -r root@<SERVER_IP>:/workspace/dataset_generator/dataset_output /Users/sayyed/development/repos/LookMax/ML/data/
```
