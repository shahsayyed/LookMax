#!/bin/bash
# ==========================================
# Qwen-Image-2512 setup for the same Vast.ai RTX 5090 box used for Flux.
# Run this ONCE on a fresh instance, or in a separate venv, before
# running test_qwen_variations.py.
#
# Why a separate install: diffusers needs a very recent (unreleased-pin)
# build to have QwenImagePipeline, and requirements.txt for the Flux
# scripts deliberately avoids touching the pre-installed PyTorch/CUDA
# build on Vast.ai images. This script follows the same rule -- it only
# adds packages on top, never reinstalls torch.
# ==========================================
set -e

echo "==> Reusing the /data convention from PLAN.md (Vast.ai maps /workspace to a tiny 10GB disk)"
mkdir -p /data
cd /data

if [ ! -d "/data/dataset_generator" ]; then
    echo "==> /data/dataset_generator not found -- copying from /workspace"
    cp -r /workspace/dataset_generator /data/
fi
cd /data/dataset_generator

echo "==> Installing diffusers from source (needed for QwenImagePipeline) + Qwen-Image deps"
pip install --upgrade "git+https://github.com/huggingface/diffusers"
pip install --upgrade "transformers>=4.51.3"
pip install accelerate sentencepiece protobuf hf_transfer

echo "==> Setup complete. Before running test_qwen_variations.py, export:"
echo '    export HF_HOME="/data/huggingface_cache"'
echo '    export HF_XET_HIGH_PERFORMANCE=1'
echo ""
echo "Qwen-Image-2512 is Apache 2.0 and NOT a gated repo, so HF_TOKEN is optional"
echo "(only needed if your HF account has custom download rate limits)."
echo ""
echo "Run inside tmux since the first run downloads ~40GB of weights:"
echo "    tmux new -s qwengen"
echo "    python3 test_qwen_variations.py"
