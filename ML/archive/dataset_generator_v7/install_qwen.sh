#!/bin/bash
# ==========================================
# Qwen-Image-2512 setup for a Vast.ai box.
#
# IMPORTANT DISK NOTE: On these Vast.ai images, /workspace is mapped to
# a tiny ~10GB loop device, while / (root) is the actual large disk you
# allocated. This script checks /data specifically (creating it if
# missing) rather than trusting the current working directory or shell
# env -- those have both proven unreliable across sessions on these
# boxes. test_qwen_variations.py independently pins its own HF_HOME to
# /data/huggingface_cache the same way, so as long as /data is really
# your large disk, neither script can end up writing to /workspace.
# ==========================================
set -e

MIN_FREE_GB=60
DATA_DIR=/data

mkdir -p "$DATA_DIR"
AVAILABLE_KB=$(df -Pk "$DATA_DIR" | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

echo "==> Checking: $DATA_DIR"
echo "==> Free space there: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
    echo ""
    echo "!! Only ${AVAILABLE_GB}GB free on $DATA_DIR -- Qwen-Image-2512's full pipeline"
    echo "!! is ~58GB on disk. Run 'df -h' and confirm $DATA_DIR is actually your LARGE"
    echo "!! disk on this machine (NOT /workspace -- that's a small loop device here)."
    echo "!! If your large disk is mounted somewhere else, edit DATA_DIR at the top of"
    echo "!! this script AND the matching DATA_DIR in test_qwen_variations.py."
    exit 1
fi

if ! python3 -c "import torch" &> /dev/null; then
    echo ""
    echo "!! No torch found in this environment. This script deliberately does NOT"
    echo "!! install torch itself -- installing the wrong CUDA build silently breaks"
    echo "!! GPU support. This means either:"
    echo "!!   (a) this instance wasn't launched from a PyTorch-preinstalled template"
    echo "!!       (e.g. Vast.ai's 'PyTorch (Vast)' template) -- easiest fix is to"
    echo "!!       relaunch from that template, or"
    echo "!!   (b) you need to install torch by hand, matched to this GPU's driver:"
    echo "!!       run 'nvidia-smi', note the 'CUDA Version' in the top-right box,"
    echo "!!       then e.g. for CUDA 12.8:"
    echo "!!         pip install torch --index-url https://download.pytorch.org/whl/cu128"
    echo "!!       (swap cu128 for the closest wheel <= your driver's reported version)"
    exit 1
fi

echo "==> Installing diffusers from source (needed for QwenImagePipeline) + Qwen-Image deps"
pip install --upgrade "git+https://github.com/huggingface/diffusers"
pip install --upgrade "transformers>=4.51.3"
pip install accelerate sentencepiece protobuf hf_transfer

echo ""
echo "==> Setup complete. No need to export HF_HOME -- test_qwen_variations.py pins it"
echo "    to $DATA_DIR/huggingface_cache itself, unconditionally, regardless of shell env."
echo ""
echo "Qwen-Image-2512 is Apache 2.0 and NOT a gated repo, so HF_TOKEN is optional."
echo ""
echo "Run inside tmux since the first run downloads ~58GB of weights:"
echo "    tmux new -s qwengen"
echo "    python3 test_qwen_variations.py"
