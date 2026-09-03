#!/bin/bash
# ==========================================================================
# Qwen-Image-2512 setup for a remote GPU box (e.g. Vast.ai).
#
# DISK NOTE (carried forward from ML/archive/dataset_generator_v7/install_qwen.sh):
# On Vast.ai images, /workspace is commonly mapped to a tiny ~10GB loop
# device, while / (root) is the actual large disk you allocated. This
# script checks DATA_DIR specifically (creating it if missing) rather than
# trusting the current working directory or shell env -- both have
# proven unreliable across sessions on these boxes in past runs.
# full_run.py independently checks the same DATA_DIR at generation time,
# so as long as DATA_DIR is really your large disk, nothing in this
# pipeline can end up writing to /workspace.
# ==========================================================================
set -e

MIN_FREE_GB=150   # ~58GB model cache + ~28,000 images (grooming 1024x1024, outfit 768x1024) + headroom
DATA_DIR="${LOOKMAX_DATA_DIR:-/data}"

mkdir -p "$DATA_DIR"
AVAILABLE_KB=$(df -Pk "$DATA_DIR" | tail -1 | awk '{print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))

echo "==> Checking: $DATA_DIR"
echo "==> Free space there: ${AVAILABLE_GB}GB"

if [ "$AVAILABLE_GB" -lt "$MIN_FREE_GB" ]; then
    echo ""
    echo "!! Only ${AVAILABLE_GB}GB free on $DATA_DIR -- the full 28,000-image run needs at least"
    echo "!! ${MIN_FREE_GB}GB (Qwen-Image-2512's full pipeline is ~58GB on disk, plus the images"
    echo "!! themselves). Run 'df -h' and confirm $DATA_DIR is actually your LARGE disk on this"
    echo "!! machine (NOT /workspace -- that's frequently a small loop device on Vast.ai)."
    echo "!! If your large disk is mounted somewhere else, set LOOKMAX_DATA_DIR before running this"
    echo "!! script, e.g.:  LOOKMAX_DATA_DIR=/mnt/big ./install.sh"
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
pip install accelerate sentencepiece protobuf hf_transfer opencv-python-headless scikit-learn tqdm

echo ""
echo "==> Setup complete."
echo ""
echo "IMPORTANT: HF_HOME and LOOKMAX_DATA_DIR must be set explicitly before running anything that"
echo "downloads the model or writes output -- do NOT trust the ambient shell env to already have"
echo "these; three separate past incidents (see PLAN.md) proved that unreliable across"
echo "sessions/reattaches on these boxes. full_run.py's own default output dir is a LOCAL folder"
echo "next to the script (so it also works on a laptop with no /data mount) -- on this box you must"
echo "override it, in the SAME shell you run the Python scripts from:"
echo ""
echo "    export HF_HOME=\"$DATA_DIR/huggingface_cache\""
echo "    export HF_XET_HIGH_PERFORMANCE=1"
echo "    export LOOKMAX_DATA_DIR=\"$DATA_DIR\""
echo ""
echo "Qwen-Image-2512 is Apache 2.0 and NOT a gated repo, so HF_TOKEN is optional."
echo ""
echo "Run everything inside tmux (the first run downloads ~58GB of weights, and the full run takes"
echo "many hours -- tmux survives a dropped connection, your shell does not):"
echo "    tmux new -s qwengen"
echo "    export HF_HOME=\"$DATA_DIR/huggingface_cache\""
echo "    python3 smoke_test.py --dry-run"
echo "    python3 smoke_test.py --per-tier"
echo "    python3 validation_sweep.py --coverage-only"
echo "    python3 full_run.py --benchmark"
echo "    python3 full_run.py"
