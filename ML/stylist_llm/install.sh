#!/bin/bash
# ==========================================================================
# Stylist LLM pipeline setup. Much lighter than dataset_generator/install.sh
# -- SmolLM2-135M-Instruct is a few hundred MB (not ~58GB), and the brief's
# own compute budget (<6GB VRAM, ~12 min on a T4/RTX, ~20 min locally on
# Apple Silicon via PyTorch MPS) means this can run on a laptop, not just a
# rented GPU box. No disk-quirk handling needed for that reason.
# ==========================================================================
set -e

echo "==> Installing stylist_llm/ dependencies"
pip install -r "$(dirname "$0")/requirements.txt"

echo ""
echo "==> Setup complete."
echo ""
echo "Next steps (see PLAN.md for the full run order):"
echo "    python3 tag_vocabulary.py           # (no CLI -- imported by the scripts below)"
echo "    python3 generate_synthetic_dataset.py --dry-run --count 10"
echo "    python3 prune_vocabulary.py --dry-run"
echo ""
echo "Real generation needs a Gemini API key:"
echo "    export GEMINI_API_KEY=\"your-key-here\""
echo "    python3 generate_synthetic_dataset.py"
echo ""
echo "This pipeline is fully isolated from ML/vision/dataset_synthetic/ (the vision image"
echo "pipeline) -- separate requirements.txt, separate data folder (ML/data/stylist_llm/),"
echo "separate checkpoints -- except tag_vocabulary.py's deliberate, one-directional read of"
echo "the vision taxonomy for the input tag-format contract. See PLAN.md."
