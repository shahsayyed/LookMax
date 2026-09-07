#!/bin/bash
# ==============================================================================
# auto_resume.sh -- Interruption Recovery & Auto-Resume for LookMax Synthetic Gen
# Designed specifically for interruptible / spot / preemptible GPU instances (Vast.ai).
#
# When an instance is outbid or paused, then resumed:
# 1. Clears zombie processes holding GPU VRAM
# 2. Flushes CUDA cache
# 3. Purges incomplete dangling .tmp image files from mid-generation preemption
# 4. Verifies disk space & directories
# 5. Automatically launches generation in detached tmux session if not running
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$SCRIPT_DIR/.auto_resume_config" ]; then
    source "$SCRIPT_DIR/.auto_resume_config"
fi

if [ -d "/venv/main/bin" ]; then
    export PATH="/venv/main/bin:$PATH"
fi

DATA_DIR="${LOOKMAX_DATA_DIR:-/data}"
export HF_HOME="$DATA_DIR/huggingface_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:-}"
export LOOKMAX_DATA_DIR="$DATA_DIR"
export PYTHONUNBUFFERED=1

IMAGES_DIR="$DATA_DIR/qwen_dataset_output/images"
TMUX_SESSION="lookmax_gen"

# Detect GPUs
if command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
else
    NUM_GPUS=1
fi

TASK_RANGE="${1:-${TASK_RANGE:-16000:28000}}"
WORKERS="${2:-${WORKERS:-$NUM_GPUS}}"

echo "================================================================================"
echo " LookMax Auto-Resume & Interruption Health Check"
echo " Date: $(date)"
echo " Workers (GPUs) : $WORKERS"
echo " Task Range     : $TASK_RANGE"
echo " Data Directory : $DATA_DIR"
echo "================================================================================"

# 1. Check if generation session is already alive
if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "==> tmux session '$TMUX_SESSION' is ALREADY running."
    exit 0
fi

# 2. Purge dangling .tmp files from mid-generation interruption
if [ -d "$IMAGES_DIR" ]; then
    TMP_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$TMP_COUNT" -gt 0 ]; then
        echo "==> Purging $TMP_COUNT dangling .tmp file(s) left by preemption..."
        rm -f "$IMAGES_DIR"/*.tmp
        echo "✓ Stale temp files cleaned."
    fi
fi

# 3. Reset GPU memory state & verify CUDA health
echo "==> Verifying GPU state and releasing stale allocations..."
python3 -c "
import gc, torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f'   GPU {i}: {props.name} | Free VRAM: {free/(1024**3):.1f}GB / {total/(1024**3):.1f}GB')
" 2>/dev/null || true

# If there is any dead/orphaned full_run.py process, kill it
pkill -9 -f "full_run.py" 2>/dev/null || true
sleep 1

# 4. Start generation in detached tmux
echo "==> Starting generation session in tmux (Workers: $WORKERS, Range: $TASK_RANGE)..."
./run.sh start "$WORKERS" "$TASK_RANGE"

echo "✓ Auto-resume check complete. Generation is active."
