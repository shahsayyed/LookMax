#!/bin/bash
# ==============================================================================
# run.sh -- Unified remote control script for LookMax synthetic dataset generation.
#
# Usage:
#   ./run.sh setup        # Install dependencies and setup environment
#   ./run.sh test         # 6-prompt GPU sanity check (quick_prompt_test.py)
#   ./run.sh smoke        # 16-image per-tier test (smoke_test.py --per-tier)
#   ./run.sh benchmark    # Compare sequential vs batched vs 2 workers
#   ./run.sh variation    # 64-image diversity check (variation_test.py)
#   ./run.sh start [N]    # Launch full 28,000 run with N workers (default: 2) in tmux
#   ./run.sh status       # Show live generation progress & GPU utilization
#   ./run.sh attach       # Attach to the background generation tmux session
#   ./run.sh stop         # Safely stop generation
#   ./run.sh measure      # Run Bucket C pixel/QA measurements on generated CSVs
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------------------
# Environment defaults
# ------------------------------------------------------------------------------
if [ -d "/venv/main/bin" ]; then
    export PATH="/venv/main/bin:$PATH"
fi

export DATA_DIR="${LOOKMAX_DATA_DIR:-/data}"
export HF_HOME="$DATA_DIR/huggingface_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHONUNBUFFERED=1
export LOOKMAX_DATA_DIR="$DATA_DIR"

OUTPUT_DIR="$DATA_DIR/qwen_dataset_output"
IMAGES_DIR="$OUTPUT_DIR/images"
TMUX_SESSION="lookmax_gen"
TOTAL_TARGET=28000

# ------------------------------------------------------------------------------
# Subcommands
# ------------------------------------------------------------------------------
case "$1" in
    setup)
        echo "==> Running environment and dependency installer..."
        bash "$SCRIPT_DIR/install.sh"
        ;;

    test)
        echo "==> Running 6-image GPU quick prompt test..."
        python3 "$SCRIPT_DIR/quick_prompt_test.py" --output-dir "$DATA_DIR/quick_prompt_test_output"
        ;;

    smoke)
        echo "==> Running 16-image per-tier test..."
        python3 "$SCRIPT_DIR/smoke_test.py" --per-tier --output-dir "$DATA_DIR/smoke_test_output"
        ;;

    benchmark)
        DEFAULT_WORKERS=1
        if command -v nvidia-smi >/dev/null 2>&1; then
            GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
            if [ -n "$GPU_COUNT" ] && [ "$GPU_COUNT" -gt 1 ]; then
                DEFAULT_WORKERS="$GPU_COUNT"
            fi
        fi
        WORKERS="${2:-$DEFAULT_WORKERS}"
        echo "==> Running hardware benchmark with $WORKERS worker(s)..."
        python3 "$SCRIPT_DIR/full_run.py" --benchmark --num-workers "$WORKERS" --data-dir "$DATA_DIR"
        ;;

    variation)
        WORKERS="${2:-2}"
        echo "==> Running 64-image variation diversity test with $WORKERS worker(s)..."
        python3 "$SCRIPT_DIR/variation_test.py" --num-workers "$WORKERS" --output-dir "$DATA_DIR/variation_test_output"
        ;;

    start)
        WORKERS="${2:-1}"
        TASK_RANGE="${3:-}"
        SHARD_ID="${4:-}"
        
        EXTRA_ARGS=""
        if [ -n "$TASK_RANGE" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --task-range $TASK_RANGE"
            echo "==> Restricting task range to [$TASK_RANGE]..."
        fi
        if [ -n "$SHARD_ID" ]; then
            EXTRA_ARGS="$EXTRA_ARGS --shard $SHARD_ID"
            echo "==> Shard ID set to $SHARD_ID (outputs saved with _shard$SHARD_ID suffix)..."
        fi

        echo "==> Launching image generation with $WORKERS worker(s)$EXTRA_ARGS..."
        mkdir -p "$IMAGES_DIR"

        CMD="python3 full_run.py --num-workers $WORKERS --data-dir '$DATA_DIR'$EXTRA_ARGS"

        if command -v tmux >/dev/null 2>&1; then
            if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
                echo "!! A tmux session '$TMUX_SESSION' is ALREADY running."
                echo "   Run './run.sh status' or './run.sh attach' to view it."
                exit 1
            fi
            echo "==> Starting generation inside detached tmux session: $TMUX_SESSION"
            tmux new-session -d -s "$TMUX_SESSION" \
                "export HF_HOME='$HF_HOME'; export HF_HUB_ENABLE_HF_TRANSFER=1; export LOOKMAX_DATA_DIR='$DATA_DIR'; cd '$SCRIPT_DIR'; $CMD; echo 'Generation finished. Press Enter to close.'; read"
            echo "✓ Session started! Commands to monitor:"
            echo "   ./run.sh status   (View live progress)"
            echo "   ./run.sh attach   (Attach to tmux session)"
        else
            echo "==> tmux not found, launching in background via nohup..."
            nohup python3 "$SCRIPT_DIR/full_run.py" --num-workers "$WORKERS" --data-dir "$DATA_DIR" $EXTRA_ARGS > "$OUTPUT_DIR/generation.log" 2>&1 &
            echo "✓ Process started (PID: $!). Monitor with './run.sh status'."
        fi
        ;;

    status)
        echo "================================================================================"
        echo " LookMax Synthetic Image Generation Status"
        echo "================================================================================"
        if [ -f "$SCRIPT_DIR/.auto_resume_config" ]; then
            source "$SCRIPT_DIR/.auto_resume_config"
        fi
        TARGET_LABEL="$TOTAL_TARGET"
        if [ -n "$TASK_RANGE" ]; then
            START_IDX=$(echo "$TASK_RANGE" | cut -d':' -f1)
            END_IDX=$(echo "$TASK_RANGE" | cut -d':' -f2)
            TOTAL_TARGET=$((END_IDX - START_IDX))
            TARGET_LABEL="$TOTAL_TARGET (Slice [$TASK_RANGE])"
        fi

        if [ -d "$IMAGES_DIR" ]; then
            # Count only finished png files (ignore .tmp)
            DONE_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
        else
            DONE_COUNT=0
        fi

        REMAINING=$((TOTAL_TARGET - DONE_COUNT))
        PERCENT=$(awk "BEGIN {printf \"%.2f\", ($DONE_COUNT / $TOTAL_TARGET) * 100}")

        echo " Output Dir  : $OUTPUT_DIR"
        echo " Completed   : $DONE_COUNT / $TARGET_LABEL ($PERCENT%)"
        echo " Remaining   : $REMAINING images"
        echo ""
        echo " Category Breakdown on Disk:"
        for cat in Men_Grooming Women_Grooming Men_Outfit Women_Outfit; do
            if [ -d "$IMAGES_DIR" ]; then
                CAT_COUNT=$(find "$IMAGES_DIR" -maxdepth 1 -name "*_${cat}_*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
            else
                CAT_COUNT=0
            fi
            echo "   - $cat : $CAT_COUNT"
        done
        echo ""

        # Check tmux session
        if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            echo " Background Task: RUNNING in tmux session '$TMUX_SESSION'"
        else
            if pgrep -f "full_run.py" >/dev/null 2>&1; then
                echo " Background Task: RUNNING (PID $(pgrep -f "full_run.py" | head -1))"
            else
                echo " Background Task: IDLE / NOT RUNNING"
            fi
        fi

        echo ""
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo " GPU State:"
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader | awk -F', ' '{print "   " $1 " | VRAM: " $2 " / " $3 " | Util: " $4 " | Temp: " $5}'
        fi
        echo "================================================================================"
        ;;

    attach)
        if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            tmux attach-session -t "$TMUX_SESSION"
        else
            echo "No active tmux session '$TMUX_SESSION' found."
            echo "Run './run.sh status' to inspect current processes."
        fi
        ;;

    stop)
        echo "==> Stopping generation process..."
        if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
            tmux kill-session -t "$TMUX_SESSION"
            echo "✓ Killed tmux session '$TMUX_SESSION'."
        fi
        pkill -f "full_run.py" || true
        echo "✓ Generation stopped."
        ;;

    measure)
        echo "==> Running extract_measured_labels.py on all 4 category CSVs..."
        for cat in Men_Grooming Women_Grooming Men_Outfit Women_Outfit; do
            CSV_PATH="$OUTPUT_DIR/labels_${cat}.csv"
            if [ -f "$CSV_PATH" ]; then
                echo "Measuring $cat..."
                python3 "$SCRIPT_DIR/extract_measured_labels.py" "$CSV_PATH" "$IMAGES_DIR" --output "$OUTPUT_DIR/labels_${cat}_measured.csv"
            else
                echo "Skipping $cat (CSV not found at $CSV_PATH)"
            fi
        done
        echo "✓ Measurement complete!"
        ;;

    resume)
        echo "==> Running interruption recovery and auto-resume check..."
        bash "$SCRIPT_DIR/auto_resume.sh"
        ;;

    *)
        echo "LookMax Synthetic Dataset Generator CLI"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  setup              Run install.sh to setup Python dependencies"
        echo "  test               Run 6-prompt GPU sanity check"
        echo "  smoke              Run 16-image per-tier test"
        echo "  benchmark [N]      Benchmark GPU(s) (auto-detects all GPUs or N workers)"
        echo "  variation [N]      Run 64-image diversity check (default: 2 workers)"
        echo "  start [N] [RANGE]  Start generation with N workers in tmux (default: 2)"
        echo "  resume             Check GPU, clean dangling tmp files, and resume generation"
        echo "  status             Check live progress, image counts, and GPU state"
        echo "  attach             Attach to running tmux session"
        echo "  stop               Stop running generation"
        echo "  measure            Extract pixel-level color and QA metrics (Bucket C)"
        echo ""
        exit 1
        ;;
esac
