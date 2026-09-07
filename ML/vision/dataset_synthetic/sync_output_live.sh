#!/bin/bash
# ==============================================================================
# sync_output_live.sh -- External, continuous background sync script.
# Runs on your LOCAL Mac to mirror newly generated images and CSV files from
# the remote GPU box as they are produced.
#
# COMPLETELY DECOUPLED:
#   - Does not affect or slow down image generation on the GPU.
#   - If Wi-Fi disconnects or Mac sleeps, generation on the GPU keeps running.
#   - When reconnected, rsync catches up seamlessly.
#   - Excludes '*.tmp' files so partially-written images are NEVER copied.
#
# Usage:
#   ./sync_output_live.sh                                # Uses 'vast' SSH host alias
#   ./sync_output_live.sh root@<IP> <PORT>               # Explicit host and port
#   ./sync_output_live.sh vast 22 /path/to/local/dir 15  # Custom directory and 15s interval
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Parse mode and options
MODE="full"
REMOTE_HOST="vast"
REMOTE_PORT=""
LOCAL_DEST=""
INTERVAL_SEC="30"

while [[ $# -gt 0 ]]; do
    case "$1" in
        test|--test)
            MODE="test"
            shift
            ;;
        smoke|--smoke)
            MODE="smoke"
            shift
            ;;
        variation|--variation)
            MODE="variation"
            shift
            ;;
        --interval)
            INTERVAL_SEC="$2"
            shift 2
            ;;
        --host)
            REMOTE_HOST="$2"
            shift 2
            ;;
        --port)
            REMOTE_PORT="$2"
            shift 2
            ;;
        *)
            if [ -z "$LOCAL_DEST" ]; then
                LOCAL_DEST="$1"
            fi
            shift
            ;;
    esac
done

# Configure remote source and target directories based on mode
case "$MODE" in
    test)
        REMOTE_SRC="${REMOTE_DATA_DIR:-/data}/quick_prompt_test_output/"
        LOCAL_DEST="${LOCAL_DEST:-$SCRIPT_DIR/quick_prompt_test_output}"
        TARGET_COUNT=6
        ;;
    smoke)
        REMOTE_SRC="${REMOTE_DATA_DIR:-/data}/smoke_test_output/"
        LOCAL_DEST="${LOCAL_DEST:-$SCRIPT_DIR/smoke_test_output}"
        TARGET_COUNT=16
        ;;
    variation)
        REMOTE_SRC="${REMOTE_DATA_DIR:-/data}/variation_test_output/"
        LOCAL_DEST="${LOCAL_DEST:-$SCRIPT_DIR/variation_test_output}"
        TARGET_COUNT=64
        ;;
    full|*)
        REMOTE_SRC="${REMOTE_DATA_DIR:-/data}/qwen_dataset_output/"
        LOCAL_DEST="${LOCAL_DEST:-$REPO_ROOT/ML/data/vision_synthetic/raw_generated}"
        TARGET_COUNT=28000
        ;;
esac

# SSH connection options:
# 1. ControlMaster/ControlPath: multiplex connections so passphrase is only entered ONCE
# 2. ServerAliveInterval/CountMax: send heartbeats so connection is NEVER dropped by cloud/firewalls
SSH_MUX_PATH="/tmp/ssh_mux_%h_%p_%r"
SSH_CMD="ssh -o ControlMaster=auto -o ControlPath=$SSH_MUX_PATH -o ControlPersist=4h -o ServerAliveInterval=30 -o ServerAliveCountMax=5"
if [ -n "$REMOTE_PORT" ]; then
    SSH_CMD="$SSH_CMD -p $REMOTE_PORT"
fi

mkdir -p "$LOCAL_DEST"

# Detect remote task range and slice target for the specified host
REMOTE_RANGE=$($SSH_CMD "$REMOTE_HOST" "cat /data/LookMax_Generator/.auto_resume_config 2>/dev/null | grep TASK_RANGE | cut -d'=' -f2 | tr -d '\"'" 2>/dev/null || true)
if [ -n "$REMOTE_RANGE" ]; then
    START_P=$(echo "$REMOTE_RANGE" | cut -d':' -f1)
    END_P=$(echo "$REMOTE_RANGE" | cut -d':' -f2)
    HOST_TARGET=$((END_P - START_P))
    SLICE_LABEL="Slice [$REMOTE_RANGE]"
else
    HOST_TARGET="$TARGET_COUNT"
    SLICE_LABEL="Full Set"
fi

echo "================================================================================"
echo " LookMax Live Synthetic Output Streamer (Local Process)"
echo "================================================================================"
echo " Mode          : $MODE"
echo " Remote Host   : $REMOTE_HOST"
echo " Target Scope  : $HOST_TARGET images ($SLICE_LABEL)"
echo " Remote Source : $REMOTE_HOST:$REMOTE_SRC"
echo " Local Archive : $LOCAL_DEST"
echo " Sync Interval : Every ${INTERVAL_SEC}s"
echo " Safe Mode     : Excluding *.tmp (only fully finalized images are transferred)"
echo " Connection    : Multiplexed with 30s keep-alive heartbeat"
echo "================================================================================"
echo ""
echo "Streaming started. Press Ctrl+C anytime to pause (remote generation continues)."
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Query completed image count directly on the remote machine
    REMOTE_COUNT=$($SSH_CMD "$REMOTE_HOST" "find ${REMOTE_SRC}images -maxdepth 1 -name '*.png' ! -name '*.tmp' 2>/dev/null | wc -l" 2>/dev/null | tr -d ' ' || echo 0)

    # Run rsync quietly with multiplexed SSH, skipping temporary and hidden files
    rsync -avz --partial \
        --exclude="*.tmp" \
        --exclude=".*" \
        -e "$SSH_CMD" \
        "$REMOTE_HOST:$REMOTE_SRC" \
        "$LOCAL_DEST/" >/dev/null 2>&1 || true

    # Count total images accumulated in the local combined archive
    if [ -d "$LOCAL_DEST/images" ]; then
        LOCAL_TOTAL=$(find "$LOCAL_DEST/images" -maxdepth 1 -name "*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
    else
        LOCAL_TOTAL=$(find "$LOCAL_DEST" -maxdepth 1 -name "*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
    fi

    PERCENT=$(awk "BEGIN {printf \"%.2f\", ($REMOTE_COUNT / $HOST_TARGET) * 100}")
    TOTAL_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($LOCAL_TOTAL / $TARGET_COUNT) * 100}")

    echo "[$TIMESTAMP] [$REMOTE_HOST] Generated: $REMOTE_COUNT / $HOST_TARGET images ($PERCENT% of $SLICE_LABEL) | Total local archive: $LOCAL_TOTAL / $TARGET_COUNT ($TOTAL_PERCENT%). Next sync in ${INTERVAL_SEC}s..."

    # If in test/smoke mode and all images arrived, do one final sync and exit cleanly
    if [[ "$MODE" != "full" ]] && [ "$REMOTE_COUNT" -ge "$HOST_TARGET" ]; then
        echo "✓ All $HOST_TARGET $MODE images downloaded from $REMOTE_HOST to $LOCAL_DEST!"
        break
    fi

    sleep "$INTERVAL_SEC"
done
