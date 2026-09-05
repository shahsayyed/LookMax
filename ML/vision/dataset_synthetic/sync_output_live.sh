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

# Parse arguments
if [ -n "$1" ] && [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
    REMOTE_HOST="$1"
    REMOTE_PORT="$2"
    LOCAL_DEST="${3:-$REPO_ROOT/ML/data/vision_synthetic/raw_generated}"
    INTERVAL_SEC="${4:-30}"
    SSH_OPT="-e ssh -p $REMOTE_PORT"
elif [ -n "$1" ]; then
    REMOTE_HOST="$1"
    REMOTE_PORT="22"
    LOCAL_DEST="${2:-$REPO_ROOT/ML/data/vision_synthetic/raw_generated}"
    INTERVAL_SEC="${3:-30}"
    SSH_OPT="-e ssh"
else
    REMOTE_HOST="vast"
    REMOTE_PORT="22"
    LOCAL_DEST="$REPO_ROOT/ML/data/vision_synthetic/raw_generated"
    INTERVAL_SEC="30"
    SSH_OPT=""
fi

REMOTE_SRC="${REMOTE_DATA_DIR:-/data}/qwen_dataset_output/"
mkdir -p "$LOCAL_DEST"

echo "================================================================================"
echo " LookMax Live Synthetic Output Streamer (Local Process)"
echo "================================================================================"
echo " Remote Source : $REMOTE_HOST:$REMOTE_SRC"
echo " Local Dest    : $LOCAL_DEST"
echo " Sync Interval : Every ${INTERVAL_SEC}s"
echo " Safe Mode     : Excluding *.tmp (only fully finalized images are transferred)"
echo "================================================================================"
echo ""
echo "Streaming started. Press Ctrl+C anytime to pause (remote generation continues)."
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Run rsync quietly, skipping temporary files
    rsync -avz --partial \
        --exclude="*.tmp" \
        --exclude=".*" \
        $SSH_OPT \
        "$REMOTE_HOST:$REMOTE_SRC" \
        "$LOCAL_DEST/" >/dev/null 2>&1 || true

    # Count downloaded completed images
    if [ -d "$LOCAL_DEST/images" ]; then
        LOCAL_COUNT=$(find "$LOCAL_DEST/images" -maxdepth 1 -name "*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
    else
        LOCAL_COUNT=0
    fi

    PERCENT=$(awk "BEGIN {printf \"%.2f\", ($LOCAL_COUNT / 28000) * 100}")
    echo "[$TIMESTAMP] Synced: $LOCAL_COUNT / 28,000 images mirrored locally ($PERCENT%). Next sync in ${INTERVAL_SEC}s..."

    sleep "$INTERVAL_SEC"
done
