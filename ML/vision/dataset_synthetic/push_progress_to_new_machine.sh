#!/bin/bash
# ==============================================================================
# push_progress_to_new_machine.sh -- Disaster Recovery / Machine Migration
#
# If your remote GPU instance dies, stops, or gets preempted, use this script
# to restore your generation state onto a NEW machine in ~5 seconds.
#
# HOW IT WORKS:
#   1. Syncs the current CSV label files and generation logs to the new machine.
#   2. Creates lightweight 0-byte markers for all already-downloaded images in
#      /data/qwen_dataset_output/images/ on the new machine.
#   3. The new machine instantly detects all completed indices and continues
#      generating ONLY the remaining images.
#   4. Zero gigabytes of image data need to be re-uploaded from your Mac!
#
# Usage:
#   ./push_progress_to_new_machine.sh                       # Uses 'vast' SSH alias
#   ./push_progress_to_new_machine.sh root@<NEW_IP> <PORT>  # Explicit host & port
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOCAL_DATA_DIR="$REPO_ROOT/ML/data/vision_synthetic/raw_generated"
REMOTE_OUT_DIR="${REMOTE_DATA_DIR:-/data}/qwen_dataset_output"

if [ -n "$1" ] && [ -n "$2" ] && [[ "$2" =~ ^[0-9]+$ ]]; then
    TARGET_HOST="$1"
    TARGET_PORT="$2"
    SSH_CMD="ssh -p $TARGET_PORT"
    RSYNC_RSH="ssh -p $TARGET_PORT"
elif [ -n "$1" ]; then
    TARGET_HOST="$1"
    SSH_CMD="ssh"
    RSYNC_RSH="ssh"
else
    TARGET_HOST="vast"
    SSH_CMD="ssh"
    RSYNC_RSH="ssh"
fi

if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "!! Local data directory not found: $LOCAL_DATA_DIR"
    echo "   Ensure you have been running sync_output_live.sh locally."
    exit 1
fi

LOCAL_IMAGE_COUNT=$(find "$LOCAL_DATA_DIR/images" -maxdepth 1 -name "*.png" ! -name "*.tmp" 2>/dev/null | wc -l | tr -d ' ')
echo "================================================================================"
echo " LookMax State Restoration onto New Machine"
echo "================================================================================"
echo " Target Machine : $TARGET_HOST ($SSH_CMD)"
echo " Local Images   : $LOCAL_IMAGE_COUNT completed images found"
echo " Remote Target  : $REMOTE_OUT_DIR"
echo "================================================================================"

# 1. Ensure remote directories exist
$SSH_CMD "$TARGET_HOST" "mkdir -p '$REMOTE_OUT_DIR/images'"

# 2. Sync label CSVs, schema JSONs, and generation logs (a few KB)
echo "==> Syncing label CSVs and metadata..."
rsync -avz \
    --include="*.csv" \
    --include="*.json" \
    --include="*.jsonl" \
    --exclude="images" \
    --exclude=".*" \
    -e "$RSYNC_RSH" \
    "$LOCAL_DATA_DIR/" \
    "$TARGET_HOST:$REMOTE_OUT_DIR/"

# 3. Create lightweight completion markers on the new machine
if [ "$LOCAL_IMAGE_COUNT" -gt 0 ]; then
    echo "==> Creating $LOCAL_IMAGE_COUNT completion markers on new machine (fast, ~2 seconds)..."
    # Extract just the filenames and stream them over SSH to 'touch' on the remote side
    (cd "$LOCAL_DATA_DIR/images" && ls -1 *.png 2>/dev/null) | \
        $SSH_CMD "$TARGET_HOST" "cd '$REMOTE_OUT_DIR/images' && xargs touch"
fi

echo ""
echo "✓ State restored successfully!"
echo "   New machine now knows about all $LOCAL_IMAGE_COUNT completed images."
echo "   When you run './run.sh start' on the new machine, it will immediately"
echo "   skip those $LOCAL_IMAGE_COUNT images and generate only the remaining tasks."
