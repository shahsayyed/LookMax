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
SSH_PROBE_CMD="ssh -q -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no"
if [ -n "$REMOTE_PORT" ]; then
    SSH_CMD="$SSH_CMD -p $REMOTE_PORT"
    SSH_PROBE_CMD="$SSH_PROBE_CMD -p $REMOTE_PORT"
fi

mkdir -p "$LOCAL_DEST"

# Invalidate and clean up stale multiplex control sockets if connection died
cleanup_stale_mux() {
    rm -f /tmp/ssh_mux_${REMOTE_HOST}_* /tmp/ssh_mux_*_${REMOTE_PORT:-22}_* 2>/dev/null || true
}

# Fast, non-interactive reachability check.
# BatchMode=yes ensures ssh never prompts for password or retries bad credentials (prevents IP bans/fail2ban)
is_host_reachable() {
    $SSH_PROBE_CMD "$REMOTE_HOST" "exit 0" >/dev/null 2>&1
    return $?
}

# Detect remote task range and slice target for the specified host
detect_remote_range() {
    local detected_range
    detected_range=$($SSH_CMD "$REMOTE_HOST" "cat /data/LookMax_Generator/.auto_resume_config 2>/dev/null | grep TASK_RANGE | cut -d'=' -f2 | tr -d '\"'" 2>/dev/null || true)
    if [ -n "$detected_range" ]; then
        START_P=$(echo "$detected_range" | cut -d':' -f1)
        END_P=$(echo "$detected_range" | cut -d':' -f2)
        HOST_TARGET=$((END_P - START_P))
        SLICE_LABEL="Slice [$detected_range]"
    else
        HOST_TARGET="$TARGET_COUNT"
        SLICE_LABEL="Full Set"
    fi
}

if is_host_reachable; then
    detect_remote_range
else
    HOST_TARGET="$TARGET_COUNT"
    SLICE_LABEL="Full Set (host currently offline)"
fi

echo "================================================================================"
echo " LookMax Live Synthetic Output Streamer (Local Process)"
echo "================================================================================"
echo " Mode          : $MODE"
echo " Remote Host   : $REMOTE_HOST"
echo " Target Scope  : $HOST_TARGET images ($SLICE_LABEL)"
echo " Remote Source : $REMOTE_HOST:$REMOTE_SRC"
echo " Local Archive : $LOCAL_DEST"
echo " Sync Interval : Every ${INTERVAL_SEC}s (when online)"
echo " Resilience    : Exponential backoff on interruption (prevents IP ban/rate-limit)"
echo " Safe Mode     : Excluding *.tmp (only fully finalized images are transferred)"
echo " Connection    : Multiplexed with 30s keep-alive heartbeat"
echo "================================================================================"
echo ""
echo "Streaming started. Press Ctrl+C anytime to pause (remote generation continues)."
echo ""

# Exponential backoff parameters for handling interruptions without hammering the host
FAIL_COUNT=0
BASE_BACKOFF=15
MAX_BACKOFF=300 # Cap probe interval at 5 minutes
WAS_OFFLINE=0

# Disable set -e for the main loop so intermittent network drops do not terminate the script
set +e

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Step 1: Fast non-interactive reachability check before executing any heavy SSH or rsync
    if ! is_host_reachable; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        WAS_OFFLINE=1
        cleanup_stale_mux

        # Calculate exponential backoff with jitter: 15s -> 30s -> 60s -> 120s -> max 300s
        EXP=$((FAIL_COUNT - 1))
        [ $EXP -gt 5 ] && EXP=5
        BACKOFF=$((BASE_BACKOFF * (1 << EXP)))
        [ $BACKOFF -gt $MAX_BACKOFF ] && BACKOFF=$MAX_BACKOFF
        JITTER=$((RANDOM % 7 - 3)) # +/- 3 seconds jitter to avoid fixed cadence
        SLEEP_SEC=$((BACKOFF + JITTER))
        [ $SLEEP_SEC -lt 5 ] && SLEEP_SEC=5

        echo "[$TIMESTAMP] [$REMOTE_HOST] ⚠️ Host is offline/interrupted (check #$FAIL_COUNT). Sync paused to prevent IP block. Next retry in ${SLEEP_SEC}s..."
        sleep "$SLEEP_SEC"
        continue
    fi

    # Step 2: If recovering from offline state, announce recovery and refresh remote config
    if [ "$WAS_OFFLINE" -eq 1 ]; then
        echo "[$TIMESTAMP] [$REMOTE_HOST] 🟢 Host is back ONLINE! Resuming active live sync."
        WAS_OFFLINE=0
        FAIL_COUNT=0
        cleanup_stale_mux
        detect_remote_range
        echo "[$TIMESTAMP] [$REMOTE_HOST] Synchronized target scope: $HOST_TARGET images ($SLICE_LABEL)"
    else
        FAIL_COUNT=0
    fi

    # Step 3: Query completed image count directly on the remote machine (safe parsing, ignoring 0-byte files)
    REMOTE_RAW=$($SSH_CMD "$REMOTE_HOST" "find ${REMOTE_SRC}images -maxdepth 1 -name '*.png' ! -name '*.tmp' ! -size 0 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    REMOTE_COUNT=$(echo "$REMOTE_RAW" | tr -cd '0-9')
    REMOTE_COUNT="${REMOTE_COUNT:-0}"

    # Step 4: Run rsync quietly with multiplexed SSH, skipping temporary, hidden, and 0-byte marker files
    if ! rsync -avz --partial \
        --min-size=1 \
        --exclude="*.tmp" \
        --exclude=".*" \
        -e "$SSH_CMD" \
        "$REMOTE_HOST:$REMOTE_SRC" \
        "$LOCAL_DEST/" >/dev/null 2>&1; then
        # If rsync dropped mid-transfer, invalidate stale multiplexer for clean retry
        cleanup_stale_mux
    fi

    # Step 5: Count total images accumulated in the local combined archive (ignoring 0-byte files)
    if [ -d "$LOCAL_DEST/images" ]; then
        LOCAL_TOTAL=$(find "$LOCAL_DEST/images" -maxdepth 1 -name "*.png" ! -name "*.tmp" ! -size 0 2>/dev/null | wc -l | tr -cd '0-9')
    else
        LOCAL_TOTAL=$(find "$LOCAL_DEST" -maxdepth 1 -name "*.png" ! -name "*.tmp" ! -size 0 2>/dev/null | wc -l | tr -cd '0-9')
    fi
    LOCAL_TOTAL="${LOCAL_TOTAL:-0}"

    [ -z "$HOST_TARGET" ] || [ "$HOST_TARGET" -le 0 ] && HOST_TARGET="$TARGET_COUNT"

    PERCENT=$(awk -v c="$REMOTE_COUNT" -v t="$HOST_TARGET" 'BEGIN {if (t > 0) printf "%.2f", (c / t) * 100; else printf "0.00"}')
    TOTAL_PERCENT=$(awk -v c="$LOCAL_TOTAL" -v t="$TARGET_COUNT" 'BEGIN {if (t > 0) printf "%.2f", (c / t) * 100; else printf "0.00"}')

    echo "[$TIMESTAMP] [$REMOTE_HOST] Generated: $REMOTE_COUNT / $HOST_TARGET images ($PERCENT% of $SLICE_LABEL) | Total local archive: $LOCAL_TOTAL / $TARGET_COUNT ($TOTAL_PERCENT%). Next sync in ${INTERVAL_SEC}s..."

    # If in test/smoke mode and all images arrived, do one final sync and exit cleanly
    if [[ "$MODE" != "full" ]] && [ "$REMOTE_COUNT" -ge "$HOST_TARGET" ]; then
        echo "✓ All $HOST_TARGET $MODE images downloaded from $REMOTE_HOST to $LOCAL_DEST!"
        break
    fi

    sleep "$INTERVAL_SEC"
done
