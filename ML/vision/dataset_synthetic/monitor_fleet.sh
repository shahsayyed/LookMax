#!/bin/bash
# ==============================================================================
# monitor_fleet.sh -- LookMax Synthetic Fleet Monitor, Sync & Auto-Destroy
#
# Usage:
#   ./monitor_fleet.sh                     # One-shot status check
#   ./monitor_fleet.sh sync                # Run live 5-min pull+push sync (foreground, Ctrl-C to stop)
#   ./monitor_fleet.sh sync-once           # Pull from all machines + push markers, then exit
#   ./monitor_fleet.sh sync-log            # Tail the sync log live
#   ./monitor_fleet.sh log                 # Tail the hourly health monitor log
#   ./monitor_fleet.sh status              # Print Markdown status report
#   ./monitor_fleet.sh loop                # Run hourly health monitor loop (background)
#   ./monitor_fleet.sh set-api-key <KEY>   # Save your Vast.ai API key securely
#   ./monitor_fleet.sh test-key            # Test your Vast.ai API key & view active instances
#   ./monitor_fleet.sh destroy <ID>        # Manually destroy a Vast.ai instance
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYNC_LOG="$SCRIPT_DIR/fleet_sync.log"
HEALTH_LOG="$SCRIPT_DIR/fleet_monitor.log"

case "$1" in
    sync)
        echo "=============================================="
        echo " LookMax Fleet Sync — Pull & Push Markers"
        echo " Interval : 30 seconds"
        echo " Pull log : $SYNC_LOG"
        echo " Ctrl-C   : stop"
        echo "=============================================="
        echo ""
        python3 "$SCRIPT_DIR/fleet_monitor.py" --sync-loop
        ;;
    sync-once)
        echo "Running single sync cycle (pull + push markers)..."
        python3 "$SCRIPT_DIR/fleet_monitor.py" --sync-once
        ;;
    sync-log)
        echo "=== Fleet Sync Log (Ctrl-C to stop) ==="
        tail -n 30 -f "$SYNC_LOG"
        ;;
    log)
        echo "=== Fleet Health Log (Ctrl-C to stop) ==="
        tail -n 25 -f "$HEALTH_LOG"
        ;;
    local)
        python3 "$SCRIPT_DIR/fleet_monitor.py" --local
        ;;
    status)
        cat "$SCRIPT_DIR/fleet_status.md"
        ;;
    loop)
        echo "Starting hourly fleet health monitor & auto-destroy loop in background..."
        nohup python3 "$SCRIPT_DIR/fleet_monitor.py" --loop > /dev/null 2>&1 &
        echo "✓ Monitor started (PID: $!). Logging to $HEALTH_LOG."
        ;;
    set-api-key)
        if [ -z "$2" ]; then
            echo "Usage: ./monitor_fleet.sh set-api-key <VAST_API_KEY>"
            exit 1
        fi
        python3 "$SCRIPT_DIR/fleet_monitor.py" --set-api-key "$2"
        ;;
    test-key)
        python3 "$SCRIPT_DIR/fleet_monitor.py" --test-api-key
        ;;
    destroy)
        if [ -z "$2" ]; then
            echo "Usage: ./monitor_fleet.sh destroy <INSTANCE_ID>"
            exit 1
        fi
        python3 "$SCRIPT_DIR/fleet_monitor.py" --destroy-instance "$2"
        ;;
    *)
        python3 "$SCRIPT_DIR/fleet_monitor.py"
        ;;
esac

