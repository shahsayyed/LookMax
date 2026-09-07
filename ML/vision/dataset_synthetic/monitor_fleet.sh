#!/bin/bash
# ==============================================================================
# monitor_fleet.sh -- LookMax Synthetic Fleet Monitor & Auto-Destroy Controller
#
# Usage:
#   ./monitor_fleet.sh                     # Run an instantaneous check and display status
#   ./monitor_fleet.sh set-api-key <KEY>   # Save your Vast.ai API key securely
#   ./monitor_fleet.sh test-key            # Test your Vast.ai API key & view active instances
#   ./monitor_fleet.sh loop                # Run hourly background monitor loop
#   ./monitor_fleet.sh log                 # Tail the hourly monitor log
#   ./monitor_fleet.sh status              # Print Markdown status report
#   ./monitor_fleet.sh destroy <ID>        # Manually destroy a Vast.ai instance
# ==============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$1" in
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
    loop)
        echo "Starting hourly fleet monitor & auto-destroy loop in background..."
        nohup python3 "$SCRIPT_DIR/fleet_monitor.py" --loop > /dev/null 2>&1 &
        echo "✓ Monitor started (PID: $!). Logging to $SCRIPT_DIR/fleet_monitor.log."
        ;;
    log)
        tail -n 25 -f "$SCRIPT_DIR/fleet_monitor.log"
        ;;
    status)
        cat "$SCRIPT_DIR/fleet_status.md"
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
