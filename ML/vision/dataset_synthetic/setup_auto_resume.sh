#!/bin/bash
# ==============================================================================
# setup_auto_resume.sh -- Installs boot/resume hooks on Vast.ai spot instance
# Usage:
#   ./setup_auto_resume.sh [TASK_RANGE] [WORKERS]
# Example:
#   ./setup_auto_resume.sh 16000:28000 2
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_RANGE="${1:-16000:28000}"
WORKERS="${2:-2}"

echo "==> Configuring Auto-Resume on this instance (Range: $TASK_RANGE, Workers: $WORKERS)..."

# Save configured settings to .auto_resume_config
cat << EOF > "$SCRIPT_DIR/.auto_resume_config"
TASK_RANGE="$TASK_RANGE"
WORKERS="$WORKERS"
LOOKMAX_DATA_DIR="/data"
HF_TOKEN="${HF_TOKEN:-}"
EOF

# 1. Hook into /root/onstart.sh (executed by Vast.ai on container start / resume)
ONSTART_SCRIPT="/root/onstart.sh"
echo "==> Updating $ONSTART_SCRIPT..."
mkdir -p /root 2>/dev/null || true
cat << 'INNER_EOF' > "$ONSTART_SCRIPT"
#!/bin/bash
if [ -f /data/LookMax_Generator/.auto_resume_config ]; then
    source /data/LookMax_Generator/.auto_resume_config
fi
cd /data/LookMax_Generator && bash auto_resume.sh "$TASK_RANGE" "$WORKERS" >> /data/auto_resume_boot.log 2>&1 &
INNER_EOF
chmod +x "$ONSTART_SCRIPT" 2>/dev/null || true

# 2. Add 5-minute cron watchdog so if preemption happens without onstart trigger, it auto-resumes
CRON_FILE="/etc/cron.d/lookmax_watchdog"
if [ -d "/etc/cron.d" ]; then
    echo "==> Installing 5-minute watchdog cronjob at $CRON_FILE..."
    cat << EOF > "$CRON_FILE"
SHELL=/bin/bash
PATH=/venv/main/bin:/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
*/5 * * * * root cd $SCRIPT_DIR && bash auto_resume.sh "$TASK_RANGE" "$WORKERS" >> /data/auto_resume_cron.log 2>&1
EOF
    chmod 0644 "$CRON_FILE" 2>/dev/null || true
fi

echo ""
echo "✓ Auto-resume hooks installed successfully!"
echo "  - Config file    : $SCRIPT_DIR/.auto_resume_config"
echo "  - Onstart trigger: $ONSTART_SCRIPT"
echo "  - Cron watchdog  : $CRON_FILE (checks every 5 minutes)"
echo ""
echo "To trigger manually at any time:"
echo "    ./run.sh resume"
