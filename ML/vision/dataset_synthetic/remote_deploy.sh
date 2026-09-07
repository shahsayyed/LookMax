#!/bin/bash
# ==============================================================================
# remote_deploy.sh -- Push ONLY the required generator scripts (616 KB) to
# the remote GPU box, avoiding cloning the large LookMax repository.
#
# Usage:
#   ./remote_deploy.sh                       # Uses 'vast' SSH host alias
#   ./remote_deploy.sh root@<IP> <PORT>      # Explicit host and port
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="${REMOTE_DIR:-/data/LookMax_Generator}"

if [ -n "$1" ] && [ -n "$2" ]; then
    TARGET_HOST="$1"
    TARGET_PORT="$2"
    SSH_CMD="ssh -p $TARGET_PORT"
    RSYNC_RSH="ssh -p $TARGET_PORT"
    DEST="$TARGET_HOST:$REMOTE_DIR"
elif [ -n "$1" ]; then
    TARGET_HOST="$1"
    SSH_CMD="ssh"
    RSYNC_RSH="ssh"
    DEST="$TARGET_HOST:$REMOTE_DIR"
else
    # Default to REMOTE_HOST env var or 'vast' SSH alias configured in ~/.ssh/config
    TARGET_HOST="${REMOTE_HOST:-vast}"
    SSH_CMD="ssh"
    RSYNC_RSH="ssh"
    DEST="$TARGET_HOST:$REMOTE_DIR"
fi

echo "==> Deploying synthetic dataset generator to: $DEST"

# 1. Ensure remote target directory exists
$SSH_CMD "$TARGET_HOST" "mkdir -p $REMOTE_DIR"

# 2. Sync only the lightweight generator scripts (~616 KB)
rsync -avz --progress \
    --exclude="output" \
    --exclude="__pycache__" \
    --exclude=".DS_Store" \
    --exclude="*.tmp" \
    --exclude="*.pyc" \
    -e "$RSYNC_RSH" \
    "$SCRIPT_DIR/" \
    "$DEST/"

# 3. Ensure shell scripts are executable on remote
$SSH_CMD "$TARGET_HOST" "chmod +x $REMOTE_DIR/*.sh"

echo ""
echo "==> Deployment complete! Files are ready at: $REMOTE_DIR"
echo ""
echo "Next step: Connect to your remote machine and run:"
echo "    ssh $TARGET_HOST"
echo "    cd $REMOTE_DIR"
echo "    ./run.sh setup"
echo "    ./run.sh test"
