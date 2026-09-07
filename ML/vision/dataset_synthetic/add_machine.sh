#!/bin/bash
# ==============================================================================
# add_machine.sh -- One-command setup for new remote GPU instances (Vast.ai, etc.)
#
# What it does automatically:
#   1. Parses IP and Port from whatever you paste (full ssh command or IP PORT).
#   2. Saves your SSH key passphrase to the macOS Keychain (prompts once, never again).
#   3. Updates ~/.ssh/config with the 'vast' alias, keep-alive, and auto-keychain.
#   4. Verifies the connection and immediately opens the remote terminal.
#
# Usage examples:
#   ./add_machine.sh ssh -p 20080 root@20.64.248.96 -L 8080:localhost:8080
#   ./add_machine.sh 20.64.248.96 20080
#   ./add_machine.sh root@20.64.248.96 20080
#   ./add_machine.sh                          # Will prompt you to paste
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_INPUT="$*"

if [ -z "$RAW_INPUT" ]; then
    echo "================================================================================"
    echo " LookMax Remote Machine Quick Configurator"
    echo "================================================================================"
    echo "Paste the SSH command or IP:PORT from your Vast.ai dashboard below:"
    read -r -p "> " RAW_INPUT
fi

if [ -z "$RAW_INPUT" ]; then
    echo "Error: No SSH command or host information provided."
    exit 1
fi

# 1. Parse IP and Port from various formats
# Example: ssh -p 20080 root@20.64.248.96 -L 8080:localhost:8080
# Example: root@20.64.248.96 20080
# Example: 20.64.248.96:20080
REMOTE_USER="root"
REMOTE_IP=""
REMOTE_PORT="22"

# Extract Port: look for -p <port>, or :<port>, or space followed by port number
if [[ "$RAW_INPUT" =~ -p[[:space:]]+([0-9]+) ]]; then
    REMOTE_PORT="${BASH_REMATCH[1]}"
elif [[ "$RAW_INPUT" =~ :([0-9]{2,5}) ]]; then
    REMOTE_PORT="${BASH_REMATCH[1]}"
elif [[ "$RAW_INPUT" =~ [[:space:]]+([0-9]{2,5})($|[[:space:]]) ]]; then
    REMOTE_PORT="${BASH_REMATCH[1]}"
fi

# Extract User: username before @ (default: root)
if [[ "$RAW_INPUT" =~ ([a-zA-Z0-9_-]+)@ ]]; then
    REMOTE_USER="${BASH_REMATCH[1]}"
fi

# Extract IP address
if [[ "$RAW_INPUT" =~ ([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+) ]]; then
    REMOTE_IP="${BASH_REMATCH[1]}"
fi

# Extract Host Alias if specified (e.g. vast2, vast_h100, etc., default: vast)
HOST_ALIAS="vast"
if [[ "$RAW_INPUT" =~ --alias[[:space:]]+([a-zA-Z0-9_-]+) ]]; then
    HOST_ALIAS="${BASH_REMATCH[1]}"
elif [[ "$RAW_INPUT" =~ [[:space:]]+(vast[0-9]*)\b ]]; then
    HOST_ALIAS="${BASH_REMATCH[1]}"
fi

if [ -z "$REMOTE_IP" ]; then
    echo "Error: Could not extract a valid IP address from: '$RAW_INPUT'"
    echo "Please provide IP and PORT, e.g.: ./add_machine.sh 20.64.248.96 20080"
    exit 1
fi

echo ""
echo "==> Configuring remote host:"
echo "    Host Alias: $HOST_ALIAS"
echo "    IP        : $REMOTE_IP"
echo "    Port      : $REMOTE_PORT"
echo "    User      : $REMOTE_USER"

# 2. Locate SSH Identity Key
KEY_FILE="$HOME/.ssh/id_ed25519"
if [ ! -f "$KEY_FILE" ]; then
    if [ -f "$HOME/.ssh/id_rsa" ]; then
        KEY_FILE="$HOME/.ssh/id_rsa"
    else
        echo "Warning: Neither ~/.ssh/id_ed25519 nor ~/.ssh/id_rsa was found."
        KEY_FILE="$HOME/.ssh/id_ed25519"
    fi
fi

# 3. macOS Keychain integration: ensure passphrase is saved
echo ""
echo "==> Checking macOS Keychain for $KEY_FILE..."
if ! ssh-add -l 2>/dev/null | grep -q -E "(ED25519|RSA)"; then
    echo "Adding SSH key to macOS Keychain (enter passphrase if prompted)..."
    ssh-add --apple-use-keychain "$KEY_FILE" 2>/dev/null || ssh-add "$KEY_FILE" 2>/dev/null || true
else
    echo "✓ SSH key is already unlocked in SSH agent."
fi

# 4. Update ~/.ssh/config in-place (replaces existing 'Host vast' entry cleanly without duplicates)
SSH_CONFIG="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
touch "$SSH_CONFIG"

PYTHON_BIN="/usr/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -c '
import sys, re

config_path = sys.argv[1]
ip = sys.argv[2]
port = sys.argv[3]
user = sys.argv[4]
key = sys.argv[5]
alias = sys.argv[6]

new_entry = (
    f"Host {alias}\n"
    f"    HostName {ip}\n"
    f"    Port {port}\n"
    f"    User {user}\n"
    f"    IdentityFile {key}\n"
    f"    StrictHostKeyChecking no\n"
    f"    UserKnownHostsFile /dev/null\n"
    f"    AddKeysToAgent yes\n"
    f"    UseKeychain yes\n"
    f"    ServerAliveInterval 30\n"
    f"    ServerAliveCountMax 5\n\n"
)

with open(config_path, "r") as f:
    content = f.read()

pattern = rf"(?m)^Host\s+{re.escape(alias)}\b(?:\n[ \t]+[^\n]*|\n(?![ \t]*Host\b)[^\n]*)*\n*"

if re.search(pattern, content):
    first = [True]
    def repl(match):
        if first[0]:
            first[0] = False
            return new_entry
        return ""
    updated = re.sub(pattern, repl, content)
else:
    trimmed = content.rstrip()
    if trimmed:
        updated = trimmed + "\n\n" + new_entry
    else:
        updated = new_entry

normalized = re.sub(r"\n{3,}", "\n\n", updated.strip()) + "\n"

with open(config_path, "w") as f:
    f.write(normalized)
' "$SSH_CONFIG" "$REMOTE_IP" "$REMOTE_PORT" "$REMOTE_USER" "$KEY_FILE" "$HOST_ALIAS"

chmod 600 "$SSH_CONFIG"
echo "✓ ~/.ssh/config updated in-place (Host '$HOST_ALIAS' configured cleanly)!"

# 5. Test connection
echo ""
echo "==> Testing SSH connection to '$HOST_ALIAS' ($REMOTE_IP:$REMOTE_PORT)..."
if ssh -o ConnectTimeout=8 -o BatchMode=yes "$HOST_ALIAS" "echo '✓ Connection handshake OK on \$(hostname)'"; then
    echo "✓ Authentication succeeded with zero password prompt!"
else
    echo "Testing connection with interactive terminal..."
    ssh -o ConnectTimeout=10 "$HOST_ALIAS" "echo '✓ Connection successful!'"
fi

# 6. Offer automatic deployment
echo ""
read -r -p "Deploy generator files (~616 KB) to /data/LookMax_Generator on $HOST_ALIAS now? [Y/n] " DEPLOY_CONFIRM
DEPLOY_CONFIRM=${DEPLOY_CONFIRM:-Y}
if [[ "$DEPLOY_CONFIRM" =~ ^[Yy]$ ]]; then
    echo "==> Deploying generator scripts..."
    REMOTE_HOST="$HOST_ALIAS" "$SCRIPT_DIR/remote_deploy.sh"
fi

# 7. Connect into the machine
echo ""
echo "================================================================================"
echo " Machine setup is 100% complete! Logging you in now..."
echo " From now on, you can simply type:  ssh $HOST_ALIAS"
echo "================================================================================"
echo ""
exec ssh "$HOST_ALIAS"
