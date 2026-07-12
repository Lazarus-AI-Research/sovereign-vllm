#!/usr/bin/env bash
# Install the Sovereign host inference agent as a launchd LaunchAgent
# (Metal Phase 2, design.md §2.6). Idempotent; run as the login user.
#
# Prerequisites: the sovereign-runtime Python package installed on the host
# (provides sovereign-runtime-agent), llama.cpp's llama-server on PATH, and
# an agent config at ~/.sovereign/agent.yaml (see agent.yaml.example).
set -euo pipefail

AGENT_HOME="${SOVEREIGN_AGENT_HOME:-$HOME/.sovereign}"
CONFIG="$AGENT_HOME/agent.yaml"
LOG_DIR="$AGENT_HOME/logs"
PLIST_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/com.lazarus.sovereign-runtime-agent.plist"
PLIST_DST="$HOME/Library/LaunchAgents/com.lazarus.sovereign-runtime-agent.plist"
LABEL="com.lazarus.sovereign-runtime-agent"

AGENT_BIN="$(command -v sovereign-runtime-agent || true)"
if [[ -z "$AGENT_BIN" ]]; then
  echo "error: sovereign-runtime-agent not found on PATH (pip install sovereign-runtime)" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "error: agent config not found at $CONFIG (copy agent.yaml.example)" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# Token: generate once, persist with owner-only permissions, share with the
# deploy config so the runtime container can authenticate.
TOKEN_FILE="$AGENT_HOME/agent.token"
if [[ ! -f "$TOKEN_FILE" ]]; then
  umask 077
  openssl rand -base64 24 > "$TOKEN_FILE"
fi
TOKEN="$(cat "$TOKEN_FILE")"

sed -e "s|__AGENT_BIN__|$AGENT_BIN|" \
    -e "s|__AGENT_CONFIG__|$CONFIG|" \
    -e "s|__AGENT_TOKEN__|$TOKEN|" \
    -e "s|__LOG_DIR__|$LOG_DIR|" \
    "$PLIST_SRC" > "$PLIST_DST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"

echo "installed: $LABEL (logs: $LOG_DIR/agent.log)"
echo "Set SOVEREIGN_AGENT_TOKEN=$TOKEN in the deploy .env so the runtime container can reach the agent."
