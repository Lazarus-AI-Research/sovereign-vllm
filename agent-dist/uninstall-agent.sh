#!/usr/bin/env bash
# Remove the Sovereign host inference agent LaunchAgent. Keeps ~/.sovereign
# (config, token, logs) unless --purge is passed.
set -euo pipefail

LABEL="com.lazarus.sovereign-runtime-agent"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$PLIST"
echo "removed: $LABEL"

if [[ "${1:-}" == "--purge" ]]; then
  rm -rf "${SOVEREIGN_AGENT_HOME:-$HOME/.sovereign}"
  echo "purged agent home"
fi
