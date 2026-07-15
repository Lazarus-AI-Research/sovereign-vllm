#!/usr/bin/env bash
set -euo pipefail

TOKEN_FILE="__TOKEN_FILE__"
AGENT_BIN="__AGENT_BIN__"
AGENT_CONFIG="__AGENT_CONFIG__"

if [[ ! -r "$TOKEN_FILE" ]]; then
  echo "Sovereign agent token file is missing or unreadable" >&2
  exit 1
fi
export SOVEREIGN_AGENT_TOKEN="$(<"$TOKEN_FILE")"
exec "$AGENT_BIN" --config "$AGENT_CONFIG"
