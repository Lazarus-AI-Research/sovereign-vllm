#!/usr/bin/env bash
# Install the pinned, self-contained Metal host agent under ~/.sovereign.
set -Eeuo pipefail

DIST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_HOME="${SOVEREIGN_AGENT_HOME:-$HOME/.sovereign}"
RUNTIME_HOME="$AGENT_HOME/runtime"
VENV="$RUNTIME_HOME/venv"
CONFIG="$AGENT_HOME/agent.yaml"
LOG_DIR="$AGENT_HOME/logs"
TOKEN_FILE="$AGENT_HOME/agent.token"
WRAPPER="$RUNTIME_HOME/bin/sovereign-agent-wrapper"
PLIST_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/com.lazarus.sovereign-runtime-agent.plist"
WRAPPER_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agent-wrapper.sh"
CONFIG_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/agent.yaml.example"
PLIST_DST="$HOME/Library/LaunchAgents/com.lazarus.sovereign-runtime-agent.plist"
LABEL="com.lazarus.sovereign-runtime-agent"

UV_BIN="$DIST_DIR/bin/uv"
PYTHON_BIN="$DIST_DIR/python/bin/python3"
LLAMA_BIN="$DIST_DIR/bin/llama-server"
for required in "$UV_BIN" "$PYTHON_BIN" "$LLAMA_BIN"; do
  [[ -x "$required" ]] || { echo "error: Metal distribution is missing $required" >&2; exit 1; }
done
LLAMA_LIBS=("$DIST_DIR"/bin/*.dylib)
[[ -e "${LLAMA_LIBS[0]}" ]] || {
  echo "error: Metal distribution is missing llama.cpp dynamic libraries" >&2
  exit 1
}
WHEEL="$(find "$DIST_DIR/wheels" -maxdepth 1 -name 'sovereign_runtime-*.whl' -print -quit)"
[[ -n "$WHEEL" ]] || { echo "error: Sovereign runtime wheel is missing" >&2; exit 1; }

umask 077
mkdir -p "$RUNTIME_HOME/bin" "$LOG_DIR" "$AGENT_HOME/models/metal" "$HOME/Library/LaunchAgents"
"$UV_BIN" venv --clear --python "$PYTHON_BIN" "$VENV"
"$UV_BIN" pip install --python "$VENV/bin/python" --no-index --find-links "$DIST_DIR/wheels" "$WHEEL"
install -m 755 "$LLAMA_BIN" "$RUNTIME_HOME/bin/llama-server"
cp -pPR "${LLAMA_LIBS[@]}" "$RUNTIME_HOME/bin/"

if [[ ! -f "$CONFIG" ]]; then
  sed "s|__AGENT_HOME__|$AGENT_HOME|g" "$CONFIG_SRC" > "$CONFIG"
fi
chmod 600 "$CONFIG"
if [[ ! -f "$TOKEN_FILE" ]]; then
  openssl rand -hex 32 > "$TOKEN_FILE"
fi
chmod 600 "$TOKEN_FILE"

AGENT_BIN="$VENV/bin/sovereign-runtime-agent"
sed -e "s|__TOKEN_FILE__|$TOKEN_FILE|g" \
    -e "s|__AGENT_BIN__|$AGENT_BIN|g" \
    -e "s|__AGENT_CONFIG__|$CONFIG|g" \
    "$WRAPPER_SRC" > "$WRAPPER"
chmod 700 "$WRAPPER"
sed -e "s|__AGENT_WRAPPER__|$WRAPPER|g" \
    -e "s|__LOG_DIR__|$LOG_DIR|g" \
    "$PLIST_SRC" > "$PLIST_DST"
chmod 600 "$PLIST_DST"
plutil -lint "$PLIST_DST" >/dev/null

if [[ "${SOVEREIGN_SKIP_AGENT_START:-0}" != 1 ]]; then
  launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
  # launchd can briefly retain a just-removed label. Retry the registration so
  # an in-place upgrade does not fail with Bootstrap error 5.
  for attempt in 1 2 3 4 5; do
    if launchctl bootstrap "gui/$(id -u)" "$PLIST_DST"; then
      break
    fi
    if [[ "$attempt" == 5 ]]; then
      echo "error: failed to register $LABEL with launchd" >&2
      exit 1
    fi
    sleep 1
  done
  launchctl kickstart -k "gui/$(id -u)/$LABEL"
fi

echo "installed: $LABEL"
echo "token file: $TOKEN_FILE"
echo "logs: $LOG_DIR/agent.log"
