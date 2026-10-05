#!/usr/bin/env bash
# Build SlimServe for Metal at its pinned commit into a wheelhouse the Metal
# distribution carries: SlimServe's own wheel and every wheel it installs
# with, each pinned by hash in requirements.lock, so the agent's installer
# needs no network. Runs on Apple Silicon with Xcode's Metal toolchain.
#
#   agent-dist/slimserve/build-slimserve.sh <python3.12> <wheelhouse>
set -Eeuo pipefail

SLIMSERVE_REPOSITORY="https://github.com/QuixiAI/SlimServe.git"
SLIMSERVE_COMMIT="6aa3c39565ab582b4aad89cf36fcd2c194363024"
# SlimServe compiles its kernels for Metal 4, which only macOS 26 loads, for
# the tensor kernels of an M5. patches/ builds them for Metal 3.2, which
# macOS 15 loads, and serves with the simdgroup kernels the tensor kernels
# replaced. Carried until SlimServe serves without tensor kernels itself.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <python3.12> <wheelhouse>" >&2
  exit 2
fi
PYTHON="$1"
WHEELHOUSE="$(mkdir -p "$2" && cd "$2" && pwd)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$(uname -s)/$(uname -m)" == Darwin/arm64 ]] || { echo "error: SlimServe for Metal builds on Apple Silicon" >&2; exit 1; }
xcrun --find metal >/dev/null || { echo "error: the Metal toolchain is missing (install Xcode)" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
git init -q "$WORK/source"
git -C "$WORK/source" fetch -q --depth 1 "$SLIMSERVE_REPOSITORY" "$SLIMSERVE_COMMIT"
git -C "$WORK/source" checkout -q FETCH_HEAD
[[ "$(git -C "$WORK/source" rev-parse HEAD)" == "$SLIMSERVE_COMMIT" ]]

for patch in "$HERE"/patches/*.patch; do
  git -C "$WORK/source" apply "$patch"
done

"$PYTHON" -m venv "$WORK/build"
"$WORK/build/bin/python" -m pip install -q --require-hashes --only-binary=:all: --requirement "$HERE/build-requirements.lock"
(
  cd "$WORK/source"
  VLLM_TARGET_DEVICE=metal MAX_JOBS="${MAX_JOBS:-8}" \
    "$WORK/build/bin/python" -m pip wheel -q --no-build-isolation --no-deps --wheel-dir "$WHEELHOUSE" .
)
"$PYTHON" -m pip download -q --require-hashes --only-binary=:all: --dest "$WHEELHOUSE" --requirement "$HERE/requirements.lock"
printf '%s\n' "$SLIMSERVE_COMMIT" > "$WHEELHOUSE/SLIMSERVE_COMMIT"
echo "SlimServe $SLIMSERVE_COMMIT built into $WHEELHOUSE"
