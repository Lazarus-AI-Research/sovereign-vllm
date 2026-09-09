#!/bin/bash
# Native build only: Docker cannot expose the Apple GPU or the Xcode Metal SDK.
# Does not modify the existing host-agent installation or invoke upstream setup.
set -euo pipefail
export PATH="/opt/homebrew/bin:${PATH}"

if [[ $# -ne 1 ]]; then
    printf 'Usage: bash docker/metal/build-slimserve.sh /absolute/path/to/locked-wheelhouse\n' >&2
    exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runtime_root="$(cd -- "${script_dir}/../.." && pwd)"
exec "${SLIMSERVE_PYTHON:-python3}" "${script_dir}/../slimserve/build.py" \
    --target metal --runtime-source "${runtime_root}" --dependencies "$1" \
    --max-jobs "${MAX_JOBS:-4}"
