"""sovereign-runtime-healthcheck (design.md §3.3, §24).

--live probes only GET /health/live: success whenever the supervising process
and control API are alive, independent of model readiness. Docker
healthchecks must use only this mode. stdlib-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="probe liveness only (default)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SOVEREIGN_RUNTIME_PORT", "8000")))
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health/live", timeout=5) as resp:
            body = json.loads(resp.read())
            return 0 if resp.status == 200 and body.get("status") == "alive" else 1
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
