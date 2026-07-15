"""sovereign-runtime-healthcheck (design.md §3.3, §24).

``--live`` succeeds whenever the supervising API is alive and is the only
mode suitable for Docker health checks. ``--ready`` succeeds only after every
required role has loaded and passed its startup smoke test. The binary stays
stdlib-only so it can diagnose a partially initialized runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="probe liveness (default)")
    mode.add_argument("--ready", action="store_true", help="probe model readiness")
    parser.add_argument("--port", type=int, default=int(os.environ.get("SOVEREIGN_RUNTIME_PORT", "8000")))
    args = parser.parse_args()

    path = "/health/ready" if args.ready else "/health/live"
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}{path}", timeout=5) as resp:
            body = json.loads(resp.read())
            if resp.status != 200:
                return 1
            return 0 if body.get("ready") is True else int(body.get("status") != "alive")
    except Exception:
        return 1


if __name__ == "__main__":
    sys.exit(main())
