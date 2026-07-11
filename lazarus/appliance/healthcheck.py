"""sovereign-runtime-healthcheck entrypoint (scaffold — real implementation in M3).

Contract (design.md §3.3, §24): `--live` probes GET /health/live and must not
depend on model readiness; Docker healthchecks use only this mode.
"""

import sys


def main() -> int:
    print("sovereign-runtime-healthcheck: not implemented yet (M3)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
