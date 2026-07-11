"""run-sovereign-runtime entrypoint (scaffold — real implementation in M3)."""

import sys


def main() -> int:
    print("run-sovereign-runtime: appliance layer not implemented yet (M3)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
