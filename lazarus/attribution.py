"""The fixed "Powered by Lazarus AI" attribution for the Sovereign Runtime.

A required branding position (see BRANDING.md in the sovereign-stack repo). These
are module constants, not configuration, so no runtime.yaml, environment, or
customer-editable value can change them. The runtime is a non-UI service, so it
prints the banner up front, at startup, before anything else can fail.

Apache-2.0 means a fork can strip this; the point is that it is inconvenient and
conspicuous, not a toggle. ``tests/test_attribution.py`` pins the exact values,
so a silent edit fails CI. Keep these identical to the control module's
attribution package and to BRANDING.md.
"""

URL = "https://github.com/Lazarus-AI-Research/sovereign-stack"
NOTICE = "Powered by Lazarus AI"


def banner() -> str:
    """The one-line startup banner the service prints up front."""
    return f"{NOTICE} — {URL}"
