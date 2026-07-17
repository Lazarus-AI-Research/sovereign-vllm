"""Enforcement: the runtime's attribution must be present and unchanged.

A silent removal or edit fails here. A required branding position; see
BRANDING.md in the sovereign-stack repo.
"""

from lazarus import attribution


def test_notice_is_exact():
    assert attribution.NOTICE == "Powered by Lazarus AI"


def test_url_points_at_the_project():
    assert attribution.URL == "https://github.com/Lazarus-AI-Research/sovereign-stack"


def test_banner_carries_both():
    b = attribution.banner()
    assert attribution.NOTICE in b
    assert attribution.URL in b


def test_launcher_prints_the_banner():
    """The banner must actually be emitted at startup, not merely defined."""
    import inspect

    from lazarus.appliance import launcher

    src = inspect.getsource(launcher.main)
    assert "banner()" in src, "launcher.main must print the attribution banner up front"
