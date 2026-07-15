"""The appliance layer must import without vLLM installed."""

import lazarus.appliance
import lazarus.appliance.healthcheck
import lazarus.appliance.launcher
from lazarus.appliance.manifest import RUNTIME_VERSION


def test_appliance_imports_without_vllm():
    assert lazarus.appliance.launcher.main is not None
    assert lazarus.appliance.healthcheck.main is not None


def test_runtime_reports_release_version():
    assert RUNTIME_VERSION == "0.1.0-rc.2"
