"""The appliance layer must import without vLLM installed."""

import lazarus.appliance
import lazarus.appliance.healthcheck
import lazarus.appliance.launcher


def test_appliance_imports_without_vllm():
    assert lazarus.appliance.launcher.main is not None
    assert lazarus.appliance.healthcheck.main is not None
