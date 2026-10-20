"""The helper that builds Strata's pack runs in Strata's own Python; what it
decides from the files and the machine is checked here without it."""

import pytest

from lazarus.agent import strata_prepare


def test_the_two_shards_of_one_size_are_found_among_the_files():
    files = {
        "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf": "b",
        "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf": "a",
        "draft/mtp.fc_embedding.weight.bin": "c",
    }
    size, first, second = strata_prepare.shards(files)
    assert size == "Q2_0"
    assert first.name.endswith("-00001-of-00002.gguf") and second.name.endswith("-00002-of-00002.gguf")


@pytest.mark.parametrize("files", [
    {"Q2_0/Flash-GSQ-RCO-Q2_0-00001-of-00002.gguf": "a"},
    {"Flash-GSQ-RCO-Q2_0-00001-of-00002.gguf": "a", "Flash-GSQ-RCO-IQ3_S-00002-of-00002.gguf": "b"},
])
def test_files_without_one_sizes_two_shards_are_refused(files):
    with pytest.raises(SystemExit):
        strata_prepare.shards(files)


@pytest.mark.parametrize(("release", "metal_4"), [("26.1", True), ("27.0.1", True), ("15.7.9", False), ("", False)])
def test_prompts_take_the_decode_window_before_macos_26(monkeypatch, release, metal_4):
    monkeypatch.setattr(strata_prepare.platform, "mac_ver", lambda: (release, ("", "", ""), ""))
    assert strata_prepare.metal_4() is metal_4


def test_a_build_out_of_time_ends_every_tool_it_started(tmp_path, monkeypatch):
    import os
    import time
    from types import SimpleNamespace

    from lazarus.agent import strata

    python = tmp_path / "strata" / "python" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    started = tmp_path / "tool.pid"
    python.write_text(f"#!/bin/sh\nsleep 60 &\necho $! > {started}\nsleep 60\n")
    python.chmod(0o755)
    agent = SimpleNamespace(config=SimpleNamespace(strata=str(tmp_path / "strata")), config_path=tmp_path / "agent.yaml")
    monkeypatch.setattr(strata, "PREPARE_TIMEOUT", 1.0)
    monkeypatch.setattr(strata, "unified_memory_gb", lambda: 128.0)
    with pytest.raises(ValueError, match="not built within"):
        strata.prepare_strata(agent, "flash", {"sha256": "a" * 64})
    tool = int(started.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(tool, 0)
