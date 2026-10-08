import asyncio

import pytest

from lazarus.agent.config import AgentConfig
from lazarus.agent.server import Agent


# llama-server names its build both ways: before llama.cpp 0.6 and since.
@pytest.mark.parametrize("line, version", [
    ("version: 9960 (a935fbffe)", "b9960-a935fbffe"),
    ("version: 0.6.0-dev (build 11459, commit f498f864f)", "b11459-f498f864f"),
])
def test_the_installed_llama_server_is_listed_at_its_build(tmp_path, line, version):
    binary = tmp_path / "llama-server"
    binary.write_text(f"#!/bin/sh\necho '{line}'\necho 'built with AppleClang for Darwin arm64'\n")
    binary.chmod(0o755)
    agent = Agent(AgentConfig(llama_server=str(binary)), tmp_path / "agent.yaml")
    asyncio.run(agent.discover_engines())
    assert [engine["version"] for engine in agent.available_engines if engine["name"] == "llama.cpp"] == [version]


# mlx-lm is listed where the agent's own Python carries it, and not where it
# does not (an agent built before it, or off Apple Silicon).
def test_mlx_lm_is_listed_where_the_agents_python_carries_it(tmp_path, monkeypatch):
    from importlib import metadata, util

    agent = Agent(AgentConfig(llama_server=str(tmp_path / "missing" / "llama-server")), tmp_path / "agent.yaml")
    monkeypatch.setattr(util, "find_spec", lambda name: object() if name == "mlx_lm" else None)
    monkeypatch.setattr(metadata, "version", lambda name: "0.32.0" if name == "mlx-lm" else "0")
    asyncio.run(agent.discover_engines())
    assert [engine["version"] for engine in agent.available_engines if engine["name"] == "mlx-lm"] == ["0.32.0"]
    monkeypatch.setattr(util, "find_spec", lambda name: None)
    asyncio.run(agent.discover_engines())
    assert not [engine for engine in agent.available_engines if engine["name"] == "mlx-lm"]
