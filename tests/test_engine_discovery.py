import asyncio

import pytest

from lazarus.agent.config import AgentConfig
from lazarus.agent.server import Agent


# llama-server names its build both ways: before llama.cpp 0.6 and since.
@pytest.mark.parametrize("line, version", [
    ("version: 9960 (a935fbffe)", "b9960-a935fbffe"),
    ("version: 0.6.0-dev (build 11457, commit 5ad1c5da0)", "b11457-5ad1c5da0"),
])
def test_the_installed_llama_server_is_listed_at_its_build(tmp_path, line, version):
    binary = tmp_path / "llama-server"
    binary.write_text(f"#!/bin/sh\necho '{line}'\necho 'built with AppleClang for Darwin arm64'\n")
    binary.chmod(0o755)
    agent = Agent(AgentConfig(llama_server=str(binary)), tmp_path / "agent.yaml")
    asyncio.run(agent.discover_engines())
    assert [engine["version"] for engine in agent.available_engines if engine["name"] == "llama.cpp"] == [version]
