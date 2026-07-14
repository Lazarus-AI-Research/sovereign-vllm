"""Host agent unit tests: config parsing, auth fail-closed, manifest shape.
No llama-server processes are spawned (empty role set)."""

import os

import pytest
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, load_agent_config
from lazarus.agent.server import Agent, build_app


def test_config_parses(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
listen: 127.0.0.1
port: 9100
roles:
  generation:
    model_path: /models/gen.gguf
    port: 9101
    context_length: 8192
  embedding:
    model_path: /models/embed.gguf
    mmproj_path: /models/mmproj.gguf
    port: 9102
    args: ["--embedding", "--pooling", "last"]
"""
    )
    config = load_agent_config(path)
    assert config.roles["embedding"].mmproj_path == "/models/mmproj.gguf"
    assert config.roles["generation"].port == 9101


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("roles: {}\nmystery: true\n")
    with pytest.raises(Exception):
        load_agent_config(path)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    agent = Agent(AgentConfig(roles={}))
    return TestClient(build_app(agent))


def test_auth_fails_closed(client):
    assert client.get("/agent/manifest").status_code == 401
    assert client.get("/agent/manifest", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_manifest_shape(client):
    resp = client.get("/agent/manifest", headers={"Authorization": "Bearer agent-secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["engine"] == "llama.cpp"
    assert body["backend"] == "metal"
    assert body["roles"] == {}


def test_proxy_requires_known_role(client):
    resp = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer agent-secret", "X-Sovereign-Role": "nope"},
        json={},
    )
    assert resp.status_code == 404


def test_tool_parser_inference():
    from lazarus.appliance.backends.vllm_engine import VllmBackend

    infer = VllmBackend._infer_tool_parser
    assert infer("google/gemma-4-E2B-it") == "gemma4"
    assert infer("Qwen/Qwen3-32B") == "hermes"
    assert infer("Qwen/Qwen3-Coder-30B") == "qwen3_coder"
    assert infer("some/unknown-model") is None


def test_vllm_backend_defaults_to_spawn_without_overriding_operator(monkeypatch):
    from lazarus.appliance.backends.vllm_engine import VllmBackend

    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    VllmBackend()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"

    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    VllmBackend()
    assert os.environ["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"


def test_generation_argv_defaults():
    from lazarus.appliance.backends.vllm_engine import VllmBackend
    from lazarus.appliance.config import RoleConfig

    backend = VllmBackend.__new__(VllmBackend)
    backend.backend_id = "cuda"
    role = RoleConfig(
        enabled=True, task="generate", source="huggingface",
        model="google/gemma-4-E2B-it", served_model_name="assistant-large",
    )
    argv = backend._role_argv("generation", role)
    joined = " ".join(argv)
    assert "--enable-auto-tool-choice" in joined
    assert "--tool-call-parser gemma4" in joined
    assert "--reasoning-parser gemma4" in joined
    assert "--enable-server-load-tracking" in joined
    assert "--disable-log-requests" in joined

    role.tool_call_parser = "off"
    role.reasoning_parser = "off"
    joined = " ".join(backend._role_argv("generation", role))
    assert "--enable-auto-tool-choice" not in joined
    assert "--reasoning-parser" not in joined
