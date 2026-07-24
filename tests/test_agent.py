"""Host agent unit tests: config parsing, auth fail-closed, manifest shape.
No llama-server processes are spawned (empty role set)."""

import os
import hashlib

import pytest
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, load_agent_config
from lazarus.agent.server import AGENT_VERSION, Agent, build_app


def test_config_parses(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
listen: 127.0.0.1
port: 9100
roles:
  generation:
    model_path: /models/gen.gguf
    revision: 69536a21d70340464240401ba38223d805f6a709
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
    assert config.roles["generation"].revision == "69536a21d70340464240401ba38223d805f6a709"


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
    assert body["agent_version"] == AGENT_VERSION == "0.1.0-rc.3"
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


def test_embedding_admin_only_accepts_managed_verified_models(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_root.mkdir()
    artifact = model_root / "custom.gguf"
    artifact.write_bytes(b"verified gguf")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("roles: {}\n")
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(model_root))
    agent = Agent(AgentConfig(roles={}), config_path)

    class FakeProcess:
        port = 9102
        model_path = str(artifact)

        def stop(self):
            pass

    monkeypatch.setattr(agent, "start_role", lambda name: FakeProcess())

    async def ready(role, timeout=120):
        return None

    monkeypatch.setattr(agent, "wait_role_ready", ready)
    with TestClient(build_app(agent)) as admin:
        denied = admin.put(
            "/agent/admin/roles/embedding",
            headers={"Authorization": "Bearer agent-secret"},
            json={
                "artifact": "../custom.gguf",
                "revision": "a" * 40,
                "sha256": checksum,
            },
        )
        assert denied.status_code == 422

        accepted = admin.put(
            "/agent/admin/roles/embedding",
            headers={"Authorization": "Bearer agent-secret"},
            json={
                "artifact": "custom.gguf",
                "revision": "a" * 40,
                "sha256": checksum,
                "pooling": "mean",
                "normalization": "l2",
            },
        )
        assert accepted.status_code == 200
        assert agent.config.roles["embedding"].model_path == str(artifact)
        assert "--embd-normalize" in agent.config.roles["embedding"].args

        removed = admin.delete(
            "/agent/admin/roles/embedding",
            headers={"Authorization": "Bearer agent-secret"},
        )
        assert removed.status_code == 200
        assert "embedding" not in agent.config.roles


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
