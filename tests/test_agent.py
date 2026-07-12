"""Host agent unit tests: config parsing, auth fail-closed, manifest shape.
No llama-server processes are spawned (empty role set)."""

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
