"""Host agent contracts: configuration, identity bounds and the admin surface;
no child is spawned."""

import hashlib
import os

import yaml

import pytest
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, load_agent_config
from lazarus.agent.server import AGENT_VERSION, Agent, build_app


@pytest.fixture(autouse=True)
def no_engine_discovery(monkeypatch):
    async def unavailable(agent):
        agent.available_engines = []

    monkeypatch.setattr("lazarus.agent.server.Agent.discover_engines", unavailable)


def test_config_parses(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
listen: 127.0.0.1
port: 9100
deployments:
  assistant-large:
    kind: generation
    model_path: /models/gen.gguf
    mmproj_path: /models/mmproj.gguf
    revision: 69536a21d70340464240401ba38223d805f6a709
    sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    port: 9110
    served_model_name: assistant-large
    context_length: 8192
"""
    )
    config = load_agent_config(path)
    assert config.deployments["assistant-large"].mmproj_path == "/models/mmproj.gguf"
    assert config.deployments["assistant-large"].port == 9110
    assert config.llama_server == "llama-server"


# A configuration an earlier agent wrote carries the fixed roles and the
# managed-instance identity; both are read past, and the next save drops them.
def test_retired_configuration_keys_are_ignored_and_dropped(tmp_path, caplog):
    path = tmp_path / "agent.yaml"
    path.write_text(yaml.safe_dump({
        "port": 9100,
        "roles": {"generation": {"model_path": "/models/gen.gguf", "port": 9101, "args": ["--jinja"]}},
        "embeddinggemma": "/installed/embeddinggemma",
        "slimserve_generation": None,
        "runtime_instance_id": "11111111-1111-4111-8111-111111111111",
        "deployment_id": "22222222-2222-4222-8222-222222222222",
        "deployments": {},
    }))
    config = load_agent_config(path)
    assert config.port == 9100 and config.deployments == {}
    assert "retired agent configuration keys: roles, embeddinggemma" in caplog.text
    Agent(config, path).save_config()
    assert set(yaml.safe_load(path.read_text())) <= {"port", "deployments"}


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("deployments: {}\nmystery: true\n")
    with pytest.raises(Exception):
        load_agent_config(path)


@pytest.mark.parametrize("model_path", [
    "", "model.gguf", "models/model.gguf", "//models/model.gguf",
    "/models/./model.gguf", "/models/dir/../model.gguf", "/models//model.gguf", "/models/model.gguf/",
])
def test_agent_config_rejects_noncanonical_model_path(tmp_path, model_path):
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump({"deployments": {"one": {
        "kind": "generation", "model_path": model_path, "port": 9110, "revision": "a" * 40, "sha256": "a" * 64,
        "served_model_name": "one", "context_length": 8192,
    }}}))
    with pytest.raises(ValueError, match="canonical absolute path"):
        load_agent_config(config_path)


@pytest.mark.parametrize("relative", [
    "parent with spaces/file.gguf", "nested dir/" * 45 + "file.gguf",
    "é/" * 165 + "file.gguf", "x" + "é" * 127,
    "modèles (reviewed)/weights..v2;[Q4]\\final.gguf",
])
def test_resolver_preserves_bounded_native_identity(tmp_path, monkeypatch, relative):
    # Private host-root bytes do not count against the /models/ identity bound.
    root = tmp_path / "private host root"
    model = root / relative
    model.parent.mkdir(parents=True)
    model.write_bytes(b"native model fixture")
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(root))
    agent = Agent(AgentConfig(), tmp_path / "agent.yaml")
    identity = f"/models/{relative}"
    assert agent.observed_model(str(model)) == identity
    if model.suffix == ".gguf":
        assert agent.resolve_model(relative, hashlib.sha256(model.read_bytes()).hexdigest()) == model
    else:
        with pytest.raises(ValueError, match=".gguf"):
            agent.resolve_model(relative, hashlib.sha256(model.read_bytes()).hexdigest())
    if relative.startswith(("nested dir/", "é/")):
        assert len(identity.encode("utf-8")) == 512
        assert len(str(model).encode("utf-8")) > 512


CONTROL_CHARACTER_NAMES = [
    "model" + chr(9) + "name.gguf", "model" + chr(10) + "name.gguf", "model" + chr(0) + ".gguf",
    "model" + chr(127) + ".gguf", "model" + chr(0x85) + ".gguf", "model" + chr(0xD800) + ".gguf",
]


@pytest.mark.parametrize("relative", [
    "nested dir/" * 45 + "xfile.gguf", "a" * 256,
    "dir/./file.gguf", "dir/../file.gguf", "dir//file.gguf", "file.gguf/",
    "é/" * 165 + "xfile.gguf", "é" * 128,
    *CONTROL_CHARACTER_NAMES,
])
def test_resolver_rejects_unsupported_paths(tmp_path, monkeypatch, relative):
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(tmp_path))
    agent = Agent(AgentConfig(), tmp_path / "agent.yaml")
    with pytest.raises(ValueError):
        agent.resolve_model(relative, "a" * 64)
    with pytest.raises(ValueError):
        agent.observed_model(f"{tmp_path}/{relative}")


@pytest.mark.parametrize("kind", ["file-symlink", "directory-symlink", "outside-root"])
def test_resolver_rejects_noncurrent_managed_paths(tmp_path, monkeypatch, kind):
    root = tmp_path / "models"
    (root / "real").mkdir(parents=True)
    original = root / "real" / "model.gguf"
    original.write_bytes(b"weights")
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(root))
    agent = Agent(AgentConfig(), tmp_path / "agent.yaml")
    if kind == "file-symlink":
        path = root / "link.gguf"
        path.symlink_to(original)
    elif kind == "directory-symlink":
        link = root / "linked"
        link.symlink_to(original.parent, target_is_directory=True)
        path = link / original.name
    else:
        path = original
        agent.model_root = root / "different-current-root"
    with pytest.raises(ValueError):
        agent.observed_model(str(path))
    if kind != "outside-root":
        with pytest.raises(ValueError, match="symlinks"):
            agent.resolve_model(path.relative_to(root).as_posix(), hashlib.sha256(original.read_bytes()).hexdigest())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    agent = Agent(AgentConfig())
    return TestClient(build_app(agent))


HEADERS = {"Authorization": "Bearer agent-secret"}


def test_auth_fails_closed(client):
    assert client.get("/agent/manifest").status_code == 401
    assert client.get("/agent/manifest", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_manifest_shape(client):
    resp = client.get("/agent/manifest", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_version"] == AGENT_VERSION == "0.1.0-rc.9"
    assert body["backend"] == "metal"
    assert body["deployments"] == {}
    assert set(body) == {"agent_version", "backend", "available_engines", "deployments"}


# The fixed-role surface is gone with the roles: nothing answers under it,
# and nothing is proxied under /v1 without a deployment in the path.
@pytest.mark.parametrize("method,path", [
    ("PUT", "/agent/admin/roles/embedding"), ("DELETE", "/agent/admin/roles/embedding"),
    ("PUT", "/agent/admin/roles/generation"), ("POST", "/agent/admin/roles/generation/quiesce"),
    ("POST", "/v1/chat/completions"), ("GET", "/v1/models"),
])
def test_role_surface_is_retired(client, method, path):
    response = client.request(method, path, headers={**HEADERS, "X-Sovereign-Role": "generation"}, json={})
    assert response.status_code == 404


def test_agent_app_rejects_empty_configured_token(monkeypatch):
    monkeypatch.delenv("SOVEREIGN_AGENT_TOKEN", raising=False)
    client = TestClient(build_app(Agent(AgentConfig())))
    assert client.get("/agent/manifest", headers={"Authorization": "Bearer "}).status_code == 401


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
        enabled=True,
        task="generate",
        source="huggingface",
        model="google/gemma-4-E2B-it",
        served_model_name="assistant-large",
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
