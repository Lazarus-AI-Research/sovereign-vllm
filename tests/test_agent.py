"""Host agent contracts and in-process native protocol tests; no child is spawned."""

import asyncio
import hashlib
import json
import os
from types import SimpleNamespace

import httpx
import yaml

import pytest
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, AgentRole, load_agent_config
from lazarus.agent.server import AGENT_VERSION, Agent, RoleProcess, build_app
from lazarus.appliance.backends.agent import AgentBackend
from lazarus.appliance.launcher import Appliance


@pytest.fixture(autouse=True)
def no_native_availability_probe(monkeypatch):
    async def unavailable(agent):
        agent.available_engines = []

    monkeypatch.setattr("lazarus.agent.server.Agent.discover_engines", unavailable)


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


@pytest.mark.parametrize("model_path", [
    "", "model.gguf", "models/model.gguf", "//models/model.gguf",
    "/models/./model.gguf", "/models/dir/../model.gguf", "/models//model.gguf", "/models/model.gguf/",
])
def test_agent_config_rejects_noncanonical_primary_model_path(tmp_path, model_path):
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump({"roles": {"generation": {"model_path": model_path, "port": 9101}}}))
    with pytest.raises(ValueError, match="canonical absolute path"):
        load_agent_config(config_path)


@pytest.mark.parametrize("relative", [
    "parent with spaces/file.gguf", "nested dir/" * 45 + "file.gguf",
    "é/" * 165 + "file.gguf", "x" + "é" * 127,
    "modèles (reviewed)/weights..v2;[Q4]\\final.gguf",
])
def test_agent_config_and_resolver_preserve_bounded_native_identity(tmp_path, monkeypatch, relative):
    # Private host-root bytes do not count against the /models/ identity bound.
    root = tmp_path / "private host root"
    model = root / relative
    model.parent.mkdir(parents=True)
    model.write_bytes(b"native model fixture")
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(root))
    config_path = tmp_path / "agent.yaml"
    config_path.write_text(yaml.safe_dump({"roles": {"generation": {"model_path": str(model), "port": 9101}}}))
    config = load_agent_config(config_path)
    agent = Agent(config, config_path)
    identity = f"/models/{relative}"
    assert config.roles["generation"].model_path == str(model)
    assert agent.observed_model(config.roles["generation"].model_path) == identity
    if model.suffix == ".gguf":
        assert agent.resolve_model(relative, hashlib.sha256(model.read_bytes()).hexdigest()) == model
    if relative.startswith(("nested dir/", "é/")):
        assert len(identity.encode("utf-8")) == 512
        assert len(str(model).encode("utf-8")) > 512


@pytest.mark.parametrize("relative", [
    "nested dir/" * 45 + "xfile.gguf", "a" * 256,
    "dir/./file.gguf", "dir/../file.gguf", "dir//file.gguf", "file.gguf/",
    "é/" * 165 + "xfile.gguf", "é" * 128,
    "model\tname.gguf", "model\nname.gguf", "model\x00.gguf", "model\x7f.gguf", "model\u0085.gguf", "model\ud800.gguf",
])
def test_native_admission_rejects_unsupported_paths_before_spawn(native_protocol, relative):
    native = native_protocol
    children = list(native.children)
    # Mutated desired configuration must be rechecked even after YAML parsing.
    native.agent.config.roles["generation"].model_path = f"{native.agent.model_root}/{relative}"
    with pytest.raises(ValueError):
        native.agent.start_role("generation")
    assert native.children == children
    with pytest.raises(ValueError):
        native.agent.resolve_model(relative, "a" * 64)


@pytest.mark.parametrize("kind", ["file-symlink", "directory-symlink", "outside-root"])
def test_native_admission_rejects_noncurrent_managed_paths(native_protocol, kind):
    native = native_protocol
    root = native.agent.model_root
    original = native.model
    if kind == "file-symlink":
        path = root / "link.gguf"
        path.symlink_to(original)
    elif kind == "directory-symlink":
        link = root / "linked"
        link.symlink_to(original.parent, target_is_directory=True)
        path = link / original.name
    else:
        path = original
        native.agent.model_root = root / "different-current-root"
    native.agent.config.roles["generation"].model_path = str(path)
    children = list(native.children)
    with pytest.raises(ValueError):
        native.agent.start_role("generation")
    assert native.children == children
    if kind != "outside-root":
        with pytest.raises(ValueError, match="symlinks"):
            native.agent.resolve_model(path.relative_to(root).as_posix(), hashlib.sha256(original.read_bytes()).hexdigest())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    agent = Agent(AgentConfig(roles={}))
    return TestClient(build_app(agent))


def test_auth_fails_closed(client):
    assert client.get("/agent/manifest").status_code == 401
    assert (
        client.get("/agent/manifest", headers={"Authorization": "Bearer wrong"}).status_code == 401
    )


def test_manifest_shape(client):
    resp = client.get("/agent/manifest", headers={"Authorization": "Bearer agent-secret"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent_version"] == AGENT_VERSION == "0.1.0-rc.5"
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


@pytest.mark.parametrize("relative", ["custom.gguf", "nested dir/" * 45 + "file.gguf", "é/" * 165 + "file.gguf"])
def test_embedding_admin_only_accepts_managed_verified_models(tmp_path, monkeypatch, relative):
    model_root = tmp_path / "models"
    model_root.mkdir()
    artifact = model_root / relative
    artifact.parent.mkdir(parents=True, exist_ok=True)
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
                "artifact": relative,
                "revision": "a" * 40,
                "sha256": checksum,
                "pooling": "mean",
                "normalization": "l2",
            },
        )
        assert accepted.status_code == 200
        assert accepted.json()["model"] == f"/models/{relative}"
        assert agent.config.roles["embedding"].model_path == str(artifact)
        assert "--embd-normalize" in agent.config.roles["embedding"].args

        removed = admin.delete(
            "/agent/admin/roles/embedding",
            headers={"Authorization": "Bearer agent-secret"},
        )
        assert removed.status_code == 200
        assert "embedding" not in agent.config.roles


@pytest.mark.parametrize("artifact", [
    "nested dir/" * 45 + "xfile.gguf", "é/" * 165 + "xfile.gguf",
    "generation/./model.gguf", "generation/../embedding/model.gguf", "generation//model.gguf",
])
def test_embedding_admin_rejects_native_path_before_changing_current_role(native_protocol, artifact):
    native = native_protocol
    previous = native.agent.roles["embedding"]
    saved = native.agent.config_path.read_bytes()
    children = list(native.children)
    response = TestClient(build_app(native.agent)).put(
        "/agent/admin/roles/embedding",
        headers={"Authorization": "Bearer agent-secret"},
        json={"artifact": artifact, "revision": "a" * 40, "sha256": "a" * 64},
    )
    assert response.status_code == 422
    assert native.agent.roles["embedding"] is previous and previous.running()
    assert native.agent.config_path.read_bytes() == saved
    assert native.children == children


@pytest.mark.parametrize("rollback_ready", [True, False])
def test_embedding_spawn_failure_verifies_rollback(tmp_path, monkeypatch, rollback_ready):
    from lazarus.agent.config import AgentRole

    root = tmp_path / "models"
    root.mkdir()
    old = root / "old.gguf"
    old.write_bytes(b"old")
    candidate = root / "new.gguf"
    candidate.write_bytes(b"new")
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(root))
    original = AgentRole(model_path=str(old), revision="a" * 40, port=9102)
    agent = Agent(AgentConfig(roles={"embedding": original}), tmp_path / "agent.yaml")
    agent.save_config()
    attempts = []

    class Process:
        def stop(self):
            pass

    agent.roles["embedding"] = Process()

    def start(name):
        attempts.append(agent.config.roles[name].model_path)
        if agent.config.roles[name].model_path == str(candidate):
            raise OSError("candidate spawn failed")
        return Process()

    async def ready(role, timeout=120):
        if not rollback_ready:
            raise RuntimeError("restored embedding probe failed")

    monkeypatch.setattr(agent, "start_role", start)
    monkeypatch.setattr(agent, "wait_role_ready", ready)
    response = TestClient(build_app(agent)).put(
        "/agent/admin/roles/embedding", headers={"Authorization": "Bearer agent-secret"},
        json={"artifact": "new.gguf", "revision": "b" * 40,
              "sha256": hashlib.sha256(b"new").hexdigest()},
    )
    assert response.status_code == 422
    assert response.json()["rollback_verified"] is rollback_ready
    assert response.json()["rolled_back"] is rollback_ready
    assert agent.config.roles["embedding"] == original
    assert attempts == [str(candidate), str(old)]


def test_agent_app_rejects_empty_configured_token(monkeypatch):
    monkeypatch.delenv("SOVEREIGN_AGENT_TOKEN", raising=False)
    client = TestClient(build_app(Agent(AgentConfig(roles={}))))
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


def test_metal_manifest_preserves_roles_without_mislabeling_agent_version(config_file, monkeypatch):
    import asyncio

    import yaml

    from lazarus.appliance.backends.agent import AgentBackend
    from lazarus.appliance.config import load_config
    from lazarus.appliance.manifest import ManifestBuilder
    from lazarus.appliance.state import StateMachine

    data = yaml.safe_load(config_file.read_text())
    data["runtime"]["profile"] = "metal-arm64"
    config_file.write_text(yaml.safe_dump(data))
    config = load_config(config_file)
    monkeypatch.delenv("SOVEREIGN_PROFILE", raising=False)
    monkeypatch.setenv("SOVEREIGN_AGENT_BACKEND_ID", "metal")
    backend = AgentBackend()
    state = StateMachine()

    async def agent_manifest(enabled_roles):
        assert enabled_roles == ["generation", "embedding"]
        return {
            "agent_version": AGENT_VERSION,
            "engine": "llama.cpp",
            "backend": "metal",
            "generation_paused": False,
            "roles": {
                name: {
                    "status": "healthy",
                    "model": f"/models/{name}/model.gguf",
                    "revision": "a" * 40,
                    "context_length": 8192,
                }
                for name, role in config.roles.items()
            },
        }

    async def embedding_probe(_body):
        return {"data": [{"embedding": [0.0] * 384}]}

    monkeypatch.setattr(backend, "_wait_for_agent", agent_manifest)
    monkeypatch.setattr(backend, "embeddings", embedding_probe)

    async def exercise():
        try:
            await backend.start(config, state.transition)
            state.transition("healthy")
            return ManifestBuilder(state=state, backend=backend, config=config, port=8000).build()
        finally:
            await backend.shutdown()

    manifest = asyncio.run(exercise())
    assert backend._agent_manifest["agent_version"] == AGENT_VERSION
    assert backend.engine_version() is None
    assert "engine" not in manifest
    assert "vllm_version" not in manifest
    assert manifest["state"] == "healthy"
    assert manifest["backend"] == "metal"
    assert manifest["profile"] == "metal-arm64"
    assert manifest["roles"]["generation"]["status"] == "healthy"
    assert manifest["roles"]["generation"]["engine_model"] == "/models/generation/model.gguf"
    assert manifest["roles"]["generation"]["revision"] == "a" * 40
    assert manifest["roles"]["embedding"]["dimensions"] == 384
    assert manifest["roles"]["embedding"]["engine_model"] == "/models/embedding/model.gguf"
    assert manifest["accelerator"] == {
        "vendor": "apple",
        "device_count": 1,
        "unified_memory": True,
    }


@pytest.fixture
def native_protocol(config_file, tmp_path, monkeypatch):
    """Only the OS child and its HTTP wire are replaced, not either adapter."""
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_URL", "http://agent")
    monkeypatch.setenv("SOVEREIGN_RUNTIME_API_KEY", "runtime-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("SOVEREIGN_PROFILE", "metal-arm64")
    monkeypatch.delenv("SOVEREIGN_RUNTIME_MANIFEST", raising=False)
    raw = yaml.safe_load(config_file.read_text())
    raw["runtime"]["profile"] = "metal-arm64"
    config_file.write_text(yaml.safe_dump(raw))
    model = tmp_path / "generation" / "model.gguf"
    embedding_model = tmp_path / "embedding" / "model.gguf"
    for path in (model, embedding_model):
        path.parent.mkdir()
        path.write_bytes(b"native model")
    children, events = [], []

    def spawn(command, **kwargs):
        child = SimpleNamespace(alive=True, command=command, env=kwargs["env"])
        child.poll = lambda: None if child.alive else 0
        child.wait = lambda timeout: 0

        def stop():
            child.alive = False
            events.append("stop")

        child.terminate = child.kill = stop
        children.append(child)
        return child

    monkeypatch.setattr("lazarus.agent.server.subprocess.Popen", spawn)
    agent = Agent(AgentConfig(roles={
        "generation": AgentRole(model_path=str(model), port=9101, revision="a" * 40, context_length=8192),
        "embedding": AgentRole(model_path=str(embedding_model), port=9102, revision="b" * 40, context_length=2048),
    }), tmp_path / "agent.yaml")
    agent.start_roles()
    agent.save_config()
    control = SimpleNamespace(
        metrics="llamacpp:requests_processing 0\nllamacpp:requests_deferred 0\n",
        metrics_status=200, unauth_status=401, snapshots=[], metrics_gate=None,
        metrics_seen=asyncio.Event(), stream_started=asyncio.Event(), stream_finish=asyncio.Event(),
        stream_closed=asyncio.Event(), terminal=True, send_failure=False,
        lose_ack=False, ack=None, die_during_metrics=False,
        manifest_failure=None, native_status=200, native_body=None,
        manifest_transform=None,
        native_media="application/json", generation_bodies=[],
        fence_status=200, fence_unauth_status=401, fence_body=[], fence_gate=None,
        fence_seen=asyncio.Event(), fence_lost=False, fence_die=False,
    )

    class NativeBytes(httpx.AsyncByteStream):
        def __init__(self, body):
            self.body = body if isinstance(body, bytes) else json.dumps(body).encode()

        async def __aiter__(self):
            yield self.body

    def json_response(body, *, status=200, media="application/json"):
        return httpx.Response(status, stream=NativeBytes(body), headers={"content-type": media})

    class NativeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            control.stream_started.set()
            await control.stream_finish.wait()
            if control.terminal:
                yield b"data: [DO"
                yield b"NE]\n\n"

        async def aclose(self):
            control.stream_closed.set()

    async def native(request):
        generation = request.url.port == 9101
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/lora-adapters":
            assert generation
            if not request.headers.get("Authorization"):
                return httpx.Response(control.fence_unauth_status)
            assert request.headers["Authorization"] == agent.roles["generation"].headers()["Authorization"]
            events.append("fence")
            control.fence_seen.set()
            if control.fence_gate is not None:
                await control.fence_gate.wait()
            if control.fence_lost:
                raise httpx.ReadError("lost native FIFO acknowledgement", request=request)
            if control.fence_die:
                children[0].alive = False
            return json_response(control.fence_body, status=control.fence_status)
        if request.url.path == "/metrics":
            assert generation
            if not request.headers.get("Authorization"):
                return httpx.Response(control.unauth_status)
            assert request.headers["Authorization"] == agent.roles["generation"].headers()["Authorization"]
            events.append("metrics")
            control.metrics_seen.set()
            if control.metrics_gate is not None:
                await control.metrics_gate.wait()
            if control.die_during_metrics:
                children[0].alive = False
            text = control.snapshots.pop(0) if control.snapshots else control.metrics
            return httpx.Response(control.metrics_status, text=text)
        if generation:
            assert request.headers["Authorization"] == agent.roles["generation"].headers()["Authorization"]
            events.append("generation")
            control.generation_bodies.append(request.content)
            if control.native_body is not None:
                return json_response(control.native_body, status=control.native_status, media=control.native_media)
            if control.send_failure:
                raise httpx.ReadError("lost native response", request=request)
            if b'"stream":true' in request.content or b'"stream": true' in request.content:
                return httpx.Response(200, stream=NativeStream(), headers={"content-type": "text/event-stream"})
            return json_response({"choices": [{"text": "ok", "message": {"content": "ok"}}]})
        assert request.url.path == "/v1/embeddings"
        events.append("embedding")
        return json_response({"data": [{"embedding": [0.0] * 384}]})

    native_transport = httpx.MockTransport(native)
    apps = {"agent": httpx.ASGITransport(app=build_app(agent))}

    class ProtocolTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.host == "127.0.0.1":
                return await native_transport.handle_async_request(request)
            if request.url.host == "agent" and request.url.path == "/agent/manifest":
                assert request.headers.get("Authorization") == "Bearer agent-secret"
                failure = control.manifest_failure
                if failure == "unavailable":
                    raise httpx.ConnectError("host admission unavailable", request=request)
            response = await apps[request.url.host].handle_async_request(request)
            if request.url.host == "agent" and request.url.path == "/agent/manifest":
                body = json.loads(await response.aread())
                await response.aclose()
                if control.manifest_failure == "missing":
                    body.pop("generation_paused")
                elif control.manifest_failure == "untyped":
                    body["generation_paused"] = 0
                if control.manifest_transform is not None:
                    body = control.manifest_transform(body)
                return httpx.Response(200, content=json.dumps(body), headers={"content-type": "application/json"})
            if request.url.host == "agent" and request.url.path.endswith(("/quiesce", "/resume")):
                if control.lose_ack:
                    await response.aclose()
                    raise httpx.ReadError("lost agent acknowledgement", request=request)
                if control.ack is not None:
                    await response.aclose()
                    return httpx.Response(200, json=control.ack)
            return response

    original_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.setdefault("transport", ProtocolTransport())
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)

    def runtime():
        appliance = Appliance(config_path=str(config_file), backend=AgentBackend())
        apps["runtime"] = httpx.ASGITransport(app=appliance.app)
        return appliance

    async def start(appliance):
        await appliance.backend.start(appliance.config, appliance.state.transition)
        appliance.state.transition("healthy")

    return SimpleNamespace(agent=agent, control=control, children=children, events=events,
                           runtime=runtime, start=start, model=model, client=client)


@pytest.mark.parametrize("name", ["generation", "embedding"])
@pytest.mark.parametrize("args", [
    [], ["-m", "/other/model.gguf"], ["--model", "/other/model.gguf"],
    ["--model-url", "https://example.invalid/model.gguf"],
    ["--hf-repo", "owner/other"], ["--hf_repo", "owner/other"],
    ["--docker-repo", "ai/other"], ["--gpt-oss-20b-default"],
])
def test_native_start_keeps_primary_file_authoritative(native_protocol, monkeypatch, name, args):
    native = native_protocol
    for key in ("MODEL", "MODEL_URL", "HF_REPO", "DOCKER_REPO"):
        monkeypatch.setenv(f"LLAMA_ARG_{key}", "competing-primary-input")
    configured = native.agent.config.roles[name]
    configured.args = args
    native.agent.roles[name].stop()
    role = native.agent.start_role(name)
    command = native.children[-1].command
    owned = [
        "--host", "127.0.0.1", "--port", str(configured.port),
        "--model-url", "", "--hf-repo", "", "--docker-repo", "",
        "-m", role.model_path,
    ]
    # The actual child argv must clear the b9960 post-parse selectors after
    # extra args/presets and supply exactly the recorded primary loader input.
    assert command[:1 + len(args)] == [native.agent.config.llama_server, *args]
    assert command[1 + len(args):1 + len(args) + len(owned)] == owned
    assert role.model_path == configured.model_path
    assert role.revision == configured.revision
    assert role.context_length == configured.context_length
    role.stop()


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_native_lifespan_cleans_child_when_later_model_is_rejected(native_protocol, monkeypatch, caplog, cleanup_failure):
    native = native_protocol
    native.agent.stop()
    native.agent.roles.clear()
    native.children.clear()
    native.events.clear()
    link = native.agent.model_root / "linked.gguf"
    link.symlink_to(native.model)
    native.agent.config.roles["embedding"].model_path = str(link)

    if cleanup_failure:
        original_stop = RoleProcess.stop

        def stop(role):
            original_stop(role)
            raise OSError("native child cleanup failed")

        monkeypatch.setattr(RoleProcess, "stop", stop)

    app = build_app(native.agent)

    async def exercise():
        with pytest.raises(ValueError, match="managed model paths cannot contain symlinks"):
            async with app.router.lifespan_context(app):
                pytest.fail("later model admission must reject startup")

    asyncio.run(exercise())
    assert len(native.children) == 1
    assert list(native.agent.roles) == ["generation"]
    generation = native.agent.roles["generation"]
    assert generation.process is native.children[0]
    assert generation.model_path == str(native.model)
    assert not generation.running()
    assert native.events == ["stop"]
    if cleanup_failure:
        assert "native child cleanup failed" in caplog.text


@pytest.mark.parametrize("cleanup_failure", [False, True])
def test_native_lifespan_monitors_readiness_and_stops_every_child(native_protocol, monkeypatch, cleanup_failure):
    native = native_protocol
    native.agent.stop()
    native.agent.roles.clear()
    native.children.clear()
    native.events.clear()
    cleanup_error = OSError("generation child cleanup failed")
    if cleanup_failure:
        original_stop = RoleProcess.stop

        def stop(role):
            original_stop(role)
            if role.name == "generation":
                raise cleanup_error

        monkeypatch.setattr(RoleProcess, "stop", stop)

    async def exercise():
        ready = asyncio.Event()
        resumed = asyncio.Event()
        wait_ready = native.agent.wait_ready
        resume_generation = native.agent.resume_generation

        async def monitor():
            await wait_ready()
            ready.set()

        async def resume():
            result = await resume_generation()
            resumed.set()
            return result

        monkeypatch.setattr(native.agent, "wait_ready", monitor)
        monkeypatch.setattr(native.agent, "resume_generation", resume)
        app = build_app(native.agent)
        async with app.router.lifespan_context(app):
            await asyncio.wait_for(ready.wait(), 2)
            await asyncio.wait_for(resumed.wait(), 2)
            assert len(native.children) == 2
            assert all(child.alive for child in native.children)
            assert native.events == []

    if cleanup_failure:
        with pytest.raises(OSError) as error:
            asyncio.run(exercise())
        assert error.value is cleanup_error
    else:
        asyncio.run(exercise())
    assert all(not child.alive for child in native.children)
    assert native.events == ["stop", "stop"]


@pytest.mark.parametrize("layout", ["staged", "spaces", "boundary", "unicode-boundary", "punctuation"])
def test_native_manifest_projects_started_files_and_metadata_end_to_end(native_protocol, layout):
    native = native_protocol
    expected = {}
    for name, revision, context in (("generation", "c" * 40, 4096), ("embedding", "d" * 40, 1024)):
        relative = f"staged/{name}/{'e' * 64}/artifact"
        if layout == "spaces":
            relative = f"parent with spaces/{name}/model.gguf"
        elif layout == "boundary":
            relative = f"{name}/" + "nested dir/" * 44
            relative += "x" * (504 - len(relative) - len("file.gguf")) + "file.gguf"
            assert len(f"/models/{relative}".encode("utf-8")) == 512
        elif layout == "unicode-boundary":
            relative = f"{name}/" + "é/" * 160
            relative += "x" * (504 - len(relative.encode("utf-8")) - len("file.gguf")) + "file.gguf"
            assert len(f"/models/{relative}".encode("utf-8")) == 512
        elif layout == "punctuation":
            relative = f"modèles (reviewed)/{name}/weights..v2;[Q4]\\final.gguf"
        path = native.agent.model_root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(name.encode())
        native.agent.roles[name].stop()
        configured = native.agent.config.roles[name]
        configured.model_path = str(path)
        configured.revision = revision
        configured.context_length = context
        native.agent.roles[name] = native.agent.start_role(name)
        expected[name] = {
            "status": "healthy", "model": f"/models/{relative}",
            "revision": revision, "context_length": context,
        }
        # Changing desired configuration must not relabel the running child.
        configured.model_path = str(native.model)
        configured.revision = "f" * 40
        configured.context_length = 32768
    native.agent.config.roles.pop("embedding")
    native.agent.available_engines = [{
        "name": "llama.cpp", "version": "b9960-discovery-only",
        "adapter": "metal-host-agent", "variants": ["metal-arm64"],
    }]

    async def exercise():
        async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
            response = await admin.get("/agent/manifest")
            assert response.status_code == 200
            assert response.json()["roles"] == expected
            assert str(native.agent.model_root) not in response.text
        appliance = native.runtime()
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                response = await client.get("/runtime/manifest")
                assert response.status_code == 200
                manifest = response.json()
                for name, observed in expected.items():
                    assert manifest["roles"][name]["status"] == "healthy"
                    assert manifest["roles"][name]["engine_model"] == observed["model"]
                    assert manifest["roles"][name]["revision"] == observed["revision"]
                    assert manifest["roles"][name]["context_length"] == observed["context_length"]
                assert manifest["roles"]["embedding"]["dimensions"] == 384
                assert str(native.agent.model_root) not in response.text
                assert appliance.backend.engine_version() is None
                assert "engine" not in manifest and "vllm_version" not in manifest
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [
    "missing", "outside", "sibling-root", "file-symlink", "directory-symlink",
    "internal-symlink", "directory", "relative", "dot", "parent", "duplicate-slash", "empty",
    "oversized", "control-character", "unicode-oversized",
])
def test_native_manifest_withholds_unprovable_file_identity(native_protocol, failure):
    native = native_protocol
    root = native.agent.model_root
    path = native.model
    if failure == "missing":
        path.unlink()
    elif failure == "outside":
        path = root.parent / f"{root.name}-outside.gguf"
        path.write_bytes(b"outside")
    elif failure == "sibling-root":
        path = root.with_name(root.name + "-other") / "model.gguf"
        path.parent.mkdir()
        path.write_bytes(b"outside")
    elif failure in {"file-symlink", "internal-symlink"}:
        target = root.parent / f"{root.name}-outside.gguf" if failure == "file-symlink" else root / "other.gguf"
        target.write_bytes(b"other")
        path.unlink()
        path.symlink_to(target)
    elif failure == "directory-symlink":
        target = root / "renamed-generation"
        path.parent.rename(target)
        path.parent.symlink_to(target, target_is_directory=True)
    elif failure == "directory":
        path = root
    elif failure == "relative":
        path = "generation/model.gguf"
    elif failure == "dot":
        path = f"{root}/generation/./model.gguf"
    elif failure == "parent":
        path = f"{root}/generation/../generation/model.gguf"
    elif failure == "duplicate-slash":
        path = f"{root}/generation//model.gguf"
    elif failure == "empty":
        path = ""
    elif failure in {"oversized", "control-character", "unicode-oversized"}:
        relative = {
            "oversized": "nested dir/" * 45 + "xfile.gguf",
            "control-character": "model\tname.gguf",
            "unicode-oversized": "é/" * 165 + "xfile.gguf",
        }[failure]
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unsupported native path")
    native.agent.roles["generation"].model_path = str(path)

    async def exercise():
        async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
            response = await admin.get("/agent/manifest")
            assert response.status_code == 200
            observed = response.json()["roles"]
            assert observed["generation"] == {"status": "unhealthy", "error_code": "MODEL_LOAD_FAILED"}
            assert observed["embedding"]["status"] == "healthy"
            assert str(root) not in response.text
        appliance = native.runtime()
        try:
            await appliance.backend.start(appliance.config, appliance.state.transition)
            assert appliance.backend.role_info("generation").status == "unhealthy"
            assert appliance.backend.role_info("generation").engine_model is None
            assert appliance.backend.role_client("generation") is None
            assert appliance.backend.role_info("embedding").status == "healthy"
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("change", ["wrong-role", "replaced", "removed"])
def test_native_manifest_rejects_changed_role_ownership(native_protocol, monkeypatch, change):
    native = native_protocol
    role = native.agent.roles["generation"]

    async def healthy():
        if change == "wrong-role":
            role.name = "embedding"
        elif change == "replaced":
            native.agent.roles["generation"] = native.agent.roles["embedding"]
        else:
            native.agent.roles.pop("generation")
        return True

    monkeypatch.setattr(role, "healthy", healthy)

    async def exercise():
        async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
            roles = (await admin.get("/agent/manifest")).json()["roles"]
            assert roles["generation"] == {"status": "unhealthy", "error_code": "MODEL_LOAD_FAILED"}
            assert roles["embedding"]["status"] == "healthy"

    asyncio.run(exercise())


@pytest.mark.parametrize("model", [
    None, "model.gguf", "owner/model", "/private/models/model.gguf", "/models",
    "/models/", "/models/../model.gguf", "/models/./model.gguf", "/models//model.gguf",
    "/models/" + "nested dir/" * 45 + "xfile.gguf", "/models/" + "a" * 256,
    "/models/" + "é/" * 165 + "xfile.gguf", "/models/" + "é" * 128,
    "/models/model\tname.gguf", "/models/model\u0085name.gguf", "/models/model\ud800.gguf",
])
def test_native_backend_rejects_legacy_or_ambiguous_model_identity(native_protocol, monkeypatch, model):
    native = native_protocol
    appliance = native.runtime()

    async def manifest(_enabled):
        return {
            "engine": "llama.cpp", "backend": "metal", "generation_paused": False,
            "roles": {name: {"status": "healthy", "model": model} for name in ("generation", "embedding")},
        }

    monkeypatch.setattr(appliance.backend, "_wait_for_agent", manifest)

    async def exercise():
        try:
            await appliance.backend.start(appliance.config, appliance.state.transition)
            for name in ("generation", "embedding"):
                info = appliance.backend.role_info(name)
                assert info.status == "unhealthy" and info.error_code == "MODEL_LOAD_FAILED"
                assert info.engine_model is None
                assert appliance.backend.role_client(name) is None
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("name,generation_enabled", [("generation", True), ("embedding", True), ("embedding", False)])
@pytest.mark.parametrize("failure", ["missing", "file-symlink", "replaced", "removed", "revision", "context_length"])
@pytest.mark.parametrize("probe", ["/runtime/manifest", "/health/ready", "/health", "/v1/models", "forward"])
def test_native_runtime_withdraws_current_role_proof(native_protocol, config_file, name, generation_enabled, failure, probe):
    native = native_protocol
    raw = yaml.safe_load(config_file.read_text())
    raw["roles"]["generation"]["enabled"] = generation_enabled
    config_file.write_text(yaml.safe_dump(raw))
    role = native.agent.roles[name]
    path = native.agent.model_root / name / "model.gguf"
    original = path.read_bytes()
    started = {"model_path": role.model_path, "revision": role.revision, "context_length": role.context_length}
    endpoints = {
        "generation": ("/v1/chat/completions", {"model": "assistant-dev", "messages": []}),
        "embedding": ("/v1/embeddings", {"model": "embedding-custom", "input": "independent"}),
    }

    async def exercise():
        appliance = native.runtime()
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                assert (await client.get("/health/ready")).status_code == 200
                before = (await client.get("/runtime/manifest")).json()["roles"]
                assert before[name]["status"] == "healthy"
                assert before[name]["engine_model"] == f"/models/{name}/model.gguf"
                if name == "embedding":
                    assert before[name]["dimensions"] == 384
                if failure == "missing":
                    path.unlink()
                elif failure == "file-symlink":
                    target = path.with_name("symlink-target.gguf")
                    target.write_bytes(original)
                    path.unlink()
                    path.symlink_to(target)
                elif failure == "replaced":
                    replacement = path.with_name("replacement.gguf")
                    replacement.write_bytes(b"different managed model")
                    role.model_path = str(replacement)
                elif failure == "removed":
                    native.agent.roles.pop(name)
                elif failure == "revision":
                    role.revision = "e" * 40
                else:
                    role.context_length *= 2
                assert role.running()  # A live loaded child is not current file/identity proof.
                calls = native.events.count(name)
                if probe == "forward":
                    endpoint, body = endpoints[name]
                    assert (await client.post(endpoint, json=body)).status_code == 503
                else:
                    observed = await client.get(probe)
                    if probe == "/health/ready":
                        assert observed.status_code == 503
                    elif probe == "/v1/models":
                        assert endpoints[name][1]["model"] not in {item["id"] for item in observed.json()["data"]}
                    else:
                        assert observed.json()["roles"][name]["status"] == "unhealthy"
                info = appliance.backend.role_info(name)
                assert info.status == "unhealthy" and info.error_code == "MODEL_LOAD_FAILED"
                assert info.engine_model is None and info.revision is None and info.context_length is None
                assert info.dimensions is None and info.modalities is None
                assert native.events.count(name) == calls
                assert appliance.backend.generation_paused is (name == "generation")
                assert native.agent.generation_admission_paused is False
                manifest = (await client.get("/runtime/manifest")).json()["roles"]
                assert not {"engine_model", "revision", "context_length", "dimensions", "modalities"}.intersection(manifest[name])
                ready = await client.get("/health/ready")
                assert ready.status_code == 503 and ready.json()["required_roles"][name] is False
                if generation_enabled:
                    other = "embedding" if name == "generation" else "generation"
                    assert manifest[other] == before[other]
                    endpoint, body = endpoints[other]
                    assert (await client.post(endpoint, json=body)).status_code == 200
                # Restoring the same proof does not silently accept either the
                # replacement or the old identity, and cannot reopen generation.
                if failure in {"missing", "file-symlink"}:
                    if path.is_symlink():
                        path.unlink()
                    path.write_bytes(original)
                native.agent.roles[name] = role
                for field, value in started.items():
                    setattr(role, field, value)
                assert (await client.get("/runtime/manifest")).json()["roles"][name]["status"] == "unhealthy"
                assert (await client.get("/health/ready")).status_code == 503
                endpoint, body = endpoints[name]
                assert (await client.post(endpoint, json=body)).status_code == 503
                assert native.events.count(name) == calls
                assert appliance.backend.generation_paused is (name == "generation")
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("name,generation_enabled", [("generation", True), ("embedding", True), ("embedding", False)])
@pytest.mark.parametrize("failure", ["role-null", "role-list", "status", "model-null", "model-list", "model-relative", "model-traversal"])
def test_native_runtime_rejects_malformed_current_role_proof(native_protocol, config_file, name, generation_enabled, failure):
    native = native_protocol
    raw = yaml.safe_load(config_file.read_text())
    raw["roles"]["generation"]["enabled"] = generation_enabled
    config_file.write_text(yaml.safe_dump(raw))

    def malformed(manifest):
        role = manifest["roles"][name]
        if failure == "role-null":
            manifest["roles"][name] = None
        elif failure == "role-list":
            manifest["roles"][name] = []
        elif failure == "status":
            role["status"] = True
        else:
            role["model"] = {"model-null": None, "model-list": [], "model-relative": "model.gguf", "model-traversal": "/models/../model.gguf"}[failure]
        return manifest

    async def exercise():
        appliance = native.runtime()
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                assert (await client.get("/health/ready")).status_code == 200
                native.control.manifest_transform = malformed
                calls = native.events.count(name)
                endpoint, body = (
                    ("/v1/completions", {"model": "assistant-dev", "prompt": "must not forward"})
                    if name == "generation" else
                    ("/v1/embeddings", {"model": "embedding-custom", "input": "must not forward"})
                )
                assert (await client.post(endpoint, json=body)).status_code == 503
                assert native.events.count(name) == calls
                manifest = (await client.get("/runtime/manifest")).json()["roles"]
                assert manifest[name]["status"] == "unhealthy"
                assert not {"engine_model", "revision", "context_length", "dimensions", "modalities"}.intersection(manifest[name])
                if generation_enabled:
                    other = "embedding" if name == "generation" else "generation"
                    assert manifest[other]["status"] == "healthy"
                native.control.manifest_transform = None
                assert (await client.get("/health/ready")).status_code == 503
                assert (await client.post(endpoint, json=body)).status_code == 503
                assert appliance.backend.generation_paused is (name == "generation")
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["manifest-null", "manifest-list", "roles-missing", "roles-null", "roles-list", "backend", "engine"])
def test_native_runtime_rejects_replaced_current_manifest(native_protocol, failure):
    native = native_protocol

    def malformed(manifest):
        if failure == "manifest-null":
            return None
        if failure == "manifest-list":
            return []
        if failure == "roles-missing":
            manifest.pop("roles")
        elif failure == "roles-null":
            manifest["roles"] = None
        elif failure == "roles-list":
            manifest["roles"] = []
        elif failure == "backend":
            manifest["backend"] = "unexpected"
        elif failure == "engine":
            manifest["engine"] = "slimserve"
        return manifest

    async def exercise():
        appliance = native.runtime()
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                assert (await client.get("/health/ready")).status_code == 200
                native.control.manifest_transform = malformed
                response = await client.get("/runtime/manifest")
                assert response.status_code == 200
                manifest = response.json()
                assert manifest["generation_paused"] is True
                for name in ("generation", "embedding"):
                    role = manifest["roles"][name]
                    if failure == "engine" and name == "embedding":
                        assert role["status"] == "healthy" and role["dimensions"] == 384
                        assert (await client.post("/v1/embeddings", json={"model": "embedding-custom", "input": "independent"})).status_code == 200
                    else:
                        assert role["status"] == "unhealthy"
                        assert not {"engine_model", "revision", "context_length", "dimensions", "modalities"}.intersection(role)
                assert (await client.get("/health/ready")).status_code == 503
                native.control.manifest_transform = None
                assert (await client.get("/health/ready")).status_code == 503
                assert (await client.post("/v1/completions", json={"model": "assistant-dev", "prompt": "still fenced"})).status_code == 503
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


def test_native_runtime_embedding_only_ignores_paused_host_generation(native_protocol, config_file):
    native = native_protocol
    raw = yaml.safe_load(config_file.read_text())
    raw["roles"]["generation"]["enabled"] = False
    config_file.write_text(yaml.safe_dump(raw))

    async def exercise():
        appliance = native.runtime()
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                assert (await client.get("/health/ready")).status_code == 200
                native.agent.generation_admission_paused = True
                native.model.unlink()
                ready = await client.get("/health/ready")
                assert ready.status_code == 200 and ready.json()["required_roles"] == {"embedding": True}
                manifest = (await client.get("/runtime/manifest")).json()
                assert manifest["roles"]["generation"]["status"] == "disabled"
                assert manifest["roles"]["embedding"]["status"] == "healthy"
                assert manifest["roles"]["embedding"]["dimensions"] == 384
                assert manifest["generation_paused"] is False
                body = {"model": "embedding-custom", "input": "generation is not required"}
                assert (await client.post("/v1/embeddings", json=body)).status_code == 200
                # Disabling generation must not disable fresh embedding proof.
                (native.agent.model_root / "embedding/model.gguf").unlink()
                calls = native.events.count("embedding")
                assert (await client.post("/v1/embeddings", json=body)).status_code == 503
                assert native.events.count("embedding") == calls
                assert (await client.get("/health/ready")).status_code == 503
                assert native.agent.generation_admission_paused is True
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


def test_native_runtime_current_proof_withdrawal_preserves_admitted_stream(native_protocol):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        stream = None
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                stream = asyncio.create_task(client.post("/v1/chat/completions", json={
                    "model": "assistant-dev", "messages": [], "stream": True,
                }))
                await asyncio.wait_for(native.control.stream_started.wait(), 2)
                native.model.unlink()
                assert (await client.get("/health/ready")).status_code == 503
                assert appliance.backend.role_client("generation") is None
                assert not stream.done() and not native.control.stream_closed.is_set()
                assert native.agent.generation_requests == 1 and native.children[0].alive
                assert (await client.post("/v1/completions", json={"model": "assistant-dev", "prompt": "new request"})).status_code == 503
                assert (await client.post("/v1/embeddings", json={"model": "embedding-custom", "input": "independent"})).status_code == 200
                native.control.stream_finish.set()
                result = await asyncio.wait_for(stream, 2)
                assert result.status_code == 200 and "data: [DONE]" in result.text
                assert native.control.stream_closed.is_set() and native.agent.generation_requests == 0
                assert native.children[0].alive and native.children[1].alive
                assert (await client.get("/health/ready")).status_code == 503
        finally:
            if stream is not None and not stream.done():
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("name", ["generation", "embedding"])
def test_native_runtime_rechecks_current_proof_after_admission_wait(native_protocol, monkeypatch, name):
    from contextlib import asynccontextmanager

    from lazarus.appliance.api import Admission

    native = native_protocol

    async def exercise():
        waiting, admit = asyncio.Event(), asyncio.Event()
        original_slot = Admission.slot

        @asynccontextmanager
        async def delayed_slot(self, role):
            async with original_slot(self, role):
                waiting.set()
                await admit.wait()
                yield

        monkeypatch.setattr(Admission, "slot", delayed_slot)
        appliance = native.runtime()
        pending = None
        try:
            await native.start(appliance)
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                assert (await client.get("/health/ready")).status_code == 200
                calls = native.events.count(name)
                endpoint, body = (
                    ("/v1/completions", {"model": "assistant-dev", "prompt": "queued"})
                    if name == "generation" else
                    ("/v1/embeddings", {"model": "embedding-custom", "input": "queued"})
                )
                pending = asyncio.create_task(client.post(endpoint, json=body))
                await asyncio.wait_for(waiting.wait(), 2)
                (native.agent.model_root / name / "model.gguf").unlink()
                admit.set()
                assert (await asyncio.wait_for(pending, 2)).status_code == 503
                assert native.events.count(name) == calls
                assert appliance.backend.role_info(name).status == "unhealthy"
        finally:
            if pending is not None and not pending.done():
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await appliance.backend.shutdown()

    asyncio.run(exercise())


def test_native_runtime_quiesce_restart_embedding_and_restore(native_protocol):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        generation = native.agent.roles["generation"]
        embedding = native.agent.roles["embedding"]
        assert generation.api_key == native.children[0].env["LLAMA_API_KEY"]
        assert generation.api_key not in " ".join(native.children[0].command)
        assert "--metrics" in native.children[0].command
        assert embedding.api_key is None
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            for _ in range(2):
                assert (await client.post("/runtime/admin/generation/quiesce")).json() == {"quiesced": True}
                assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
                assert (await client.post("/v1/chat/completions", json={
                    "model": "assistant-dev", "messages": [],
                })).status_code == 503
                assert (await client.post("/v1/embeddings", json={
                    "model": "embedding-custom", "input": "independent",
                })).status_code == 200
            assert native.events.count("metrics") == 2, "repeat pause must obtain fresh proof"
            async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
                observed = (await admin.get("/agent/manifest")).json()
                assert observed["generation_paused"] is True
                assert generation.running() and embedding.running()
                response = await admin.put("/agent/admin/roles/embedding", json={
                    "artifact": native.model.relative_to(native.agent.model_root).as_posix(), "revision": "c" * 40,
                    "sha256": hashlib.sha256(native.model.read_bytes()).hexdigest(),
                })
                assert response.status_code == 200
                assert native.agent.roles["generation"] is generation
                assert generation.running() and native.agent.generation_admission_paused
            await appliance.backend.shutdown()
            assert generation.running(), "Runtime restart never kills the host generation child"
            restarted = native.runtime()
            await native.start(restarted)
            assert native.agent.roles["generation"] is generation
            assert native.agent.generation_admission_paused is False
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is False
            assert (await client.post("/runtime/admin/generation/quiesce")).status_code == 200
            async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
                before = native.events.count("metrics")
                assert (await admin.post("/agent/admin/roles/generation/llama")).status_code == 200
                assert native.events.count("metrics") == before + 1, "native cutback cannot use health as idle"
                assert native.agent.roles["generation"] is generation
            assert (await client.post("/runtime/admin/generation/resume")).json() == {"resumed": True}
            assert (await client.post("/v1/completions", json={
                "model": "assistant-dev", "prompt": "restored",
            })).status_code == 200
            await restarted.backend.shutdown()
        assert all(child.alive for child in (native.children[0], native.children[-1]))

    asyncio.run(exercise())


@pytest.mark.parametrize("through_runtime", [False, True])
@pytest.mark.parametrize("cancel_admin", [False, True])
def test_native_quiesce_never_cuts_admitted_streams(native_protocol, through_runtime, cancel_admin):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        control = native.control
        control.metrics_gate = asyncio.Event()
        headers = {"Authorization": "Bearer runtime-secret"} if through_runtime else {
            "Authorization": "Bearer agent-secret", "X-Sovereign-Role": "generation",
        }
        path = "/runtime/admin/generation/quiesce" if through_runtime else "/agent/admin/roles/generation/quiesce"
        async with native.client(base_url="http://runtime" if through_runtime else "http://agent", headers=headers) as client:
            stream = asyncio.create_task(client.post("/v1/chat/completions", json={
                "model": "assistant-dev", "messages": [], "stream": True,
            }))
            await asyncio.wait_for(control.stream_started.wait(), 2)
            pause = asyncio.create_task(client.post(path))

            async def closed():
                while not (appliance.backend.generation_paused if through_runtime else native.agent.generation_admission_paused):
                    await asyncio.sleep(0)

            await asyncio.wait_for(closed(), 2)
            assert not pause.done() and not control.metrics_seen.is_set()
            assert native.agent.generation_requests == 1
            assert not control.stream_closed.is_set()
            if cancel_admin:
                pause.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pause
                assert not stream.done() and native.children[0].alive
                pause = asyncio.create_task(client.post(path))
            control.stream_finish.set()
            result = await asyncio.wait_for(stream, 2)
            assert result.status_code == 200 and "data: [DONE]" in result.text
            await asyncio.wait_for(control.metrics_seen.wait(), 2)
            assert control.stream_closed.is_set() and native.agent.generation_requests == 0
            assert not pause.done(), "HTTP completion is not native scheduler idle proof"
            control.metrics_gate.set()
            assert (await asyncio.wait_for(pause, 2)).status_code == 200
            assert native.agent.roles["generation"].execution_uncertain is False
            assert native.children[0].alive and native.children[1].alive
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [
    "unsupported", "public", "wrong-key", "missing", "duplicate", "nan", "negative", "fractional",
    "oversized", "dead", "dies-after-probe", "no-generation", "wrong-engine",
])
def test_native_unknown_idle_proof_stays_closed(native_protocol, failure):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        control = native.control
        if failure == "unsupported":
            control.metrics_status = 501
        elif failure == "public":
            control.unauth_status = 200
        elif failure == "wrong-key":
            control.metrics_status = 401
        elif failure == "missing":
            control.metrics = "llamacpp:requests_processing 0\n"
        elif failure == "duplicate":
            control.metrics += "llamacpp:requests_processing 0\n"
        elif failure in {"nan", "negative", "fractional"}:
            value = {"nan": "NaN", "negative": "-1", "fractional": "0.5"}[failure]
            control.metrics = f"llamacpp:requests_processing {value}\nllamacpp:requests_deferred 0\n"
        elif failure == "oversized":
            control.metrics += "#" * 65537
        elif failure == "dead":
            native.children[0].alive = False
        elif failure == "dies-after-probe":
            control.die_during_metrics = True
        elif failure == "no-generation":
            native.agent.roles.pop("generation")
        elif failure == "wrong-engine":
            native.agent.generation_state = "configuration_error"
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            for operation in ("quiesce", "quiesce", "resume"):
                response = await client.post(f"/runtime/admin/generation/{operation}")
                assert response.status_code == 503
                assert response.json()["error"]["code"] == f"ENGINE_{operation.upper()}_FAILED"
                assert native.agent.generation_admission_paused is True
                assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
                assert (await client.get("/health/ready")).status_code == 503
            assert (await client.post("/v1/embeddings", json={
                "model": "embedding-custom", "input": "still independent",
            })).status_code == 200
        assert "stop" not in native.events
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("busy", ["processing", "deferred"])
def test_native_waits_for_real_scheduler_counts(native_protocol, busy):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.snapshots = [native.control.metrics.replace(f"requests_{busy} 0", f"requests_{busy} 1")]
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            assert (await asyncio.wait_for(client.post("/runtime/admin/generation/quiesce"), 2)).status_code == 200
        assert native.events.count("metrics") == 2
        assert native.agent.generation_requests == 0 and native.children[0].alive
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("through_runtime", [False, True])
@pytest.mark.parametrize("failure", ["send", "missing-terminal", "disconnect"])
def test_native_uncertain_requests_cannot_be_cleared_by_zero_metrics(native_protocol, failure, through_runtime):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.send_failure = failure == "send"
        native.control.terminal = False
        headers = {"Authorization": "Bearer runtime-secret"} if through_runtime else {
            "Authorization": "Bearer agent-secret", "X-Sovereign-Role": "generation",
        }
        async with native.client(base_url="http://runtime" if through_runtime else "http://agent", headers=headers) as client:
            request = asyncio.create_task(client.post("/v1/chat/completions", json={
                "model": "assistant-dev", "messages": [], "stream": True,
            }))
            if failure == "send":
                assert (await request).status_code == 503
            else:
                await asyncio.wait_for(native.control.stream_started.wait(), 2)
                if failure == "disconnect":
                    request.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request
                else:
                    native.control.stream_finish.set()
                    assert (await request).status_code == 200
            assert native.agent.roles["generation"].execution_uncertain is True
            assert native.agent.generation_requests == 0
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as public:
                assert (await public.get("/health/ready")).status_code == 503
                assert (await public.get("/runtime/manifest")).json()["generation_paused"] is True
                assert (await public.post("/v1/embeddings", json={
                    "model": "embedding-custom", "input": "independent after uncertainty",
                })).status_code == 200
            async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
                for operation in ("quiesce", "resume", "llama"):
                    assert (await admin.post(f"/agent/admin/roles/generation/{operation}")).status_code in {422, 503}
            assert native.agent.generation_admission_paused is True
            assert "metrics" not in native.events, "zero snapshots cannot repair a high-priority queue race"
            assert "stop" not in native.events
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
@pytest.mark.parametrize("ack", ["lost", {"paused": 1, "idle": 1}, {"paused": 0}, {"status": "healthy"}])
def test_native_missing_or_untyped_ack_never_opens_runtime(native_protocol, operation, ack):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            assert (await client.post("/runtime/admin/generation/quiesce")).status_code == 200
            native.control.lose_ack = ack == "lost"
            native.control.ack = None if ack == "lost" else ack
            response = await client.post(f"/runtime/admin/generation/{operation}")
            assert response.status_code == 503
            assert appliance.backend.generation_paused is True
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
            native.control.lose_ack, native.control.ack = False, None
            # An explicit retry obtains new proof; a cached pause/health is not an ACK.
            before = native.events.count("metrics")
            assert (await client.post("/runtime/admin/generation/resume")).status_code == 200
            assert native.events.count("metrics") == before + 1
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
def test_native_managed_commands_remain_authenticated_and_bodyless(native_protocol, operation):
    async def exercise():
        async with native_protocol.client(base_url="http://agent") as client:
            path = f"/agent/admin/roles/generation/{operation}"
            assert (await client.post(path)).status_code == 401
            for body in (b"{}", b"null", b" ", b'{"abort":true}'):
                assert (await client.post(path, headers={"Authorization": "Bearer agent-secret"}, content=body)).status_code == 422
            assert native_protocol.agent.generation_admission_paused is False
            assert native_protocol.events == []

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["uncertain", "lost-resume-ack"])
def test_native_restart_never_discards_host_pause_or_uncertainty(native_protocol, failure):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        await appliance.backend.quiesce()
        await appliance.backend.shutdown()
        role = native.agent.roles["generation"]
        if failure == "uncertain":
            role.execution_uncertain = True
        else:
            native.control.lose_ack = True
        restarted = native.runtime()
        from lazarus.appliance.backends.base import BackendStartError

        with pytest.raises(BackendStartError, match="could not resume"):
            await native.start(restarted)
        assert restarted.backend.generation_paused is True
        assert native.agent.roles["generation"] is role and role.running()
        assert role.execution_uncertain is (failure == "uncertain")
        assert native.agent.roles["embedding"].running()
        assert "stop" not in native.events
        await restarted.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("flag", ["--api-key", "--api-key-file", "--api_key", "--api_key_file"])
def test_native_extra_child_credentials_are_not_an_admission_bypass(native_protocol, flag):
    before = len(native_protocol.children)
    with pytest.raises(ValueError, match="agent-owned"):
        RoleProcess(
            "generation", ["llama-server", flag, "unowned-key"], 9101, str(native_protocol.model),
            revision=None, context_length=None,
        )
    assert len(native_protocol.children) == before


def test_native_generation_credentials_never_enter_manifest_or_logs(native_protocol):
    async def exercise():
        role = native_protocol.agent.roles["generation"]
        async with native_protocol.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as client:
            body = (await client.get("/agent/manifest")).text
        assert role.api_key not in body
        assert role.api_key not in role.log_path.read_text()
        # The audited b9960 trace shows key[-4:], not the random prefix.
        assert role.api_key[-4:] == "gent"

    asyncio.run(exercise())


def test_native_commands_reject_an_unexpected_engine(native_protocol):
    async def exercise():
        async with native_protocol.client(base_url="http://agent", headers={
            "Authorization": "Bearer agent-secret", "X-Sovereign-Engine": "slimserve",
        }) as client:
            for operation in ("quiesce", "resume"):
                assert (await client.post(f"/agent/admin/roles/generation/{operation}")).status_code == 409
        assert native_protocol.agent.generation_admission_paused is True
        assert "metrics" not in native_protocol.events and "stop" not in native_protocol.events

    asyncio.run(exercise())


def test_native_failed_role_start_does_not_reopen_host_admission(native_protocol):
    async def exercise():
        appliance = native_protocol.runtime()
        await native_protocol.start(appliance)
        await appliance.backend.quiesce()
        await appliance.backend.shutdown()
        native_protocol.children[1].alive = False
        before = native_protocol.events.count("metrics")
        restarted = native_protocol.runtime()
        await restarted.backend.start(restarted.config, restarted.state.transition)
        assert restarted.backend.role_info("embedding").status == "unhealthy"
        assert restarted.backend.generation_paused is True
        assert native_protocol.agent.generation_admission_paused is True
        assert native_protocol.events.count("metrics") == before
        await restarted.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["paused", "unavailable", "missing", "untyped"])
@pytest.mark.parametrize("probe", ["/runtime/manifest", "/health/ready"])
def test_native_fresh_admission_observation_never_passively_reopens(native_protocol, failure, probe):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            assert (await client.get("/health/ready")).status_code == 200
            if failure == "paused":
                native.agent.generation_admission_paused = True
            else:
                native.control.manifest_failure = failure
            observed = await client.get(probe)
            if probe == "/runtime/manifest":
                assert observed.json()["generation_paused"] is True
                assert observed.json()["roles"]["embedding"]["status"] == "healthy"
            else:
                assert observed.status_code == 503
            assert appliance.backend.generation_paused is True
            # Losing observation closes only Runtime admission, not a fictional
            # host gate. A later genuine open-host fact still cannot clear it.
            assert native.agent.generation_admission_paused is (failure == "paused")
            native.control.manifest_failure = None
            async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret"}) as admin:
                assert (await admin.post("/agent/admin/roles/generation/resume")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
            assert (await client.post("/v1/chat/completions", json={
                "model": "assistant-dev", "messages": [],
            })).status_code == 503
            assert (await client.post("/v1/embeddings", json={
                "model": "embedding-custom", "input": "independent",
            })).status_code == 200
            before = native.events.count("metrics")
            assert (await client.post("/runtime/admin/generation/resume")).status_code == 200
            assert native.events.count("metrics") == before + 1
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.post("/v1/completions", json={
                "model": "assistant-dev", "prompt": "served after owned resume",
            })).status_code == 200
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("counts", [{}, {"n": 1}, {"n": 2, "n_cmpl": 1}])
def test_native_complete_rejections_then_valid_serving(native_protocol, stream, counts):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        if stream:
            native.control.fence_body = [{
                "id": 0, "path": "/models/adapter.gguf", "scale": 1.0,
                "task_name": "", "prompt_prefix": "",
                "alora_invocation_string": "activate", "alora_invocation_tokens": [42],
            }]
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            for messages, error in (
                ("not-an-array", {"code": 400, "type": "invalid_request_error", "message": "messages must be an array"}),
                ([{"role": "user", "content": "oversized " * 9000}], {"code": 400, "type": "exceed_context_size_error", "message": "context exceeded", "n_prompt_tokens": 9000, "n_ctx": 8192}),
            ):
                native.control.native_status = 400
                native.control.native_body = {"error": error}
                native.control.fence_gate = asyncio.Event()
                native.control.fence_seen.clear()
                body = {"model": "assistant-dev", "messages": messages, "stream": stream, **counts}
                pending = asyncio.create_task(client.post("/v1/chat/completions", json=body))
                await asyncio.wait_for(native.control.fence_seen.wait(), 2)
                assert not pending.done(), "a complete rejection cannot retire before its FIFO acknowledgement"
                assert json.loads(native.control.generation_bodies[-1]) == body
                assert native.agent.generation_requests == 1
                assert native.agent.generation_admission_paused is False
                assert (await client.get("/health/ready")).status_code == 200
                assert (await client.get("/runtime/manifest")).json()["generation_paused"] is False
                assert (await client.post("/v1/embeddings", json={"model": "embedding-custom", "input": "independent"})).status_code == 200
                native.control.fence_gate.set()
                response = await asyncio.wait_for(pending, 2)
                assert response.status_code == 400 and response.json() == {"error": error}
                assert native.agent.roles["generation"].execution_uncertain is False
                assert native.agent.generation_requests == 0
            native.control.native_body = None
            assert (await client.post("/v1/chat/completions", json={"model": "assistant-dev", "messages": []})).status_code == 200
            assert (await client.get("/health/ready")).status_code == 200
        assert native.events.count("fence") == 2 and "metrics" not in native.events
        assert "stop" not in native.events
        await appliance.backend.shutdown()

    asyncio.run(exercise())


def test_native_rejection_retirement_preserves_concurrent_generation(native_protocol):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.metrics = "llamacpp:requests_processing 1\nllamacpp:requests_deferred 0\n"
        async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret", "X-Sovereign-Role": "generation"}) as direct:
            active = asyncio.create_task(direct.post("/v1/chat/completions", json={"messages": [], "stream": True}))
            await asyncio.wait_for(native.control.stream_started.wait(), 2)
            native.control.native_status = 400
            native.control.native_body = {"error": {"code": 400, "type": "exceed_context_size_error", "message": "context exceeded", "n_prompt_tokens": 9000, "n_ctx": 8192}}
            async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
                response = await asyncio.wait_for(client.post("/v1/completions", json={"model": "assistant-dev", "prompt": "oversized " * 9000}), 2)
                assert response.status_code == 400
                assert not active.done() and native.agent.generation_requests == 1
                assert (await client.get("/health/ready")).status_code == 200
                assert (await client.get("/runtime/manifest")).json()["generation_paused"] is False
                assert native.agent.generation_admission_paused is False
                assert native.agent.roles["generation"].execution_uncertain is False
                assert "metrics" not in native.events
                native.control.native_body = None
                assert (await client.post("/v1/completions", json={"model": "assistant-dev", "prompt": "valid concurrent"})).status_code == 200
            native.control.stream_finish.set()
            assert (await asyncio.wait_for(active, 2)).status_code == 200
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["lost", "public", "unauthorized", "status", "object", "entry", "truncated", "oversized", "dead"])
def test_native_rejection_requires_owned_complete_fifo_ack(native_protocol, failure):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.native_status = 400
        native.control.native_body = {"error": {"code": 400, "type": "invalid_request_error", "message": "invalid request"}}
        if failure == "lost":
            native.control.fence_lost = True
        elif failure == "public":
            native.control.fence_unauth_status = 200
        elif failure in {"unauthorized", "status"}:
            native.control.fence_status = 401 if failure == "unauthorized" else 503
        elif failure == "object":
            native.control.fence_body = {"status": "healthy"}
        elif failure == "entry":
            native.control.fence_body = [{"status": "healthy"}]
        elif failure == "truncated":
            native.control.fence_body = b"["
        elif failure == "oversized":
            native.control.fence_body = ["x" * 65537]
        else:
            native.control.fence_die = True
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            response = await client.post("/v1/chat/completions", json={"model": "assistant-dev", "messages": "not-an-array"})
            assert response.status_code == 400
            assert native.agent.roles["generation"].execution_uncertain is True
            assert native.agent.generation_admission_paused is True
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
            assert (await client.post("/v1/embeddings", json={"model": "embedding-custom", "input": "still served"})).status_code == 200
        assert "metrics" not in native.events and "stop" not in native.events
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("prompt,counts,safe", [
    ("one", {}, True), ([10, 20], {}, True), ([10, "mixed"], {}, True),
    (["one"], {}, True), ([[10, 20]], {}, True),
    (["one", "two"], {}, False), ([[10], [20]], {}, False),
    ("one", {"n": 2}, False), ("one", {"n": 1, "n_cmpl": 2}, False),
    ("one", {"n": 2, "n_cmpl": 1}, True),
    ("one", {"n_cmpl": True}, False), ("one", {"n_cmpl": "1"}, False),
])
def test_native_rejection_respects_actual_task_cardinality(native_protocol, prompt, counts, safe):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.native_status = 400
        native.control.native_body = {"error": {"code": 400, "type": "exceed_context_size_error", "message": "context exceeded", "n_prompt_tokens": 9000, "n_ctx": 8192}}
        body = {"model": "assistant-dev", "prompt": prompt, **counts}
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            assert (await client.post("/v1/completions", json=body)).status_code == 400
            assert json.loads(native.control.generation_bodies[-1]) == body, "classification must not suppress or rewrite input"
            assert native.agent.roles["generation"].execution_uncertain is (not safe)
            assert native.agent.generation_admission_paused is (not safe)
            assert native.events.count("fence") == int(safe)
            assert (await client.get("/health/ready")).status_code == (200 if safe else 503)
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is (not safe)
        await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["server-error", "unknown-400", "wrong-code", "extra-field", "truncated", "empty", "oversized", "untyped-counts", "sse-error"])
def test_native_ambiguous_error_outcomes_stay_fenced(native_protocol, failure):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.native_status = 400
        error = {"code": 400, "type": "invalid_request_error", "message": "invalid request"}
        native.control.native_body = {"error": error}
        if failure == "server-error":
            native.control.native_status = 500
            error.update(code=500, type="server_error")
        elif failure == "unknown-400":
            error["type"] = "unproven_error"
        elif failure == "wrong-code":
            error["code"] = "400"
        elif failure == "extra-field":
            error["unproven"] = True
        elif failure == "truncated":
            native.control.native_body = b'{"error":'
        elif failure == "empty":
            native.control.native_body = b""
        elif failure == "oversized":
            error["message"] = "x" * 65537
        elif failure == "untyped-counts":
            error.update(type="exceed_context_size_error", n_prompt_tokens="9000", n_ctx=8192)
        else:
            native.control.native_status = 200
            native.control.native_media = "text/event-stream"
            native.control.native_body = b'data: {"error":{"code":400,"type":"invalid_request_error","message":"stream failed"}}\n\n'
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            response = await client.post("/v1/chat/completions", json={"model": "assistant-dev", "messages": []})
            assert response.status_code == native.control.native_status
            assert native.agent.roles["generation"].execution_uncertain is True
            assert native.agent.generation_admission_paused is True
            assert native.agent.generation_requests == 0
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.post("/v1/completions", json={"model": "assistant-dev", "prompt": "cannot reopen"})).status_code == 503
        assert "fence" not in native.events and "metrics" not in native.events
        assert native.events.count("generation") == 1
        await appliance.backend.shutdown()

    asyncio.run(exercise())


def test_native_rejection_fifo_ack_never_clears_other_uncertainty(native_protocol):
    native = native_protocol

    async def exercise():
        appliance = native.runtime()
        await native.start(appliance)
        native.control.native_status = 400
        native.control.native_body = {"error": {"code": 400, "type": "invalid_request_error", "message": "invalid request"}}
        native.control.fence_gate = asyncio.Event()
        async with native.client(base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as client:
            pending = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "assistant-dev", "messages": []}))
            await asyncio.wait_for(native.control.fence_seen.wait(), 2)
            native.control.native_body = None
            native.control.send_failure = True
            async with native.client(base_url="http://agent", headers={"Authorization": "Bearer agent-secret", "X-Sovereign-Role": "generation"}) as direct:
                assert (await direct.post("/v1/completions", json={"prompt": "lost concurrent task"})).status_code == 503
            assert native.agent.roles["generation"].execution_uncertain is True
            native.control.fence_gate.set()
            assert (await pending).status_code == 400
            assert native.agent.roles["generation"].execution_uncertain is True
            assert native.agent.generation_admission_paused is True
            assert (await client.get("/health/ready")).status_code == 503
        await appliance.backend.shutdown()

    asyncio.run(exercise())
