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
                    "model": role.model,
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
    assert manifest["roles"]["generation"]["revision"] == "a" * 40
    assert manifest["roles"]["embedding"]["dimensions"] == 384
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
    model = tmp_path / "model.gguf"
    model.write_bytes(b"native model")
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
        "generation": AgentRole(model_path=str(model), port=9101, revision="a" * 40),
        "embedding": AgentRole(model_path=str(model), port=9102, revision="b" * 40),
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
                if failure == "missing":
                    return httpx.Response(200, json={"engine": "llama.cpp", "backend": "metal"})
                if failure == "untyped":
                    return httpx.Response(200, json={"engine": "llama.cpp", "backend": "metal", "generation_paused": 0})
            response = await apps[request.url.host].handle_async_request(request)
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
                    "artifact": native.model.name, "revision": "c" * 40,
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
        RoleProcess("generation", ["llama-server", flag, "unowned-key"], 9101, str(native_protocol.model))
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
