"""Native host lifecycle boundaries; no real engine or accelerator is launched."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace

import httpx
import yaml
import pytest
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, AgentRole, load_agent_config
from lazarus.agent.server import Agent, build_app
from lazarus.appliance.backends.agent import AgentBackend
from lazarus.appliance.backends.base import BackendStartError, RoleInfo
from lazarus.appliance.backends.slimserve_agent import SlimServeAgentBackend
from lazarus.appliance.config import RuntimeConfig, SLIMSERVE_COMMIT, QUIXICORE_METAL_COMMIT
from lazarus.appliance.backends.slimserve import CAPABILITIES
from lazarus.appliance.launcher import Appliance

HEADERS = {"Authorization": "Bearer agent-secret"}


@pytest.fixture
def native(tmp_path, monkeypatch):
    root = tmp_path / "models"
    model = root / "staged" / "artifact" / ("a" * 64) / "profile" / "model"
    model.mkdir(parents=True)
    (model / "weights.safetensors").write_bytes(b"sealed")
    old_model = root / "old.gguf"
    old_model.write_bytes(b"old generation")
    config = RuntimeConfig.model_validate({
        "schema_version": "1.2", "runtime": {"profile": "metal-arm64"},
        "roles": {"generation": {
            "enabled": True, "engine": "slimserve", "engine_profile_id": "profile",
            "source": "local", "task": "generate",
            "model": "/models/staged/artifact/" + "a" * 64 + "/profile/model",
            "revision": "b" * 40, "served_model_name": "assistant-large",
            "max_model_len": 1024, "max_concurrent_requests": 1,
            "slimserve": {
                "source_commit": SLIMSERVE_COMMIT, "profile_id": "test-profile",
                "variant": "metal", "quant": "bf16",
                "artifacts": [{"role": "model", "repository": "owner/model", "revision": "b" * 40,
                    "files": [{"file": "weights.safetensors", "size_bytes": 6, "sha256": "c" * 64}]}],
            },
        }},
    })
    events, live = [], set()
    control = SimpleNamespace(fail_validate=False, fail_start=False, fail_shutdown=False,
                              fail_quiesce=False, quiesce_gate=None, quiesce_seen=asyncio.Event())
    process_count = 0

    class Process:
        def __init__(self, name, role):
            nonlocal process_count
            process_count += 1
            self.name, self.key = name, f"{name}-{process_count}"
            self.model_path, self.port = role.model_path, role.port
            self.revision, self.context_length = role.revision, role.context_length
            if name == "generation":
                assert not live, "overlapping generation processes"
                live.add(self.key)
            self.alive = True
            events.append(("start-llama", name))

        def running(self):
            return self.alive

        async def healthy(self):
            return self.alive

        async def wait_idle(self):
            if not self.alive:
                raise RuntimeError("owned llama process stopped")
            events.append(("llama-idle-ack", self.name))

        def stop(self):
            if self.alive:
                events.append(("stop-llama", self.name))
                self.alive = False
                live.discard(self.key)

    class Backend:
        def __init__(self):
            self.alive, self.client = False, None
            self.info, self.payload = RoleInfo(status="loading"), {}
            self.generation_paused = False

        @classmethod
        async def available_engines(cls):
            return []

        async def validate(self, candidate):
            events.append(("validate", candidate.roles.generation.model))
            assert candidate.roles.generation.model == str(model)
            if control.fail_validate:
                raise BackendStartError("CONFIG_INVALID", "candidate closure rejected")

        async def start(self, candidate, callback):
            assert not live, "overlapping generation processes"
            events.append(("start-slimserve", candidate.roles.generation.model))
            if control.fail_start:
                control.fail_start = False
                raise BackendStartError("MODEL_LOAD_FAILED", "candidate startup rejected")
            self.alive = True
            live.add("slimserve")
            self.info = RoleInfo(status="healthy", engine_model=str(model), revision="b" * 40,
                                 context_length=1024, device_count=1, tensor_parallel_size=1)
            self.payload = {
                "engine_model": str(model), "revision": "b" * 40, "context_length": 1024,
                "device_count": 1, "tensor_parallel_size": 1,
                "engine": {"name": "slimserve", "version": SLIMSERVE_COMMIT, "adapter": "slimserve-runtime"},
                "kernels": {"library": "quixicore-metal", "version": f"{QUIXICORE_METAL_COMMIT}+slimserve.{SLIMSERVE_COMMIT}", "backend": "metal"},
                "engine_profile_id": "profile", "upstream_profile_id": "test-profile", "quant": "bf16",
                "max_concurrent_requests": 1, "capabilities": list(CAPABILITIES),
                "accelerator": {"vendor": "apple", "device_count": 1, "unified_memory": True,
                    "devices": [{"identity_kind": "apple_platform", "platform_id": "apple-platform-integrated-gpu-v1:" + "1" * 64,
                                 "stable_identifier": "apple-platform-integrated-gpu-v1:" + "1" * 64}]},
            }

            class ResponseBody(httpx.AsyncByteStream):
                async def __aiter__(self):
                    yield b'{"model":"assistant-large","native":true}'

            self.client = httpx.AsyncClient(base_url="http://native", transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=ResponseBody(), headers={"content-type": "application/json"})
            ))
            callback("loading")

        async def shutdown(self):
            if self.alive and control.fail_shutdown:
                raise RuntimeError("candidate process could not be reaped")
            if self.alive:
                events.append(("stop-slimserve", "generation"))
                self.alive = False
                live.discard("slimserve")
            if self.client is not None:
                await self.client.aclose()
                self.client = None

        async def quiesce(self):
            self.generation_paused = True
            control.quiesce_seen.set()
            if control.quiesce_gate is not None:
                await control.quiesce_gate.wait()
            if control.fail_quiesce:
                raise RuntimeError("native scheduler idle was not acknowledged")
            events.append(("idle-ack", "generation"))

        async def resume(self):
            self.generation_paused = False

        def role_info(self, name):
            return self.info

        def role_client(self, name):
            return self.client

        def observation(self):
            return self.payload

    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(root))
    monkeypatch.setattr("lazarus.agent.server.SlimServeBackend", Backend)
    config_path = tmp_path / "agent.yaml"
    agent = Agent(AgentConfig(roles={
        "generation": AgentRole(model_path=str(old_model), port=9101, revision="d" * 40),
        "embedding": AgentRole(model_path=str(old_model), port=9102, revision="e" * 40),
    }), config_path)
    agent.save_config()
    monkeypatch.setattr(agent, "start_role", lambda name: Process(name, agent.config.roles[name]))
    agent.start_roles()
    return SimpleNamespace(agent=agent, config=config, events=events, live=live, control=control,
                           path=config_path, model=model, root=root, Process=Process, Backend=Backend)


def test_rejected_closure_preserves_generation_and_embedding(native):
    old_generation = native.agent.roles["generation"]
    old_embedding = native.agent.roles["embedding"]
    original = native.path.read_bytes()
    native.control.fail_validate = True
    with pytest.raises(BackendStartError, match="closure rejected"):
        asyncio.run(native.agent.configure_generation(native.config))
    assert old_generation.running() and old_embedding.running()
    assert native.agent.roles["generation"] is old_generation
    assert native.agent.roles["embedding"] is old_embedding
    assert native.path.read_bytes() == original
    assert not any(event[0].startswith("stop") for event in native.events)


def test_success_validates_before_swap_and_preserves_embedding(native):
    embedding = native.agent.roles["embedding"]

    async def exercise():
        result = await native.agent.configure_generation(native.config)
        assert result["status"] == "healthy"
        assert native.agent.roles["embedding"] is embedding and embedding.running()
        assert "generation" not in native.agent.roles
        assert native.live == {"slimserve"}
        assert load_agent_config(native.path).slimserve_generation == native.config
        validation = next(i for i, event in enumerate(native.events) if event[0] == "validate")
        launch = next(i for i, event in enumerate(native.events) if event[0] == "start-slimserve")
        assert validation < native.events.index(("stop-llama", "generation")) < launch
        await native.agent.shutdown()

    asyncio.run(exercise())


def test_start_failure_restores_verified_previous_generation(native):
    native.control.fail_start = True
    embedding = native.agent.roles["embedding"]
    result = asyncio.run(native.agent.configure_generation(native.config))
    assert result["status"] == "unhealthy"
    assert result["rollback_verified"] is True and result["rolled_back"] is True
    assert native.agent.generation_state == "configuration_error"
    assert native.agent.config.slimserve_generation is None
    assert native.agent.roles["generation"].running()
    assert native.agent.roles["embedding"] is embedding and embedding.running()
    assert load_agent_config(native.path).slimserve_generation is None
    assert len(native.live) == 1


def test_failed_previous_readiness_does_not_claim_rollback(native, monkeypatch):
    native.control.fail_start = True

    async def not_ready(role, timeout=120):
        raise RuntimeError("previous generation did not become healthy")

    monkeypatch.setattr(native.agent, "wait_generation_ready", not_ready)
    result = asyncio.run(native.agent.configure_generation(native.config))
    assert result["rollback_verified"] is False and result["rolled_back"] is False
    assert "did not become healthy" in result["rollback_error"]
    assert native.agent.generation_state == "configuration_error"
    assert len(native.live) == 1


def test_failed_candidate_shutdown_never_starts_previous_tree(native, monkeypatch):
    def fail_save():
        native.control.fail_shutdown = True
        raise OSError("persist failed")

    monkeypatch.setattr(native.agent, "save_config", fail_save)
    result = asyncio.run(native.agent.configure_generation(native.config))
    assert result["rollback_verified"] is False
    assert "could not be reaped" in result["rollback_error"]
    assert native.live == {"slimserve"}
    assert native.events.count(("start-llama", "generation")) == 1
    native.control.fail_shutdown = False
    asyncio.run(native.agent.shutdown())


def test_restart_revalidates_persisted_generation_without_llama_overlap(native, monkeypatch):
    async def exercise():
        await native.agent.configure_generation(native.config)
        await native.agent.shutdown()
        restarted = Agent(load_agent_config(native.path), native.path)
        monkeypatch.setattr(restarted, "start_role", lambda name: native.Process(name, restarted.config.roles[name]))
        restarted.start_roles()
        assert set(restarted.roles) == {"embedding"}
        assert (await restarted.resume_generation())["status"] == "healthy"
        assert native.live == {"slimserve"}
        assert sum(event[0] == "validate" for event in native.events) == 2
        await restarted.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("field,value", [
    ("command", ["arbitrary"]), ("env", {"HOME": "/tmp"}), ("host_root", "/tmp"),
    ("engine_args", ["--unsafe"]),
])
def test_native_request_rejects_untyped_mutation_inputs(native, field, value):
    body = native.config.model_dump(exclude_unset=True)
    body["roles"]["generation"][field] = value
    response = TestClient(build_app(native.agent)).put("/agent/admin/roles/generation", headers=HEADERS, json=body)
    assert response.status_code == 422
    assert native.agent.roles["generation"].running()
    assert not any(event[0] == "validate" for event in native.events)


@pytest.mark.parametrize("path", [
    "/tmp/model", "/models/staged/../../model", "/models/staged/artifact/" + "a" * 64 + "/other/model",
])
def test_native_request_rejects_unmanaged_paths(native, path):
    body = native.config.model_dump(exclude_unset=True)
    body["roles"]["generation"]["model"] = path
    response = TestClient(build_app(native.agent)).put("/agent/admin/roles/generation", headers=HEADERS, json=body)
    assert response.status_code == 422
    assert native.agent.roles["generation"].running()


def test_native_request_rejects_symlink_and_runtime_env(native):
    native.model.rename(native.model.with_name("real-model"))
    native.model.symlink_to(native.model.with_name("real-model"), target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        native.agent.translate_generation(native.config)
    body = native.config.model_dump(exclude_unset=True)
    body["runtime"]["api_key_env"] = "ARBITRARY_HOST_ENV"
    response = TestClient(build_app(native.agent)).put("/agent/admin/roles/generation", headers=HEADERS, json=body)
    assert response.status_code == 422
    assert native.agent.roles["generation"].running()


def test_native_proxy_uses_backend_client_and_true_idle_ack(native):
    async def exercise():
        await native.agent.configure_generation(native.config)
        original_client = native.agent.generation_backend.role_client("generation")
        client = TestClient(build_app(native.agent))
        headers = {**HEADERS, "X-Sovereign-Role": "generation"}
        for _ in range(2):
            response = client.post("/v1/chat/completions", headers=headers, json={"model": "assistant-large"})
            assert response.status_code == 200 and response.json()["native"] is True
            assert not original_client.is_closed
        assert client.post("/v1/embeddings", headers=headers, json={}).status_code == 404
        manifest = client.get("/agent/manifest", headers=HEADERS).json()
        assert manifest["observation"]["engine_model"] == str(native.model)
        assert manifest["model_mapping"]["runtime"] == native.config.roles.generation.model
        assert manifest["roles"]["embedding"]["model"] == "/models/old.gguf"
        assert manifest["roles"]["embedding"]["revision"] == "e" * 40
        assert client.post("/agent/admin/roles/generation/quiesce").status_code == 401
        assert client.post("/agent/admin/roles/generation/quiesce", headers=HEADERS, json={}).status_code == 422
        assert client.post("/agent/admin/roles/generation/quiesce", headers=HEADERS).json() == {"paused": True, "idle": True}
        assert ("idle-ack", "generation") in native.events and native.live == {"slimserve"}
        assert client.post("/v1/chat/completions", headers=headers, json={}).status_code == 503
        assert native.agent.roles["embedding"].running()
        assert client.post("/agent/admin/roles/generation/resume", headers=HEADERS).json() == {"paused": False}
        assert client.post("/v1/chat/completions", headers=headers, json={}).status_code == 200
        await native.agent.shutdown()

    asyncio.run(exercise())


def test_unknown_llama_idle_evidence_fails_closed(native):
    response = TestClient(build_app(native.agent)).post("/agent/admin/roles/generation/quiesce", headers=HEADERS)
    assert response.status_code == 503
    assert native.agent.roles["generation"].running()


def test_typed_stop_does_not_stop_newer_control_generation(native):
    async def exercise():
        await native.agent.configure_generation(native.config)
        different = native.config.model_copy(deep=True)
        different.roles.generation.max_model_len = 2048
        assert await native.agent.stop_generation(different) is False
        assert native.live == {"slimserve"}
        assert await native.agent.stop_generation(native.config) is True
        assert not native.live
        assert native.agent.config.slimserve_generation == native.config
        assert native.agent.roles["embedding"].running()
        assert (await native.agent.resume_generation())["status"] == "healthy"
        await native.agent.shutdown()

    asyncio.run(exercise())


def test_remote_accepts_only_exact_observed_managed_mapping(native, monkeypatch):
    backend = SlimServeAgentBackend()
    backend._config = native.config
    payload = {"engine_model": str(native.model), "revision": "b" * 40, "context_length": 1024}
    manifest = {
        "engine": "slimserve", "backend": "metal", "state": "healthy", "errors": [],
        "generation_paused": False,
        "roles": {"generation": {"status": "healthy", "engine_model": str(native.model)}},
        "observation": payload,
        "model_mapping": {"runtime": native.config.roles.generation.model, "host": str(native.model)},
    }
    accepted = []

    def validate(config, observation):
        accepted.append(deepcopy(observation))
        return observation

    monkeypatch.setattr("lazarus.appliance.backends.slimserve_agent.validate_observation", validate)
    backend._accept_manifest(manifest)
    assert accepted[0]["engine_model"] == native.config.roles.generation.model
    assert backend.role_info("generation").engine_model == native.config.roles.generation.model
    wrong = deepcopy(manifest)
    wrong["observation"]["engine_model"] = str(native.model.parent / "other-model")
    wrong["roles"]["generation"]["engine_model"] = wrong["observation"]["engine_model"]
    with pytest.raises(BackendStartError, match="outside its managed input"):
        backend._accept_manifest(wrong)
    assert len(accepted) == 1
    wrong = deepcopy(manifest)
    wrong["state"] = "runtime_error"
    with pytest.raises(BackendStartError, match="not ready"):
        backend._accept_manifest(wrong)
    assert len(accepted) == 1


@pytest.mark.parametrize("embedding_model", [
    "/models/embedding/model.gguf", "/models/" + "nested dir/" * 45 + "file.gguf",
    "/models/" + "é/" * 165 + "file.gguf", "/models/modèles (reviewed)/weights..v2;[Q4].gguf",
])
def test_remote_sends_only_typed_generation_and_keeps_embedding_proxy(native, monkeypatch, embedding_model):
    from lazarus.appliance.config import RoleConfig

    config = native.config.model_copy(deep=True)
    config.runtime.api_key_env = "PRIVATE_CONTAINER_TOKEN"
    config.roles.embedding = RoleConfig(enabled=True, source="local", model="embedding.gguf", served_model_name="embedding")
    recorded = []
    payload = {"engine_model": str(native.model), "revision": "b" * 40, "context_length": 1024}
    manifest = {
        "engine": "slimserve", "backend": "metal", "state": "healthy", "errors": [],
        "generation_paused": False,
        "roles": {
            "generation": {"status": "healthy", "engine_model": str(native.model), "model": str(native.model)},
            "embedding": {"status": "healthy", "model": embedding_model, "revision": "e" * 40},
        },
        "observation": payload,
        "model_mapping": {"runtime": native.config.roles.generation.model, "host": str(native.model)},
    }

    def handle(request):
        import json

        recorded.append(request)
        assert request.headers["Authorization"] == "Bearer agent-secret"
        if request.method == "PUT":
            wire = json.loads(request.content)
            assert set(wire) == {"schema_version", "runtime", "roles"}
            assert wire["runtime"] == {"profile": "metal-arm64"}
            assert set(wire["roles"]) == {"generation"}
            assert wire["roles"]["generation"] == native.config.roles.generation.model_dump(exclude_unset=True)
            return httpx.Response(200, json={"status": "healthy"})
        if request.url.path == "/agent/manifest":
            return httpx.Response(200, json=manifest)
        if request.url.path == "/v1/embeddings":
            assert request.headers["X-Sovereign-Role"] == "embedding"
            return httpx.Response(200, json={"data": [{"embedding": [0.0, 1.0]}]})
        assert request.method == "DELETE"
        return httpx.Response(200, json={"status": "stopped"})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: original_client(*args, transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr("lazarus.appliance.backends.slimserve_agent.validate_observation", lambda config, payload: payload)
    backend = SlimServeAgentBackend()

    async def exercise():
        await backend.start(config, lambda state: None)
        assert backend.role_info("generation").engine_model == config.roles.generation.model
        assert backend.role_info("embedding").dimensions == 2
        assert backend.role_info("embedding").engine_model == embedding_model
        manifest["roles"]["embedding"]["model"] = "/models/other/model.gguf"
        backend._accept_manifest(manifest)
        assert backend.role_info("embedding").status == "unhealthy"
        assert backend.role_info("embedding").dimensions is None
        assert backend.role_info("embedding").engine_model is None
        assert backend.role_info("embedding").revision is None
        assert backend.role_client("embedding") is None
        assert backend.role_info("generation").engine_model == config.roles.generation.model
        manifest["roles"]["embedding"]["model"] = embedding_model
        backend._accept_manifest(manifest)
        assert backend.role_info("embedding").status == "unhealthy"
        assert backend.role_info("embedding").engine_model is None
        assert backend.role_info("embedding").dimensions is None
        await backend.shutdown()

    asyncio.run(exercise())
    assert recorded[0].method == "PUT" and recorded[-1].method == "DELETE"


def test_remote_monitor_clears_readiness_when_host_disappears(native, monkeypatch):
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: original_client(*args, transport=httpx.MockTransport(lambda request: httpx.Response(503)), **kwargs))

    async def no_delay(seconds):
        return None

    monkeypatch.setattr("lazarus.appliance.backends.slimserve_agent.asyncio.sleep", no_delay)
    backend = SlimServeAgentBackend()
    backend._config = native.config
    backend._observation = {"engine_model": native.config.roles.generation.model}
    states = []
    asyncio.run(backend._monitor(states.append))
    assert states == ["runtime_error"]
    assert backend.role_info("generation").status == "unhealthy"
    assert backend.observation() == {}


def test_remote_failed_resume_keeps_admission_paused(native, monkeypatch):
    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: original_client(*args, transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"paused": True})), **kwargs))
    backend = SlimServeAgentBackend()
    backend.generation_paused = True
    with pytest.raises(RuntimeError, match="did not acknowledge"):
        asyncio.run(backend.resume())
    assert backend.generation_paused is True


def test_direct_host_stream_finishes_before_true_engine_idle_ack(native):
    async def exercise():
        await native.agent.configure_generation(native.config)
        started, release = asyncio.Event(), asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b"data: first\n\n"
                await release.wait()
                yield b"data: [DONE]\n\n"

        engine = native.agent.generation_backend
        await engine.client.aclose()
        engine.client = httpx.AsyncClient(base_url="http://native", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})
        ))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(native.agent)), base_url="http://agent") as client:
            headers = {**HEADERS, "X-Sovereign-Role": "generation"}
            request = asyncio.create_task(client.post("/v1/chat/completions", headers=headers, json={"model": "assistant-large"}))
            await asyncio.wait_for(started.wait(), timeout=1)
            assert native.agent.generation_requests == 1
            pause = asyncio.create_task(client.post("/agent/admin/roles/generation/quiesce", headers=HEADERS))

            async def admission_closed():
                while not native.agent.generation_admission_paused:
                    await asyncio.sleep(0)

            await asyncio.wait_for(admission_closed(), timeout=1)
            assert not pause.done() and ("idle-ack", "generation") not in native.events
            assert (await client.post("/v1/chat/completions", headers=headers, json={})).status_code == 503
            assert native.agent.roles["embedding"].running()
            release.set()
            assert (await request).status_code == 200
            assert (await pause).json() == {"paused": True, "idle": True}
            assert native.agent.generation_requests == 0
        await native.agent.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("adapter", [AgentBackend, SlimServeAgentBackend])
@pytest.mark.parametrize("output,expected", [
    (b"version: 9960 (a935fbffe)\nbuilt with Apple clang\n", "b9960-a935fbffe"),
    (b"version: 1 (abc1234)\n", "b1-abc1234"),
    (b"unverified version\n", None), (b"x" * 4097, None),
])
def test_native_llama_availability_reports_only_observed_build(native, monkeypatch, output, expected, adapter):
    binary = native.root / "llama-server"
    binary.write_bytes(b"installed fixture")
    binary.chmod(0o700)
    native.agent.config.llama_server = str(binary)
    commands = []

    class Stdout:
        async def read(self, count):
            assert count == 4097
            return output

    class Process:
        stdout = Stdout()
        returncode = None

        async def wait(self):
            self.returncode = 0
            return 0

        def kill(self):
            self.returncode = -9

    async def spawn(*args, **kwargs):
        commands.append(args)
        return Process()

    monkeypatch.setattr("lazarus.agent.server.asyncio.create_subprocess_exec", spawn)
    asyncio.run(native.agent.discover_engines())
    assert commands == [(str(binary), "--version")]
    versions = [entry["version"] for entry in native.agent.available_engines]
    assert versions == ([] if expected is None else [expected])
    assert all("kernel_library" not in entry for entry in native.agent.available_engines)
    original_client = httpx.AsyncClient
    app = build_app(native.agent)
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: original_client(
        *args, transport=httpx.ASGITransport(app=app), **kwargs,
    ))
    assert asyncio.run(adapter().available_engines()) == native.agent.available_engines


def test_explicit_llama_cutback_preserves_embedding_and_stops_native_first(native):
    embedding = native.agent.roles["embedding"]

    async def exercise():
        await native.agent.configure_generation(native.config)
        result = await native.agent.restore_llama_generation()
        assert result["status"] == "healthy"
        assert native.agent.generation_backend is None
        assert native.agent.config.slimserve_generation is None
        assert load_agent_config(native.path).slimserve_generation is None
        assert native.agent.roles["generation"].running()
        assert native.agent.roles["embedding"] is embedding and embedding.running()
        assert len(native.live) == 1 and "slimserve" not in native.live
        stopped = native.events.index(("stop-slimserve", "generation"))
        assert native.events[stopped + 1] == ("start-llama", "generation")
        await native.agent.shutdown()

    asyncio.run(exercise())


def test_agent_availability_comes_from_authenticated_host_before_load(native, monkeypatch):
    from lazarus.appliance.backends.agent import AgentBackend

    available = [{"name": "llama.cpp", "version": "b9960-a935fbffe", "adapter": "metal-host-agent", "variants": ["metal-arm64"]}]

    def handle(request):
        assert request.url.path == "/agent/manifest"
        assert request.headers["Authorization"] == "Bearer agent-secret"
        return httpx.Response(200, json={"available_engines": available})

    original_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: original_client(*args, transport=httpx.MockTransport(handle), **kwargs))
    assert asyncio.run(AgentBackend().available_engines()) == available
    assert not any(event[0] == "validate" for event in native.events)


@pytest.mark.parametrize("operation", ["replace", "repair", "llama", "stop"])
@pytest.mark.parametrize("acknowledged", [False, True])
def test_surviving_native_scheduler_ack_precedes_every_withdrawal(native, operation, acknowledged):
    async def exercise():
        await native.agent.configure_generation(native.config)
        engine = native.agent.generation_backend
        embedding = native.agent.roles["embedding"]
        started = asyncio.Event()

        class DisconnectedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b"data: first\n\n"
                await asyncio.Event().wait()

        await engine.client.aclose()
        engine.client = httpx.AsyncClient(base_url="http://native", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=DisconnectedStream(), headers={"content-type": "text/event-stream"})
        ))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=build_app(native.agent)), base_url="http://agent", headers=HEADERS) as client:
            inference = asyncio.create_task(client.post("/v1/chat/completions", headers={"X-Sovereign-Role": "generation"}, json={}))
            await asyncio.wait_for(started.wait(), 1)
            inference.cancel()
            with pytest.raises(asyncio.CancelledError):
                await inference
            assert native.agent.generation_requests == 0 and native.agent.generation_idle.is_set()
            native.control.quiesce_gate = asyncio.Event()
            native.control.fail_quiesce = not acknowledged
            before = len(native.events)
            if operation == "replace":
                replacement = client.put("/agent/admin/roles/generation", json=native.config.model_dump(exclude_unset=True))
            elif operation == "stop":
                replacement = client.request("DELETE", "/agent/admin/roles/generation", json=native.config.model_dump(exclude_unset=True))
            else:
                replacement = client.post(f"/agent/admin/roles/generation/{operation}")
            pending = asyncio.create_task(replacement)
            await asyncio.wait_for(native.control.quiesce_seen.wait(), 1)
            assert not pending.done()
            assert engine.alive and native.agent.generation_backend is engine
            assert embedding.running() and native.agent.roles["embedding"] is embedding
            assert native.agent.generation_admission_paused is True
            assert not any(event[0].startswith(("stop", "start")) for event in native.events[before:])
            native.control.quiesce_gate.set()
            response = await asyncio.wait_for(pending, 1)
            if acknowledged:
                assert response.status_code == 200
                ordered = native.events[before:]
                assert ordered.index(("idle-ack", "generation")) < ordered.index(("stop-slimserve", "generation"))
                assert engine.alive is False
            else:
                assert response.status_code in {422, 500}
                assert native.agent.generation_backend is engine and engine.alive
                assert native.agent.generation_admission_paused is True
                assert not any(event[0].startswith(("stop", "start")) for event in native.events[before:])
            assert embedding.running()
        native.control.fail_quiesce = False
        await native.agent.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("acknowledged", [False, True])
def test_runtime_restart_waits_for_surviving_host_scheduler(native, monkeypatch, acknowledged):
    monkeypatch.setenv("SOVEREIGN_AGENT_URL", "http://agent")
    monkeypatch.setenv("SOVEREIGN_RUNTIME_API_KEY", "runtime-secret")
    monkeypatch.setenv("SOVEREIGN_PROFILE", "metal-arm64")
    monkeypatch.delenv("SOVEREIGN_RUNTIME_MANIFEST", raising=False)
    native.config.runtime.api_key_env = "SOVEREIGN_RUNTIME_API_KEY"
    path = native.path.with_name("runtime.yaml")
    path.write_text(yaml.safe_dump(native.config.model_dump(exclude_unset=True)))
    original_client = httpx.AsyncClient
    host_transport = httpx.ASGITransport(app=build_app(native.agent))

    def client(*args, **kwargs):
        kwargs.setdefault("transport", host_transport)
        return original_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)

    async def exercise():
        appliance = Appliance(config_path=str(path), backend=SlimServeAgentBackend())
        await appliance.backend.start(appliance.config, appliance.state.transition)
        appliance.state.transition("healthy")
        engine = native.agent.generation_backend
        started = asyncio.Event()

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                yield b"data: first\n\n"
                await asyncio.Event().wait()

        await engine.client.aclose()
        engine.client = original_client(base_url="http://native", transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})
        ))
        async with original_client(transport=httpx.ASGITransport(app=appliance.app), base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as public:
            inference = asyncio.create_task(public.post("/v1/chat/completions", json={"model": "assistant-large", "messages": [], "stream": True}))
            await asyncio.wait_for(started.wait(), 1)
            inference.cancel()
            with pytest.raises(asyncio.CancelledError):
                await inference
        # Container loss does not run a graceful native DELETE. Stop only its
        # observer and clients; the host process and scheduler request survive.
        monitor, appliance.backend._monitor_task = appliance.backend._monitor_task, None
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        await AgentBackend.shutdown(appliance.backend)
        assert native.agent.generation_idle.is_set() and engine.alive
        native.control.quiesce_gate = asyncio.Event()
        native.control.fail_quiesce = not acknowledged
        before = len(native.events)
        restarted = Appliance(config_path=str(path), backend=SlimServeAgentBackend())
        startup = asyncio.create_task(restarted.backend.start(restarted.config, restarted.state.transition))
        await asyncio.wait_for(native.control.quiesce_seen.wait(), 1)
        assert not startup.done() and native.agent.generation_backend is engine and engine.alive
        assert not any(event[0].startswith(("start", "stop")) for event in native.events[before:])
        native.control.quiesce_gate.set()
        if acknowledged:
            await asyncio.wait_for(startup, 1)
            restarted.state.transition("healthy")
            async with original_client(transport=httpx.ASGITransport(app=restarted.app), base_url="http://runtime", headers={"Authorization": "Bearer runtime-secret"}) as public:
                assert (await public.get("/health/ready")).status_code == 200
                async with original_client(transport=host_transport, base_url="http://agent", headers=HEADERS) as admin:
                    assert (await admin.post("/agent/admin/roles/generation/quiesce")).status_code == 200
                    assert (await public.get("/runtime/manifest")).json()["generation_paused"] is True
                    assert (await public.get("/health/ready")).status_code == 503
                    assert (await admin.post("/agent/admin/roles/generation/resume")).status_code == 200
                assert (await public.get("/health/ready")).status_code == 503
                assert (await public.post("/runtime/admin/generation/resume")).status_code == 200
                assert (await public.get("/health/ready")).status_code == 200
                assert (await public.post("/v1/completions", json={"model": "assistant-large", "prompt": "resumed"})).status_code == 200
            ordered = native.events[before:]
            assert ordered.index(("idle-ack", "generation")) < ordered.index(("stop-slimserve", "generation"))
        else:
            with pytest.raises(BackendStartError, match="rejected configuration"):
                await startup
            assert native.agent.generation_backend is engine and engine.alive
            assert native.agent.generation_admission_paused is True
        native.control.fail_quiesce = False
        await restarted.backend.shutdown()
        await native.agent.shutdown()

    asyncio.run(exercise())
