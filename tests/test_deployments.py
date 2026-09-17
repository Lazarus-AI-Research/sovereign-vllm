"""Deployments: created and removed through the admin API, each its own
process with its own port and admission gate. Only the OS child and its HTTP
wire are replaced."""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, AgentRole, load_agent_config
from lazarus.agent.server import Agent, build_app


@pytest.fixture(autouse=True)
def no_native_availability_probe(monkeypatch):
    async def unavailable(agent):
        agent.available_engines = []

    monkeypatch.setattr("lazarus.agent.server.Agent.discover_engines", unavailable)


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    """An agent with one installer role, a fake llama-server that answers
    health and inference on whatever port it was started on, and a record of
    every child spawned and stopped."""
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "agent-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SOVEREIGN_AGENT_MODEL_ROOT", str(tmp_path / "models"))
    models = tmp_path / "models" / "metal"
    models.mkdir(parents=True)
    weights = models / "second.gguf"
    weights.write_bytes(b"second model")
    projector = models / "second-mmproj.gguf"
    projector.write_bytes(b"projector")
    generation = models / "generation.gguf"
    generation.write_bytes(b"generation")
    children, stopped = [], []
    healthy_ports = set()
    inference = []

    def spawn(command, **kwargs):
        port = int(command[command.index("--port") + 1])
        child = SimpleNamespace(alive=True, command=command, port=port, env=kwargs.get("env"))
        child.poll = lambda: None if child.alive else 0
        child.wait = lambda timeout=None: 0

        def stop():
            child.alive = False
            stopped.append(port)
            healthy_ports.discard(port)

        child.terminate = child.kill = stop
        children.append(child)
        # A refused spawn is a child that never answers; the next spawn on the
        # same port may, which is what a rollback relies on.
        if control.refuse_next > 0:
            control.refuse_next -= 1
            healthy_ports.discard(port)
        else:
            healthy_ports.add(port)
        return child

    monkeypatch.setattr("lazarus.agent.server.subprocess.Popen", spawn)

    class Body(httpx.AsyncByteStream):
        def __init__(self, value):
            self.value = json.dumps(value).encode()

        async def __aiter__(self):
            yield self.value

    def answer(value):
        return httpx.Response(200, stream=Body(value), headers={"content-type": "application/json"})

    async def wire(request):
        port = request.url.port
        if port not in healthy_ports:
            raise httpx.ConnectError("refused")
        if request.url.path == "/health":
            return answer({"status": "ok"})
        inference.append((port, request.url.path, request.headers.get("Authorization"), request.content))
        if request.url.path == "/v1/embeddings":
            return answer({"data": [{"embedding": [0.1, 0.2]}]})
        return answer({"choices": [{"message": {"content": f"from {port}"}}]})

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=httpx.MockTransport(wire), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    control = SimpleNamespace(refuse_next=0)
    agent = Agent(AgentConfig(roles={
        "generation": AgentRole(model_path=str(generation), port=9101, revision="a" * 40, context_length=8192),
    }), tmp_path / "agent.yaml")
    agent.deployment_ready_timeout = 2
    agent.save_config()
    return SimpleNamespace(
        agent=agent, children=children, stopped=stopped, inference=inference, control=control,
        weights=weights, projector=projector, config_path=tmp_path / "agent.yaml",
        headers={"Authorization": "Bearer agent-secret"},
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def second_request(harness, **overrides):
    request = {
        "kind": "generation", "artifact": "metal/second.gguf", "sha256": digest(harness.weights),
        "mmproj": "metal/second-mmproj.gguf", "mmproj_sha256": digest(harness.projector),
        "revision": "c" * 40, "served_model_name": "assistant-second", "context_length": 4096,
    }
    request.update(overrides)
    return request


def test_deployment_is_its_own_process_on_its_own_port_and_is_persisted(harness):
    with TestClient(build_app(harness.agent)) as api:
        created = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] == "healthy" and body["port"] == 9110 and body["model"] == "/models/metal/second.gguf"

        child = harness.children[-1]
        assert child.port == 9110 and child.command[child.command.index("-c") + 1] == "4096"
        assert child.command[child.command.index("--alias") + 1] == "assistant-second"
        assert "--jinja" in child.command and "--mmproj" in child.command and child.command[-1] == "--metrics"
        assert child.env["LLAMA_API_KEY"].endswith("-agent")

        listed = api.get("/agent/deployments", headers=harness.headers).json()["deployments"]
        assert listed["assistant-second"]["served_model_name"] == "assistant-second"
        assert listed["assistant-second"]["status"] == "healthy"
        manifest = api.get("/agent/manifest", headers=harness.headers).json()
        assert "assistant-second" in manifest["deployments"] and "generation" in manifest["roles"]

    saved = yaml.safe_load(harness.config_path.read_text())
    assert saved["deployments"]["assistant-second"]["port"] == 9110
    reloaded = load_agent_config(harness.config_path)
    assert reloaded.deployments["assistant-second"].kind == "generation"


def test_deployment_proxy_forwards_to_that_process_and_only_its_kind(harness):
    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        answer = api.post("/deployments/assistant-second/v1/chat/completions", headers=harness.headers, json={"messages": []})
        assert answer.status_code == 200 and answer.json()["choices"][0]["message"]["content"] == "from 9110"
        port, path, authorization, _ = harness.inference[-1]
        assert (port, path) == (9110, "/v1/chat/completions") and authorization.startswith("Bearer ")

        assert api.post("/deployments/assistant-second/v1/embeddings", headers=harness.headers, json={}).status_code == 404
        assert api.post("/deployments/nope/v1/chat/completions", headers=harness.headers, json={}).status_code == 404
        assert api.post("/deployments/assistant-second/v1/chat/completions", json={}).status_code == 401


def test_embedding_deployment_gets_pooling_flags_and_is_probed(harness):
    with TestClient(build_app(harness.agent)) as api:
        created = api.put("/agent/admin/deployments/embed-two", headers=harness.headers, json=second_request(
            harness, kind="embedding", mmproj=None, mmproj_sha256=None, served_model_name="embedding-two",
            pooling="last", normalization="none", context_length=2048,
        ))
        assert created.status_code == 200, created.text
        command = harness.children[-1].command
        assert command[1:6] == ["--embedding", "--pooling", "last", "--embd-normalize", "-1"]
        assert harness.inference[-1][1] == "/v1/embeddings"
        assert harness.children[-1].env is None


def test_removing_a_deployment_stops_only_that_process(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        api.put("/agent/admin/deployments/assistant-third", headers=harness.headers, json=second_request(harness, served_model_name="assistant-third"))
        removed = api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers)
        assert removed.status_code == 200 and removed.json()["status"] == "stopped"
        assert harness.stopped == [9110]
        assert "assistant-second" not in harness.agent.config.deployments
        assert "assistant-third" in harness.agent.deployments and harness.agent.roles["generation"].running()
        assert api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers).json()["status"] == "absent"
    assert "assistant-second" not in yaml.safe_load(harness.config_path.read_text()).get("deployments", {})


def test_failed_start_leaves_no_deployment_and_replacing_restores_the_previous(harness):
    with TestClient(build_app(harness.agent)) as api:
        harness.control.refuse_next = 1
        failed = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        assert failed.status_code == 422, failed.text
        assert failed.json()["rolled_back"] is True
        assert "assistant-second" not in harness.agent.config.deployments and harness.stopped == [9110]
        assert "deployments" not in yaml.safe_load(harness.config_path.read_text()) or not yaml.safe_load(harness.config_path.read_text())["deployments"]

        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        first = harness.agent.deployments["assistant-second"]

        harness.control.refuse_next = 1
        replaced = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, context_length=2048))
        # The candidate could not serve, so the earlier process is running again
        # with the earlier configuration.
        assert replaced.status_code == 422
        restored = harness.agent.config.deployments["assistant-second"]
        assert restored.context_length == 4096 and harness.agent.deployments["assistant-second"] is not first
        assert api.get("/agent/deployments", headers=harness.headers).json()["deployments"]["assistant-second"]["status"] in {"healthy", "loading"}


@pytest.mark.parametrize("bad", [
    {"artifact": "../second.gguf"}, {"sha256": "0" * 64}, {"mmproj": "metal/second-mmproj.gguf", "mmproj_sha256": None},
    {"served_model_name": "bad name"}, {"kind": "generation", "pooling": "last"},
])
def test_bad_requests_are_refused_before_any_process_starts(harness, bad):
    with TestClient(build_app(harness.agent)) as api:
        response = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, **bad))
        assert response.status_code == 422, response.text
        assert len(harness.children) == 1
    assert api.put("/agent/admin/deployments/Bad_ID", headers=harness.headers, json=second_request(harness)).status_code == 422
    assert api.put("/agent/admin/deployments/generation", headers=harness.headers, json=second_request(harness)).status_code == 409


def test_restarted_agent_serves_its_recorded_deployments(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
    restarted = Agent(load_agent_config(harness.config_path), harness.config_path)
    restarted.start_roles()
    assert {child.port for child in harness.children[-2:]} == {9101, 9110}
    assert asyncio.run(restarted.wait_ready(timeout=5)) is None
    assert "assistant-second" in restarted.deployments


# A deployment that hangs must not make the manifest, which the runtime reads
# with a five-second budget, fail for the roles beside it.
def test_a_hung_deployment_does_not_stall_the_manifest(harness, monkeypatch):
    import time as clock

    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))

        async def hang():
            await asyncio.sleep(30)

        # Only the deployment's own probe hangs; the manifest's role probes are
        # unbounded and would otherwise stall the request themselves.
        monkeypatch.setattr(harness.agent.deployments["assistant-second"], "healthy", hang)
        started = clock.monotonic()
        manifest = api.get("/agent/manifest", headers=harness.headers)
        assert manifest.status_code == 200 and clock.monotonic() - started < 4
        assert manifest.json()["deployments"]["assistant-second"]["status"] == "loading"


# Listing a deployment that is removed while it is being probed reports it
# absent rather than failing the whole listing.
def test_listing_survives_a_deployment_removed_mid_probe(harness, monkeypatch):
    from lazarus.agent.server import RoleProcess

    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        original = RoleProcess.healthy

        async def remove_then_answer(self):
            harness.agent.config.deployments = {}
            harness.agent.deployments.pop("assistant-second", None)
            return await original(self)

        monkeypatch.setattr(RoleProcess, "healthy", remove_then_answer)
        listed = api.get("/agent/deployments", headers=harness.headers)
        assert listed.status_code == 200 and listed.json()["deployments"] == {}


# A removal whose record cannot be written is reported as a failure and
# finished by the retry, never as an absence a restarted agent would contradict.
def test_removal_that_cannot_persist_is_retried_not_forgotten(harness, monkeypatch):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        real_save = harness.agent.save_config
        attempts = []

        def failing_save():
            attempts.append(1)
            if len(attempts) == 1:
                raise OSError("disk full")
            real_save()

        monkeypatch.setattr(harness.agent, "save_config", failing_save)
        first = api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers)
        assert first.status_code == 500 and "assistant-second" in harness.agent.config.deployments
        assert "assistant-second" in yaml.safe_load(harness.config_path.read_text())["deployments"]
        second = api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers)
        assert second.status_code == 200 and second.json()["status"] == "stopped"
        assert "assistant-second" not in yaml.safe_load(harness.config_path.read_text()).get("deployments", {})


# The checksum of a multi-gigabyte artifact never runs on the event loop.
def test_artifacts_are_checksummed_off_the_event_loop(harness, monkeypatch):
    threads = []
    real_to_thread = asyncio.to_thread

    async def recording_to_thread(function, *args, **kwargs):
        threads.append(getattr(function, "__name__", str(function)))
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", recording_to_thread)
    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
    assert threads.count("resolve_model") == 2
