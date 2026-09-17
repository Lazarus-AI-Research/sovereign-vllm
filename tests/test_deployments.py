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
    # The port ledger is what is under test; whatever listens on this host
    # (a live appliance, another test) must not shift the ports it hands out.
    monkeypatch.setattr("lazarus.agent.deployments.port_available", lambda port: True, raising=False)

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


async def settled(agent):
    """A transition worker holds the role lock until it is done; taking the
    lock waits for it."""
    async with agent.role_lock:
        pass


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


# A replacement or removal cancelled while it drains an in-flight request has
# changed nothing, so the process that was serving must keep admitting.
def test_a_cancelled_drain_reopens_the_deployment(harness):
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment, remove_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        admission = harness.agent.deployment_admission["assistant-second"]
        admission.enter()
        removal = asyncio.create_task(remove_deployment(harness.agent, "assistant-second"))
        await asyncio.sleep(0.05)
        assert admission.paused is True
        removal.cancel()
        with pytest.raises(asyncio.CancelledError):
            await removal
        admission.leave()
        await settled(harness.agent)
        return admission.paused, "assistant-second" in harness.agent.deployments

    paused, present = asyncio.run(scenario())
    assert paused is False and present
    assert harness.stopped == []


# Terminating a child blocks for as long as the child takes to die; that wait
# must not stall the event loop every other deployment streams on.
def test_a_slow_child_shutdown_does_not_block_other_traffic(harness):
    import time as clock

    from lazarus.agent.deployments import DeploymentRequest, apply_deployment, remove_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        child = harness.children[-1]
        original = child.terminate

        def slow_terminate():
            clock.sleep(0.4)
            original()

        child.terminate = slow_terminate
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        clock_task = asyncio.create_task(ticker())
        await remove_deployment(harness.agent, "assistant-second")
        clock_task.cancel()
        return ticks

    assert asyncio.run(scenario()) >= 5


# The fixed role names stay reserved whether or not the role is configured:
# enabling the embedding role later must never collide with a deployment.
def test_role_names_are_reserved_even_when_the_role_is_absent(harness):
    with TestClient(build_app(harness.agent)) as api:
        assert "embedding" not in harness.agent.config.roles
        refused = api.put("/agent/admin/deployments/embedding", headers=harness.headers, json=second_request(harness))
        assert refused.status_code == 409


# While a replacement waits for the old child to die, the deployment is still
# configured: a request then is told to retry, not that the name is unknown.
def test_a_deployment_between_processes_answers_503_not_404(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        harness.agent.deployment_admission["assistant-second"].paused = True
        harness.agent.deployments.pop("assistant-second")
        answer = api.post("/deployments/assistant-second/v1/chat/completions", headers=harness.headers, json={"messages": []})
        assert answer.status_code == 503
        assert api.post("/deployments/never-there/v1/chat/completions", headers=harness.headers, json={}).status_code == 404


# A replacement cancelled while the previous child is still shutting down
# restores that child and reopens admission, exactly as any other failure.
def test_a_replacement_cancelled_during_the_old_shutdown_restores_it(harness):
    import time as clock

    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        first = harness.children[-1]
        original = first.terminate

        def slow_terminate():
            clock.sleep(0.3)
            original()

        first.terminate = slow_terminate
        replacement = asyncio.create_task(apply_deployment(
            harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.1)
        replacement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replacement
        await settled(harness.agent)
        restored = harness.agent.deployments.get("assistant-second")
        return restored is not None and restored.running(), harness.agent.deployment_admission["assistant-second"].paused, harness.agent.config.deployments["assistant-second"].revision

    running, paused, revision = asyncio.run(scenario())
    assert running and paused is False and revision == "c" * 40


# A weight overwritten between a deployment's creation and an agent restart
# is refused at the restart: the record's checksum, not the path, is trusted.
def test_a_restart_refuses_a_weight_that_no_longer_matches_its_checksum(harness, caplog):
    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
    saved = load_agent_config(harness.config_path)
    assert saved.deployments["assistant-second"].mmproj_sha256 == digest(harness.projector)
    harness.weights.write_bytes(b"tampered model")
    restarted = Agent(saved, harness.config_path)
    restarted.deployment_ready_timeout = 2
    restarted.start_roles()
    assert "assistant-second" not in restarted.deployments
    assert "no longer matches its recorded checksum" in caplog.text
    restarted.stop()


# A healthy child behind a closed gate is reported paused, never healthy:
# nothing reaches it until the gate reopens.
def test_a_paused_deployment_is_not_reported_healthy(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        harness.agent.deployment_admission["assistant-second"].paused = True
        listed = api.get("/agent/deployments", headers=harness.headers).json()["deployments"]["assistant-second"]
        assert listed["status"] == "paused" and listed["admission"] == "paused"
        harness.agent.deployment_admission["assistant-second"].paused = False
        listed = api.get("/agent/deployments", headers=harness.headers).json()["deployments"]["assistant-second"]
        assert listed["status"] == "healthy" and listed["admission"] == "open"


# The old child is gone before a cancelled replacement restores it, so the
# restored child never competes with it for the port.
def test_a_cancelled_replacement_waits_for_the_old_child_to_die_first(harness):
    import time as clock

    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        first = harness.children[-1]
        original = first.terminate
        seen = {}

        def slow_terminate():
            clock.sleep(0.3)
            original()
            seen["children_when_old_died"] = len(harness.children)

        first.terminate = slow_terminate
        replacement = asyncio.create_task(apply_deployment(
            harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.1)
        replacement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await replacement
        await settled(harness.agent)
        restored = harness.agent.deployments["assistant-second"]
        return seen["children_when_old_died"], harness.children.index(next(c for c in harness.children if c.port == restored.port and c.alive))

    old_died_at, restored_index = asyncio.run(scenario())
    assert restored_index >= old_died_at


# A rollback restores the previous files only if they still match the
# checksums recorded for them; otherwise the deployment stays closed.
def test_a_rollback_refuses_previous_files_that_changed_on_disk(harness):
    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        harness.weights.write_bytes(b"rewritten under the same name")
        harness.control.refuse_next = 1
        failed = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, revision="d" * 40))
        body = failed.json()
        assert failed.status_code == 422 and body["rolled_back"] is False
        assert "no longer matches its recorded checksum" in body["rollback_error"]
        assert harness.agent.deployment_admission["assistant-second"].paused is True
        # The rejected candidate is never what is recorded.
        assert harness.agent.config.deployments["assistant-second"].revision == "c" * 40
        listed = api.get("/agent/deployments", headers=harness.headers).json()["deployments"]["assistant-second"]
        assert listed["status"] != "healthy"


# A request cancelled through the HTTP layer, under the server's own cancel
# scope, still leaves the deployment restored: the worker is never cancelled.
def test_a_request_cancelled_under_the_server_scope_still_restores(harness):
    import time as clock

    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        first = harness.children[-1]
        original = first.terminate

        def slow_terminate():
            clock.sleep(0.3)
            original()

        first.terminate = slow_terminate
        import anyio

        outcome = {}
        with anyio.CancelScope() as scope:
            async def cancel_soon():
                await asyncio.sleep(0.1)
                scope.cancel()

            canceller = asyncio.create_task(cancel_soon())
            try:
                await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40)))
            except asyncio.CancelledError:
                outcome["cancelled"] = True
            await canceller
        await settled(harness.agent)
        restored = harness.agent.deployments["assistant-second"]
        return outcome.get("cancelled"), restored.running(), harness.agent.config.deployments["assistant-second"].revision, harness.agent.deployment_admission["assistant-second"].paused

    cancelled, running, revision, paused = asyncio.run(scenario())
    assert running and revision == "c" * 40 and paused is False


# A child whose termination fails stays registered, so the next removal
# stops it again instead of reporting a process that is still there as gone.
def test_a_failed_stop_keeps_the_child_for_a_retry(harness):
    with TestClient(build_app(harness.agent), raise_server_exceptions=False) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        child = harness.children[-1]
        original = child.terminate
        attempts = {"count": 0}

        def flaky_terminate():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("wait timed out")
            original()

        child.terminate = flaky_terminate
        first = api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers)
        assert first.status_code == 500 and "wait timed out" in first.json()["error"]
        assert harness.agent.deployments["assistant-second"].process is child
        assert "assistant-second" in harness.agent.config.deployments
        second = api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers)
        assert second.status_code == 200 and second.json()["status"] == "stopped"
        assert "assistant-second" not in harness.agent.deployments and not child.alive


# The files are checked again right before they are loaded: a weight rewritten
# while the transition waited never starts under the checksum it was pinned to.
def test_a_candidate_rewritten_during_the_drain_is_refused(harness, monkeypatch):
    import lazarus.agent.deployments as deployments

    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        replacement = second_request(harness, revision="d" * 40)
        real_quiesce = deployments.quiesce

        async def rewrite_during_drain(agent, deployment_id):
            harness.weights.write_bytes(b"rewritten while draining")
            return await real_quiesce(agent, deployment_id)

        monkeypatch.setattr(deployments, "quiesce", rewrite_during_drain)
        answer = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=replacement)
        assert answer.status_code == 422
        assert "no longer matches its recorded checksum" in answer.json()["error"]
        assert harness.agent.config.deployments["assistant-second"].revision == "c" * 40


# A candidate that fails and then cannot be stopped still leaves the previous
# record as what is persisted; the failed child stays registered for a retry
# and the gate stays closed.
def test_a_candidate_that_cannot_be_stopped_still_restores_the_record(harness):
    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        real_spawn = harness.agent.deployments["assistant-second"].process
        harness.control.refuse_next = 1
        answer = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, revision="d" * 40))
        # The refused candidate is the newest child; its termination fails.
        candidate = harness.children[-1]
        assert candidate is not real_spawn
        assert answer.status_code == 422
        body = answer.json()
        assert body["rolled_back"] is True or body["rollback_error"]
        assert harness.agent.config.deployments["assistant-second"].revision == "c" * 40
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, revision="e" * 40))
        saved = load_agent_config(harness.config_path)
        assert saved.deployments["assistant-second"].revision == "e" * 40


def test_a_candidate_whose_stop_raises_keeps_the_previous_record_and_the_child(harness, monkeypatch):
    import lazarus.agent.deployments as deployments

    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        real_wait = deployments.wait_deployment_ready
        seen = {}

        async def fail_then_break_stop(agent, deployment, process):
            if deployment.revision == "d" * 40:
                seen["candidate"] = process
                process.process.terminate = lambda: (_ for _ in ()).throw(RuntimeError("wait timed out"))
                raise RuntimeError("did not become ready")
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", fail_then_break_stop)
        answer = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, revision="d" * 40))
        body = answer.json()
        assert answer.status_code == 422 and body["rolled_back"] is False and "wait timed out" in body["rollback_error"]
        assert harness.agent.config.deployments["assistant-second"].revision == "c" * 40
        assert harness.agent.deployments["assistant-second"] is seen["candidate"]
        assert harness.agent.deployment_admission["assistant-second"].paused is True
        saved = load_agent_config(harness.config_path)
        assert saved.deployments["assistant-second"].revision == "c" * 40
        # The agent's shutdown stops the child it still holds; that must work.
        candidate = seen["candidate"].process
        candidate.terminate = candidate.kill


# A first creation whose candidate fails and cannot be stopped keeps the child
# registered; the next removal stops it, and the next creation stops it before
# taking its handle.
def test_a_child_a_failed_creation_could_not_stop_is_still_stopped_later(harness, monkeypatch):
    import lazarus.agent.deployments as deployments

    with TestClient(build_app(harness.agent), raise_server_exceptions=False) as api:
        real_wait = deployments.wait_deployment_ready
        attempts = {"stops": 0}

        async def fail_and_break_stop(agent, deployment, process):
            original = process.process.terminate

            def flaky():
                attempts["stops"] += 1
                if attempts["stops"] == 1:
                    raise RuntimeError("wait timed out")
                original()

            process.process.terminate = flaky
            raise RuntimeError("did not become ready")

        monkeypatch.setattr(deployments, "wait_deployment_ready", fail_and_break_stop)
        failed = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        assert failed.status_code == 422 and failed.json()["rolled_back"] is False
        child = harness.children[-1]
        assert "assistant-second" not in harness.agent.config.deployments
        assert harness.agent.deployments["assistant-second"].process is child and child.alive
        monkeypatch.setattr(deployments, "wait_deployment_ready", real_wait)
        removed = api.delete("/agent/admin/deployments/assistant-second", headers=harness.headers)
        assert removed.status_code == 200 and removed.json()["status"] == "stopped"
        assert not child.alive and "assistant-second" not in harness.agent.deployments


def test_a_new_creation_stops_a_child_left_registered_without_a_record(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        stray = harness.agent.deployments["assistant-second"]
        # The record is gone but the child was left behind, as a failed
        # creation whose stop failed leaves it.
        harness.agent.config.deployments = {}
        created = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, revision="d" * 40))
        assert created.status_code == 200
        assert not stray.process.alive and harness.agent.deployments["assistant-second"] is not stray


# A transition cancelled while it waited for the lock touches nothing: the
# deployment is never paused for a request that is already gone.
def test_a_transition_abandoned_while_waiting_for_the_lock_touches_nothing(harness, monkeypatch):
    import time as clock

    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment, remove_deployment

    # A request in flight would make a drain wait the whole idle timeout; an
    # abandoned transition must not drain at all.
    monkeypatch.setattr(deployments, "IDLE_TIMEOUT", 3)

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        admission = harness.agent.deployment_admission["assistant-second"]
        admission.enter()
        outcomes = []
        async with harness.agent.role_lock:
            replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
            removal = asyncio.create_task(remove_deployment(harness.agent, "assistant-second"))
            await asyncio.sleep(0.05)
            for task in (replacement, removal):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    outcomes.append("cancelled")
        started = clock.monotonic()
        await settled(harness.agent)
        elapsed = clock.monotonic() - started
        admission.leave()
        return outcomes, admission.paused, harness.agent.config.deployments["assistant-second"].revision, harness.stopped, elapsed

    outcomes, paused, revision, stopped, elapsed = asyncio.run(scenario())
    assert outcomes == ["cancelled", "cancelled"] and paused is False and revision == "c" * 40 and stopped == []
    assert elapsed < 1.0
