"""Deployments: created and removed through the admin API, each its own
process with its own port and admission gate. Only the OS child and its HTTP
wire are replaced."""

import asyncio
import hashlib
import json
import sys
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from lazarus.agent.config import AgentConfig, load_agent_config
from lazarus.agent.server import Agent, build_app


@pytest.fixture(autouse=True)
def no_native_availability_probe(monkeypatch):
    async def unavailable(agent):
        agent.available_engines = []

    monkeypatch.setattr("lazarus.agent.server.Agent.discover_engines", unavailable)


@pytest.fixture()
def harness(tmp_path, monkeypatch):
    """An agent with no deployments yet, a fake llama-server that answers
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
    diffusion = models / "flux1-schnell-Q4_0.gguf"
    diffusion.write_bytes(b"diffusion")
    for name in ("clip_l-Q8_0.gguf", "t5xxl-Q8_0.gguf", "ae.safetensors"):
        (models / name).write_bytes(name.encode())
    ears = models / "ggml-small.bin"
    ears.write_bytes(b"whisper weights")
    voice = models / "en_US-ljspeech-medium.onnx"
    voice.write_bytes(b"voice weights")
    (models / "en_US-ljspeech-medium.onnx.json").write_bytes(b'{"audio": {"sample_rate": 22050}}')
    children, stopped = [], []
    healthy_ports = set()
    inference = []

    def spawn(command, **kwargs):
        port_flag = "--port" if "--port" in command else "--listen-port"
        port = int(command[command.index(port_flag) + 1])
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
        if request.url.path == "/v1/models" and request.method == "GET":
            return answer({"data": [{"id": "served", "object": "model"}]})
        if request.url.path == "/voices" and request.method == "GET":
            return answer({"en_US-ljspeech-medium": {}})
        inference.append((port, request.url.path, request.headers.get("Authorization"), request.content))
        if request.url.path == "/v1/embeddings":
            return answer({"data": [{"embedding": [0.1, 0.2]}]})
        if request.url.path == "/v1/audio/transcriptions":
            return answer({"text": f"heard on {port}"})
        if request.url.path == "/synthesize":
            return httpx.Response(200, content=b"RIFF....WAVEfake", headers={"content-type": "audio/wav"})
        return answer({"choices": [{"message": {"content": f"from {port}"}}]})

    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs.pop("transport", None)
        return real_client(*args, transport=httpx.MockTransport(wire), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    control = SimpleNamespace(refuse_next=0)
    agent = Agent(AgentConfig(), tmp_path / "agent.yaml")
    agent.deployment_ready_timeout = 2
    agent.save_config()
    return SimpleNamespace(
        agent=agent, children=children, stopped=stopped, inference=inference, control=control,
        weights=weights, projector=projector, diffusion=diffusion, ears=ears, voice=voice, models=models, config_path=tmp_path / "agent.yaml",
        headers={"Authorization": "Bearer agent-secret"},
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def settled(agent, deployment_id="assistant-second"):
    """A transition worker holds the deployment's lock until it is done;
    taking the lock waits for it."""
    from lazarus.agent.deployments import deployment_lock

    async with deployment_lock(agent, deployment_id):
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
        assert "--jinja" in child.command and "--mmproj" in child.command and "--metrics" not in child.command
        assert child.env["LLAMA_API_KEY"].endswith("-agent")

        listed = api.get("/agent/deployments", headers=harness.headers).json()["deployments"]
        assert listed["assistant-second"]["served_model_name"] == "assistant-second"
        assert listed["assistant-second"]["status"] == "healthy"
        manifest = api.get("/agent/manifest", headers=harness.headers).json()
        assert "assistant-second" in manifest["deployments"] and "roles" not in manifest

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
        assert "assistant-third" in harness.agent.deployments and harness.agent.deployments["assistant-third"].running()
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
        assert harness.children == []
    assert api.put("/agent/admin/deployments/Bad_ID", headers=harness.headers, json=second_request(harness)).status_code == 422


def test_restarted_agent_serves_its_recorded_deployments(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
    restarted = Agent(load_agent_config(harness.config_path), harness.config_path)
    restarted.start_deployments()
    assert {child.port for child in harness.children[-1:]} == {9110}
    assert asyncio.run(restarted.wait_ready(timeout=5)) is None
    assert "assistant-second" in restarted.deployments


# A deployment that hangs must not make the manifest, which Control reads
# with a five-second budget, fail for the deployments beside it.
def test_a_hung_deployment_does_not_stall_the_manifest(harness, monkeypatch):
    import time as clock

    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))

        async def hang():
            await asyncio.sleep(30)

        # Only the deployment's own probe hangs.
        monkeypatch.setattr(harness.agent.deployments["assistant-second"], "healthy", hang)
        started = clock.monotonic()
        manifest = api.get("/agent/manifest", headers=harness.headers)
        assert manifest.status_code == 200 and clock.monotonic() - started < 4
        assert manifest.json()["deployments"]["assistant-second"]["status"] == "loading"


# Listing a deployment that is removed while it is being probed reports it
# absent rather than failing the whole listing.
def test_listing_survives_a_deployment_removed_mid_probe(harness, monkeypatch):
    from lazarus.agent.server import ServerProcess

    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        original = ServerProcess.healthy

        async def remove_then_answer(self):
            harness.agent.config.deployments = {}
            harness.agent.deployments.pop("assistant-second", None)
            return await original(self)

        monkeypatch.setattr(ServerProcess, "healthy", remove_then_answer)
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
    restarted.start_deployments()
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

        async def rewrite_during_drain(agent, deployment_id, transition=None):
            harness.weights.write_bytes(b"rewritten while draining")
            return await real_quiesce(agent, deployment_id, transition)

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
        async with deployments.deployment_lock(harness.agent, "assistant-second"):
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


# A child that exited keeps its record until it is stopped, but nothing is
# forwarded to a port that may now belong to something else.
def test_an_exited_child_is_not_forwarded_to(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        harness.children[-1].alive = False
        before = len(harness.inference)
        answer = api.post("/deployments/assistant-second/v1/chat/completions", headers=harness.headers, json={"messages": []})
        assert answer.status_code == 503 and "not running" in answer.json()["error"]
        assert len(harness.inference) == before
        assert harness.agent.deployment_admission["assistant-second"].requests == 0


# A request that cannot be built never leaves an admission count behind, so
# no later transition drains for it.
def test_an_unbuildable_request_leaves_no_admission_count(harness):
    with TestClient(build_app(harness.agent)) as api:
        api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        # Sent as raw bytes: the wire carries it, and the server decodes it.
        bad = "application/json; boundary=\u00e9".encode("latin-1")
        answer = api.post("/deployments/assistant-second/v1/chat/completions", headers={**harness.headers, "Content-Type": bad}, content=b"{}")
        assert answer.status_code == 400
        assert harness.agent.deployment_admission["assistant-second"].requests == 0


# A long transition on one deployment never holds up another: only ports and
# the saved configuration are shared.
def test_a_slow_replacement_does_not_block_another_deployments_removal(harness, monkeypatch):
    import time as clock

    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment, remove_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        await apply_deployment(harness.agent, "assistant-third", DeploymentRequest(**second_request(harness, served_model_name="assistant-third")))
        real_wait = deployments.wait_deployment_ready

        async def slow_wait(agent, deployment, process):
            if deployment.revision == "d" * 40:
                await asyncio.sleep(1.0)
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", slow_wait)
        slow = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.05)
        started = clock.monotonic()
        removed = await remove_deployment(harness.agent, "assistant-third")
        elapsed = clock.monotonic() - started
        await slow
        return removed["status"], elapsed

    status, elapsed = asyncio.run(scenario())
    assert status == "stopped" and elapsed < 0.5


# Two deployments created at once never share a port: a port handed out is
# reserved until the transition that took it commits or gives it back.
def test_concurrent_creations_never_share_a_port(harness):
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    async def scenario():
        first = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness))))
        second = asyncio.create_task(apply_deployment(harness.agent, "assistant-third", DeploymentRequest(**second_request(harness, served_model_name="assistant-third"))))
        await asyncio.gather(first, second)
        return {name: d.port for name, d in harness.agent.config.deployments.items()}

    ports = asyncio.run(scenario())
    assert len(set(ports.values())) == 2 and harness.agent.port_reservations == set()


# A candidate is never in the configuration before it is confirmed, so a save
# another deployment makes meanwhile persists only what was verified.
def test_a_candidate_is_never_persisted_before_it_is_confirmed(harness, monkeypatch):
    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        real_wait = deployments.wait_deployment_ready

        async def slow_wait(agent, deployment, process):
            if deployment.revision == "d" * 40:
                await asyncio.sleep(0.4)
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", slow_wait)
        replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.1)
        async with harness.agent.records_lock:
            harness.agent.save_config()
        during = load_agent_config(harness.config_path).deployments["assistant-second"].revision
        await replacement
        after = load_agent_config(harness.config_path).deployments["assistant-second"].revision
        return during, after

    during, after = asyncio.run(scenario())
    assert during == "c" * 40 and after == "d" * 40


# A request that goes while its confirmed candidate waits for the shared lock
# is still a cancelled request: the candidate is rolled back, not committed.
def test_abandonment_is_rechecked_before_the_record_is_saved(harness, monkeypatch):
    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    seen = {}
    real_run = deployments.run_transition

    async def observed_run(agent, worker_coroutine, transition, http_request=None):
        seen["transition"] = transition
        return await real_run(agent, worker_coroutine, transition, http_request)

    monkeypatch.setattr(deployments, "run_transition", observed_run)

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        seen.clear()
        gate = asyncio.Event()
        real_wait = deployments.wait_deployment_ready

        async def gated_wait(agent, deployment, process):
            if deployment.revision == "d" * 40:
                await gate.wait()
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", gated_wait)
        replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.1)
        # The worker is past its port and waiting for the gate; the shared
        # lock is taken before the gate opens, so the confirmed candidate
        # waits to commit. Only then does the request go.
        async with harness.agent.records_lock:
            gate.set()
            for _ in range(200):
                await asyncio.sleep(0.02)
                if getattr(seen.get("transition"), "committing", False):
                    break
            assert seen["transition"].committing
            replacement.cancel()
            try:
                await replacement
            except asyncio.CancelledError:
                pass
        await settled(harness.agent)
        return harness.agent.config.deployments["assistant-second"].revision, harness.agent.deployment_admission["assistant-second"].paused, load_agent_config(harness.config_path).deployments["assistant-second"].revision

    revision, paused, saved = asyncio.run(scenario())
    assert revision == "c" * 40 and saved == "c" * 40 and paused is False


# The agent joins every transition still in flight before it stops its
# children, so a worker never starts one after the final sweep.
def test_shutdown_joins_abandoned_transitions(harness, monkeypatch):
    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        real_wait = deployments.wait_deployment_ready

        async def slow_wait(agent, deployment, process):
            await asyncio.sleep(0.3)
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", slow_wait)
        replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.05)
        replacement.cancel()
        try:
            await replacement
        except asyncio.CancelledError:
            pass
        assert harness.agent.transitions
        await harness.agent.join_transitions()
        harness.agent.stop()
        return [child.alive for child in harness.children], len(harness.agent.transitions)

    alive, remaining = asyncio.run(scenario())
    assert not any(alive) and remaining == 0


# A candidate whose save fails is not a record: the previous record stays,
# in memory as on disk, and the previous process is restored.
def test_a_candidate_whose_save_fails_is_not_persisted_by_the_rollback(harness, monkeypatch):
    with TestClient(build_app(harness.agent)) as api:
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness)).status_code == 200
        real_save = harness.agent.save_config
        failures = {"left": 1}

        def flaky_save():
            if failures["left"] > 0 and harness.agent.config.deployments["assistant-second"].revision == "d" * 40:
                failures["left"] -= 1
                raise OSError("disk full")
            real_save()

        monkeypatch.setattr(harness.agent, "save_config", flaky_save)
        answer = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, revision="d" * 40))
        body = answer.json()
        assert answer.status_code == 422 and "disk full" in body["error"] and body["rolled_back"] is True
        assert harness.agent.config.deployments["assistant-second"].revision == "c" * 40
        assert load_agent_config(harness.config_path).deployments["assistant-second"].revision == "c" * 40
        assert harness.agent.deployment_admission["assistant-second"].paused is False


# A request that goes while its worker waits for the shared lock never
# begins: no drain, no process.
def test_a_transition_abandoned_while_waiting_for_the_shared_lock_touches_nothing(harness, monkeypatch):
    import time as clock

    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    monkeypatch.setattr(deployments, "IDLE_TIMEOUT", 3)

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        admission = harness.agent.deployment_admission["assistant-second"]
        admission.enter()
        spawned = len(harness.children)
        async with harness.agent.records_lock:
            replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
            await asyncio.sleep(0.05)
            replacement.cancel()
            try:
                await replacement
            except asyncio.CancelledError:
                pass
        started = clock.monotonic()
        await settled(harness.agent)
        elapsed = clock.monotonic() - started
        admission.leave()
        return admission.paused, len(harness.children) - spawned, harness.stopped, elapsed

    paused, spawned, stopped, elapsed = asyncio.run(scenario())
    assert paused is False and spawned == 0 and stopped == [] and elapsed < 1.0


# Once the agent begins to stop, no transition starts a child: the final
# sweep never races a worker.
def test_no_child_starts_once_the_agent_is_stopping(harness):
    with TestClient(build_app(harness.agent)) as api:
        harness.agent.stopping = True
        spawned = len(harness.children)
        answer = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        assert answer.status_code == 422 and "shutting down" in answer.json()["error"]
        assert len(harness.children) == spawned
        harness.agent.stopping = False


# A child kept registered without a record still owns its port: no new
# deployment is handed it until the child is confirmed gone.
def test_a_retained_child_keeps_its_port(harness, monkeypatch):
    import lazarus.agent.deployments as deployments

    with TestClient(build_app(harness.agent), raise_server_exceptions=False) as api:
        real_wait = deployments.wait_deployment_ready

        async def fail_and_break_stop(agent, deployment, process):
            original = process.process.terminate

            def flaky():
                process.process.terminate = original
                raise RuntimeError("wait timed out")

            process.process.terminate = flaky
            raise RuntimeError("did not become ready")

        monkeypatch.setattr(deployments, "wait_deployment_ready", fail_and_break_stop)
        failed = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness))
        assert failed.status_code == 422
        retained = harness.agent.deployments["assistant-second"]
        assert "assistant-second" not in harness.agent.config.deployments
        monkeypatch.setattr(deployments, "wait_deployment_ready", real_wait)
        created = api.put("/agent/admin/deployments/assistant-third", headers=harness.headers, json=second_request(harness, served_model_name="assistant-third"))
        assert created.status_code == 200
        assert harness.agent.config.deployments["assistant-third"].port != retained.port


# A candidate that exits while its worker waits for the commit lock is a
# failed candidate: the previous process comes back and the record stands.
def test_a_candidate_that_dies_before_the_commit_is_rolled_back(harness, monkeypatch):
    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    seen = {}
    real_run = deployments.run_transition

    async def observed_run(agent, worker_coroutine, transition, http_request=None):
        seen["transition"] = transition
        return await real_run(agent, worker_coroutine, transition, http_request)

    monkeypatch.setattr(deployments, "run_transition", observed_run)

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        seen.clear()
        gate = asyncio.Event()
        real_wait = deployments.wait_deployment_ready

        async def gated_wait(agent, deployment, process):
            if deployment.revision == "d" * 40:
                await gate.wait()
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", gated_wait)
        replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.1)
        async with harness.agent.records_lock:
            gate.set()
            for _ in range(200):
                await asyncio.sleep(0.02)
                if getattr(seen.get("transition"), "committing", False):
                    break
            assert seen["transition"].committing
            # The confirmed candidate dies while it waits to be recorded.
            harness.children[-1].alive = False
        result = await replacement
        return result, harness.agent.config.deployments["assistant-second"].revision, harness.agent.deployments["assistant-second"].running(), load_agent_config(harness.config_path).deployments["assistant-second"].revision

    result, revision, running, saved = asyncio.run(scenario())
    assert result["status"] == "unhealthy" and result["rolled_back"] is True and "exited" in result["error"]
    assert revision == "c" * 40 and saved == "c" * 40 and running


# A drain waiting on an in-flight request wakes the moment its transition is
# abandoned, and the gate goes back at once.
def test_an_abandoned_drain_wakes_at_once(harness, monkeypatch):
    import time as clock

    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    monkeypatch.setattr(deployments, "IDLE_TIMEOUT", 3)

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        admission = harness.agent.deployment_admission["assistant-second"]
        admission.enter()
        replacement = asyncio.create_task(apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40))))
        await asyncio.sleep(0.1)
        assert admission.paused is True
        replacement.cancel()
        try:
            await replacement
        except asyncio.CancelledError:
            pass
        started = clock.monotonic()
        await settled(harness.agent)
        elapsed = clock.monotonic() - started
        paused = admission.paused
        admission.leave()
        return paused, elapsed, harness.stopped

    paused, elapsed, stopped = asyncio.run(scenario())
    assert paused is False and stopped == [] and elapsed < 1.0


# A client that goes away mid-transition is not cancelled by the server; the
# transition is abandoned all the same, and rolled back.
def test_a_disconnected_request_abandons_its_transition(harness, monkeypatch):
    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    monkeypatch.setattr(deployments, "DISCONNECT_POLL", 0.02)

    class GoneClient:
        def __init__(self):
            self.polls = 0

        async def is_disconnected(self):
            self.polls += 1
            return self.polls > 2

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        real_wait = deployments.wait_deployment_ready

        async def slow_wait(agent, deployment, process):
            if deployment.revision == "d" * 40:
                await asyncio.sleep(0.3)
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", slow_wait)
        result = await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness, revision="d" * 40)), http_request=GoneClient())
        return result, harness.agent.config.deployments["assistant-second"].revision, harness.agent.deployment_admission["assistant-second"].paused

    result, revision, paused = asyncio.run(scenario())
    assert result["status"] == "unhealthy" and result["rolled_back"] is True and "cancelled" in result["error"]
    assert revision == "c" * 40 and paused is False


# A real client disconnect, delivered through the application's own ASGI
# stack (authentication included), abandons the transition and rolls it back.
def test_a_real_http_disconnect_abandons_the_transition(harness, monkeypatch):
    import time as clock

    import lazarus.agent.deployments as deployments
    from lazarus.agent.deployments import DeploymentRequest, apply_deployment

    monkeypatch.setattr(deployments, "DISCONNECT_POLL", 0.02)
    app = build_app(harness.agent)

    async def scenario():
        await apply_deployment(harness.agent, "assistant-second", DeploymentRequest(**second_request(harness)))
        real_wait = deployments.wait_deployment_ready

        async def slow_wait(agent, deployment, process):
            if deployment.revision == "d" * 40:
                await asyncio.sleep(0.5)
            return await real_wait(agent, deployment, process)

        monkeypatch.setattr(deployments, "wait_deployment_ready", slow_wait)
        body = json.dumps(second_request(harness, revision="d" * 40)).encode()
        messages = [{"type": "http.request", "body": body, "more_body": False}]
        gone_at = clock.monotonic() + 0.15

        async def receive():
            if messages:
                return messages.pop(0)
            if clock.monotonic() >= gone_at:
                return {"type": "http.disconnect"}
            await asyncio.sleep(3600)

        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "PUT", "scheme": "http",
            "path": "/agent/admin/deployments/assistant-second", "raw_path": b"/agent/admin/deployments/assistant-second",
            "query_string": b"", "root_path": "", "client": ("127.0.0.1", 1), "server": ("127.0.0.1", 9100),
            "headers": [(b"authorization", b"Bearer agent-secret"), (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        }
        await app(scope, receive, send)
        await settled(harness.agent)
        return harness.agent.config.deployments["assistant-second"].revision, load_agent_config(harness.config_path).deployments["assistant-second"].revision, harness.agent.deployment_admission["assistant-second"].paused

    revision, saved, paused = asyncio.run(scenario())
    assert revision == "c" * 40 and saved == "c" * 40 and paused is False


def image_request(harness, **overrides):
    request = {
        "kind": "image", "artifact": "metal/flux1-schnell-Q4_0.gguf", "sha256": digest(harness.diffusion),
        "revision": "c" * 40, "served_model_name": "pictures",
        "components": {
            name: {"artifact": f"metal/{file}", "sha256": digest(harness.models / file)}
            for name, file in (("clip_l", "clip_l-Q8_0.gguf"), ("t5xxl", "t5xxl-Q8_0.gguf"), ("vae", "ae.safetensors"))
        },
        "steps": 4, "cfg_scale": 1.0, "sampler": "euler",
    }
    request.update(overrides)
    return request


# An image deployment is a stable-diffusion.cpp server child: its diffusion
# weights and the named components it loads beside them, each checksummed,
# started with the pinned sampling, probed on its models route, and proxied
# on the images route alone.
def test_an_image_deployment_is_an_sd_server_child(harness):
    with TestClient(build_app(harness.agent)) as api:
        created = api.put("/agent/admin/deployments/pictures", headers=harness.headers, json=image_request(harness))
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] == "healthy" and body["engine"] == "stable-diffusion.cpp" and body["model"] == "/models/metal/flux1-schnell-Q4_0.gguf"
        child = harness.children[-1]
        command = child.command
        assert command[0] == "sd-server" and child.env is None
        assert command[command.index("--diffusion-model") + 1] == str(harness.diffusion)
        assert command[command.index("--clip_l") + 1].endswith("clip_l-Q8_0.gguf")
        assert command[command.index("--t5xxl") + 1].endswith("t5xxl-Q8_0.gguf")
        assert command[command.index("--vae") + 1].endswith("ae.safetensors")
        assert command[command.index("--steps") + 1] == "4" and command[command.index("--cfg-scale") + 1] == "1.0"
        assert command[command.index("--sampling-method") + 1] == "euler" and "--port" not in command
        assert command[command.index("--listen-port") + 1] == "9110"

        pictured = api.post("/deployments/pictures/v1/images/generations", headers=harness.headers, json={"prompt": "a cat"})
        assert pictured.status_code == 200 and harness.inference[-1][0] == 9110 and harness.inference[-1][1] == "/v1/images/generations"
        assert api.post("/deployments/pictures/v1/chat/completions", headers=harness.headers, json={}).status_code == 404

    saved = load_agent_config(harness.config_path)
    record = saved.deployments["pictures"]
    assert record.kind == "image" and set(record.components) == {"clip_l", "t5xxl", "vae"} and record.steps == 4
    assert record.components["vae"].sha256 == digest(harness.models / "ae.safetensors")


# A component the server has no flag for, a component on a language model,
# or a safetensors file for llama-server are refused before anything starts;
# a component whose bytes changed since is refused at restart.
def test_image_components_are_constrained_and_verified(harness, caplog):
    with TestClient(build_app(harness.agent)) as api:
        wrong = image_request(harness)
        wrong["components"]["lora"] = wrong["components"]["vae"]
        assert api.put("/agent/admin/deployments/pictures", headers=harness.headers, json=wrong).status_code == 422
        mixed = second_request(harness, components=image_request(harness)["components"])
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=mixed).status_code == 422
        tensors = second_request(harness, artifact="metal/ae.safetensors", sha256=digest(harness.models / "ae.safetensors"), mmproj=None, mmproj_sha256=None)
        refused = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=tensors)
        assert refused.status_code == 422 and ".gguf" in refused.text
        assert harness.children == []
        assert api.put("/agent/admin/deployments/pictures", headers=harness.headers, json=image_request(harness)).status_code == 200
    (harness.models / "t5xxl-Q8_0.gguf").write_bytes(b"tampered encoder")
    restarted = Agent(load_agent_config(harness.config_path), harness.config_path)
    restarted.deployment_ready_timeout = 2
    restarted.start_deployments()
    assert "pictures" not in restarted.deployments
    assert "no longer matches its recorded checksum" in caplog.text
    restarted.stop()


# A single-file checkpoint carries its own encoders and autoencoder, and the
# server loads it under its full-model flag rather than as standalone
# diffusion weights.
def test_a_single_file_checkpoint_loads_as_a_full_model(harness):
    checkpoint = harness.models / "sd-v1-5.safetensors"
    checkpoint.write_bytes(b"checkpoint")
    with TestClient(build_app(harness.agent)) as api:
        request = image_request(harness, artifact="metal/sd-v1-5.safetensors", sha256=digest(checkpoint), components={}, steps=None, cfg_scale=None, sampler=None)
        created = api.put("/agent/admin/deployments/sketches", headers=harness.headers, json=request)
        assert created.status_code == 200, created.text
        command = harness.children[-1].command
        assert command[command.index("--model") + 1] == str(checkpoint) and "--diffusion-model" not in command
        assert command[command.index("--steps") + 1] == "20" and command[command.index("--cfg-scale") + 1] == "7.0"


# A transcription deployment is a whisper-server child: ggml weights, the
# OpenAI transcription path as its inference route, the language it was
# pinned with, and the multipart request forwarded as it came.
def test_a_transcription_deployment_is_a_whisper_server_child(harness):
    with TestClient(build_app(harness.agent)) as api:
        request = {
            "kind": "transcription", "artifact": "metal/ggml-small.bin", "sha256": digest(harness.ears),
            "revision": "c" * 40, "served_model_name": "assistant-transcribe", "language": "en",
        }
        created = api.put("/agent/admin/deployments/ears", headers=harness.headers, json=request)
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] == "healthy" and body["engine"] == "whisper.cpp" and body["context_length"] == 0
        command = harness.children[-1].command
        assert command[0] == "whisper-server" and command[command.index("-m") + 1] == str(harness.ears)
        assert command[command.index("--inference-path") + 1] == "/v1/audio/transcriptions"
        assert command[command.index("--language") + 1] == "en" and "--no-timestamps" in command
        assert command[command.index("--port") + 1] == "9110" and harness.children[-1].env is None

        heard = api.post(
            "/deployments/ears/v1/audio/transcriptions", headers=harness.headers,
            files={"file": ("audio.wav", b"RIFFwav", "audio/wav")}, data={"model": "assistant-transcribe"},
        )
        assert heard.status_code == 200 and heard.json()["text"] == "heard on 9110"
        port, path, _, content = harness.inference[-1]
        assert (port, path) == (9110, "/v1/audio/transcriptions") and b"RIFFwav" in content
        assert api.post("/deployments/ears/v1/chat/completions", headers=harness.headers, json={}).status_code == 404

        # A browser's own recording is not decoded by the server; the caller
        # is told what to send. An MP3 or WAV upload is forwarded as it came.
        webm = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01" + b"\x00" * 32
        refused = api.post("/deployments/ears/v1/audio/transcriptions", headers=harness.headers, files={"file": ("clip.webm", webm, "audio/webm")})
        assert refused.status_code == 415 and "WebM" in refused.text and harness.inference[-1][3] != webm
        m4a = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 24
        assert api.post("/deployments/ears/v1/audio/transcriptions", headers=harness.headers, files={"file": ("clip.m4a", m4a, "audio/mp4")}).status_code == 415
        mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb" * 16
        assert api.post("/deployments/ears/v1/audio/transcriptions", headers=harness.headers, files={"file": ("clip.mp3", mp3, "audio/mpeg")}).status_code == 200

        # GGUF is not what whisper-server loads; a language pinned on a
        # language model means nothing.
        wrong = dict(request, artifact="metal/second.gguf", sha256=digest(harness.weights))
        refused = api.put("/agent/admin/deployments/ears-two", headers=harness.headers, json=wrong)
        assert refused.status_code == 422 and ".bin" in refused.text
        assert api.put("/agent/admin/deployments/assistant-second", headers=harness.headers, json=second_request(harness, language="en")).status_code == 422

    saved = load_agent_config(harness.config_path).deployments["ears"]
    assert saved.kind == "transcription" and saved.language == "en" and saved.context_length == 0


def voice_request(harness, **overrides):
    request = {
        "kind": "speech", "artifact": "metal/en_US-ljspeech-medium.onnx", "sha256": digest(harness.voice),
        "revision": "c" * 40, "served_model_name": "assistant-speech",
        "components": {"config": {"artifact": "metal/en_US-ljspeech-medium.onnx.json", "sha256": digest(harness.models / "en_US-ljspeech-medium.onnx.json")}},
    }
    request.update(overrides)
    return request


# A speech deployment is piper's HTTP server run by the agent's own
# interpreter: the voice by path with its configuration beside it, probed on
# its voices listing, and the OpenAI speech request translated to piper's
# synthesis call with the answer returned as WAV.
def test_a_speech_deployment_is_a_piper_server_child(harness):
    with TestClient(build_app(harness.agent)) as api:
        created = api.put("/agent/admin/deployments/mouth", headers=harness.headers, json=voice_request(harness))
        assert created.status_code == 200, created.text
        assert created.json()["engine"] == "piper"
        command = harness.children[-1].command
        assert command[:3] == [sys.executable, "-m", "piper.http_server"]
        assert command[-2:] == ["-m", str(harness.voice)] and command[command.index("--port") + 1] == "9110"

        spoken = api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, json={"model": "assistant-speech", "input": "Hello there.", "voice": "alloy", "speed": 1.25})
        assert spoken.status_code == 200 and spoken.headers["content-type"] == "audio/wav" and spoken.content.startswith(b"RIFF")
        port, path, _, content = harness.inference[-1]
        assert (port, path) == (9110, "/synthesize") and json.loads(content) == {"text": "Hello there.", "length_scale": 0.8}

        # What is spoken is the visible answer: thinking and markup are set aside.
        answer = "<thought>Five words. Formulate.</thought>\n## Greeting\n\n**Hello**, how are *you* today?\n\n- one `item`\n- [two](http://x)\n\n```python\nprint(1)\n```\n"
        assert api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, json={"input": answer}).status_code == 200
        assert json.loads(harness.inference[-1][3])["text"] == "Greeting\n\nHello, how are you today?\n\none item\ntwo"
        assert api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, json={"input": "<think>still thinking"}).status_code == 400
        # A tag named in prose or code is a word; arithmetic keeps its stars.
        for spoken, heard in (
            ("Use the `<think>` tag. The answer is 42.", "Use the <think> tag. The answer is 42."),
            ("```\n<think>\n```\nThe prose after a fence stays.", "The prose after a fence stays."),
            ("2 * 3 * 4 equals 24, and a_b_c is a name.", "2 * 3 * 4 equals 24, and a_b_c is a name."),
            ("This is *really* **important**.", "This is really important."),
        ):
            assert api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, json={"input": spoken}).status_code == 200
            assert json.loads(harness.inference[-1][3])["text"] == heard, spoken
        assert api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, json={"input": "   "}).status_code == 400
        assert api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, json={"input": "x", "speed": 9}).status_code == 400
        assert api.post("/deployments/mouth/v1/audio/speech", headers=harness.headers, content=b"not json").status_code == 400
        assert api.post("/deployments/mouth/v1/images/generations", headers=harness.headers, json={}).status_code == 404

        # The configuration is the one component, and it is the file piper
        # finds by the weights' name.
        assert api.put("/agent/admin/deployments/mouth-two", headers=harness.headers, json=voice_request(harness, components={})).status_code == 422
        stray = voice_request(harness, components={"config": {"artifact": "metal/clip_l-Q8_0.gguf", "sha256": digest(harness.models / "clip_l-Q8_0.gguf")}})
        assert api.put("/agent/admin/deployments/mouth-two", headers=harness.headers, json=stray).status_code == 422

    saved = load_agent_config(harness.config_path).deployments["mouth"]
    assert saved.kind == "speech" and saved.components["config"].path == str(harness.voice) + ".json"


# A language model's thinking is the deployment's to set: off answers within
# the caller's token limit, on with a budget bounds the thinking; both reach
# the server through its environment, and neither applies to another kind.
def test_thinking_is_set_through_the_servers_environment(harness):
    with TestClient(build_app(harness.agent)) as api:
        created = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers,
                          json=second_request(harness, thinking="off"))
        assert created.status_code == 200, created.text
        child = harness.children[-1]
        assert child.env["LLAMA_ARG_REASONING"] == "off" and "LLAMA_ARG_THINK_BUDGET" not in child.env
        assert child.env["LLAMA_API_KEY"].endswith("-agent")
        assert not any(arg.startswith("--reasoning") for arg in child.command)
        listed = api.get("/agent/deployments", headers=harness.headers).json()["deployments"]
        assert listed["assistant-second"]["thinking"] == "off" and listed["assistant-second"]["thinking_budget"] is None

        budgeted = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers,
                           json=second_request(harness, thinking="on", thinking_budget=256))
        assert budgeted.status_code == 200, budgeted.text
        child = harness.children[-1]
        assert child.env["LLAMA_ARG_REASONING"] == "on" and child.env["LLAMA_ARG_THINK_BUDGET"] == "256"

        refused = api.put("/agent/admin/deployments/assistant-second", headers=harness.headers,
                          json=second_request(harness, thinking_budget=256))
        assert refused.status_code == 422 and "when thinking is on" in refused.text

        embedding = api.put("/agent/admin/deployments/embed-two", headers=harness.headers, json={
            "kind": "embedding", "artifact": "metal/second.gguf", "sha256": digest(harness.weights),
            "revision": "c" * 40, "served_model_name": "embed-two", "context_length": 2048, "thinking": "off",
        })
        assert embedding.status_code == 422 and "generation deployments only" in embedding.text
