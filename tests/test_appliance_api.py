"""Appliance integration tests over the fake backend: the full contract
surface without vLLM installed."""

import asyncio
import json
import math
from contextlib import asynccontextmanager

import httpx

import pytest
import yaml
from fastapi.testclient import TestClient

from lazarus.appliance.api import Admission
from lazarus.appliance.backends.fake import FakeBackend
from lazarus.appliance.launcher import Appliance


def make_appliance(config_path, monkeypatch, **env) -> Appliance:
    for key in (
        "SOVEREIGN_RUNTIME_API_KEY",
        "SOVEREIGN_FAKE_FAIL_ROLE",
        "SOVEREIGN_RUNTIME_MANIFEST",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("SOVEREIGN_PROFILE", "mock")
    appliance = Appliance(config_path=str(config_path), backend=FakeBackend())
    asyncio.run(appliance.run_lifecycle())
    return appliance


@pytest.fixture()
def healthy(config_file, monkeypatch) -> TestClient:
    appliance = make_appliance(config_file, monkeypatch)
    return TestClient(appliance.app)


def test_healthy_lifecycle(healthy):
    assert healthy.get("/health/live").json() == {"status": "alive", "state": "healthy"}
    ready = healthy.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    health = healthy.get("/health").json()
    assert health["roles"]["generation"]["status"] == "healthy"
    assert health["roles"]["embedding"]["status"] == "healthy"


def test_manifest_reports_discovered_dimensions(healthy):
    manifest = healthy.get("/runtime/manifest").json()
    assert manifest["schema_version"] == "1.3"
    assert manifest["topology"] == "single_process_multi_role"
    assert manifest["state"] == "healthy"
    assert manifest["roles"]["embedding"]["dimensions"] == 384
    assert manifest["roles"]["embedding"]["normalization"] == "l2"
    assert manifest["profile"] == "mock"
    assert manifest["generation_paused"] is False


@pytest.mark.parametrize("count", [1, 2, 4])
def test_manifest_reports_exact_managed_multi_gpu_execution(
    managed_cuda_config_file, monkeypatch, count
):
    import yaml

    data = yaml.safe_load(managed_cuda_config_file.read_text())
    devices = [f"GPU-00000000-0000-0000-0000-{rank:012x}" for rank in range(count, 0, -1)]
    data["roles"]["generation"]["accelerator_device_ids"] = devices
    data["roles"]["generation"]["tensor_parallel_size"] = count
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    manifest = (
        TestClient(make_appliance(managed_cuda_config_file, monkeypatch).app)
        .get("/runtime/manifest")
        .json()
    )
    assert manifest["state"] == "healthy"
    generation = manifest["roles"]["generation"]
    assert generation["device_count"] == count
    assert generation["tensor_parallel_size"] == count
    assert [item["gpu_uuid"] for item in manifest["accelerator"]["devices"]] == devices
    assert [item["local_rank"] for item in manifest["accelerator"]["devices"]] == list(range(count))


def test_models_chat_and_embeddings(healthy):
    ids = {m["id"] for m in healthy.get("/v1/models").json()["data"]}
    assert ids == {"assistant-dev", "embedding-custom"}

    chat = healthy.post(
        "/v1/chat/completions",
        json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"]

    emb = healthy.post("/v1/embeddings", json={"model": "embedding-custom", "input": "hello"})
    vector = emb.json()["data"][0]["embedding"]
    assert len(vector) == 384
    assert abs(math.sqrt(sum(v * v for v in vector)) - 1.0) < 1e-6


def test_text_generation_rejects_multimodal_content_and_recovers(healthy):
    rejected = healthy.post(
        "/v1/chat/completions",
        json={
            "model": "assistant-dev",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                    ],
                }
            ],
        },
    )
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "unsupported_modality"
    recovered = healthy.post(
        "/v1/chat/completions",
        json={
            "model": "assistant-dev",
            "messages": [{"role": "user", "content": "Reply with OK."}],
        },
    )
    assert recovered.status_code == 200


def test_multimodal_embeddings_messages_schema(healthy):
    # Extended schema: `messages` replaces `input` (runtime-contract §embeddings).
    body = {
        "model": "embedding-custom",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                    {"type": "text", "text": "caption"},
                ],
            }
        ],
    }
    emb = healthy.post("/v1/embeddings", json=body)
    assert emb.status_code == 200
    assert emb.json()["data"][0]["embedding"]

    neither = healthy.post("/v1/embeddings", json={"model": "embedding-custom"})
    assert neither.status_code == 400


def test_multimodal_embeddings_reject_remote_urls(healthy):
    # Sovereignty: the runtime never fetches media; only data: URIs pass.
    body = {
        "model": "embedding-custom",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
    }
    emb = healthy.post("/v1/embeddings", json=body)
    assert emb.status_code == 400
    assert "data URIs" in emb.json()["error"]["message"]


def test_streaming_ends_with_done(healthy):
    with healthy.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "assistant-dev",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        },
    ) as resp:
        text = "".join(resp.iter_text())
    assert "data: " in text and text.rstrip().endswith("data: [DONE]")


def test_role_mismatch_404(healthy):
    resp = healthy.post(
        "/v1/chat/completions",
        json={"model": "embedding-custom", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404


def test_auth_enforced(config_file, monkeypatch):
    appliance = make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret")
    client = TestClient(appliance.app)
    denied = client.post(
        "/v1/chat/completions",
        json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert denied.status_code == 401
    allowed = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer secret"},
        json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert allowed.status_code == 200
    # health endpoints stay unauthenticated
    assert client.get("/health/live").status_code == 200


def test_config_error_stays_alive(tmp_path, monkeypatch):
    appliance = make_appliance(tmp_path / "missing.yaml", monkeypatch)
    client = TestClient(appliance.app)
    assert appliance.state.state == "configuration_error"
    assert client.get("/health/live").status_code == 200
    ready = client.get("/health/ready")
    assert ready.status_code == 503 and ready.json()["ready"] is False
    errors = client.get("/runtime/errors").json()["errors"]
    assert errors and errors[0]["code"] == "CONFIG_INVALID"
    # manifest still serves and is shape-complete
    manifest = client.get("/runtime/manifest").json()
    assert manifest["state"] == "configuration_error"
    assert "generation" in manifest["roles"]


def test_degraded_embedding(config_file, monkeypatch):
    appliance = make_appliance(config_file, monkeypatch, SOVEREIGN_FAKE_FAIL_ROLE="embedding")
    client = TestClient(appliance.app)
    assert appliance.state.state == "degraded"
    health = client.get("/health").json()
    assert health["roles"]["generation"]["status"] == "healthy"
    assert health["roles"]["embedding"]["status"] == "unhealthy"
    assert client.get("/health/ready").status_code == 503
    codes = {e["code"] for e in client.get("/runtime/errors").json()["errors"]}
    assert "MODEL_LOAD_FAILED" in codes
    # generation still serves in degraded state? No — not ready. 503 expected.
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 503


def test_generation_only_runtime_is_ready(config_file, tmp_path, monkeypatch):
    import yaml

    data = yaml.safe_load(config_file.read_text())
    del data["roles"]["embedding"]
    path = tmp_path / "generation-only.yaml"
    path.write_text(yaml.safe_dump(data))
    client = TestClient(make_appliance(path, monkeypatch).app)

    ready = client.get("/health/ready")
    assert ready.status_code == 200
    assert ready.json()["required_roles"] == {"generation": True}
    assert "embedding" not in client.get("/runtime/manifest").json()["roles"]
    assert (
        client.post(
            "/v1/chat/completions",
            json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}]},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/embeddings", json={"model": "embedding-custom", "input": "hello"}
        ).status_code
        == 404
    )


def test_manifest_written_to_file(config_file, tmp_path, monkeypatch):
    target = tmp_path / "state" / "manifest.json"
    make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_MANIFEST=str(target))
    assert target.is_file()


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
@pytest.mark.parametrize("configured_key", [None, "secret"])
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "Basic secret"])
def test_managed_generation_requires_configured_bearer_key(
    config_file, monkeypatch, operation, configured_key, authorization
):
    env = {"SOVEREIGN_RUNTIME_API_KEY": configured_key} if configured_key else {}
    appliance = make_appliance(config_file, monkeypatch, **env)
    client = TestClient(appliance.app)
    headers = {"Authorization": authorization} if authorization else {}
    denied = client.post(f"/runtime/admin/generation/{operation}", headers=headers)
    assert denied.status_code == 401
    assert appliance.backend.generation_paused is False
    assert client.get("/health/ready").status_code == 200
    if configured_key is None:
        assert client.post(
            f"/runtime/admin/generation/{operation}",
            headers={"Authorization": "Bearer secret"},
        ).status_code == 401


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
@pytest.mark.parametrize("body", [b"{}", b"null", b" ", b"mode=abort", b'{"path":"/pause"}'])
def test_managed_generation_rejects_every_nonempty_body(config_file, monkeypatch, operation, body):
    appliance = make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret")
    client = TestClient(appliance.app)
    response = client.post(
        f"/runtime/admin/generation/{operation}", content=body,
        headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "request body is not allowed"
    assert appliance.backend.generation_paused is False


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
def test_legacy_backend_cannot_acknowledge_managed_generation(config_file, monkeypatch, operation):
    appliance = make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret")
    client = TestClient(appliance.app)
    response = client.post(
        f"/runtime/admin/generation/{operation}", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == f"ENGINE_{operation.upper()}_FAILED"
    assert appliance.backend.generation_paused is True
    assert client.get("/health/ready").status_code == 503
    assert client.get("/health").json()["roles"]["generation"]["status"] == "healthy"


def test_managed_generation_waits_for_pause_and_resume_ack(config_file, tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    appliance = make_appliance(
        config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret",
        SOVEREIGN_RUNTIME_MANIFEST=str(target),
    )
    backend, runtime_config = appliance.backend, appliance.config
    generation = backend.role_info("generation")
    config_snapshot = runtime_config.model_dump()

    async def exercise():
        pause_called, pause_ack = asyncio.Event(), asyncio.Event()
        resume_called, resume_ack = asyncio.Event(), asyncio.Event()

        async def quiesce():
            assert appliance.backend.generation_paused is True
            pause_called.set()
            await pause_ack.wait()

        async def resume():
            assert appliance.backend.generation_paused is True
            resume_called.set()
            await resume_ack.wait()
            appliance.backend.generation_paused = False

        monkeypatch.setattr(appliance.backend, "quiesce", quiesce)
        monkeypatch.setattr(appliance.backend, "resume", resume)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=appliance.app), base_url="http://runtime",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is False
            assert json.loads(target.read_text())["generation_paused"] is False
            pending = asyncio.create_task(client.post("/runtime/admin/generation/quiesce"))
            await asyncio.wait_for(pause_called.wait(), 2)
            assert not pending.done(), "API admission idle is not an engine ACK"
            assert (await client.get("/health/ready")).json()["ready"] is False
            health = (await client.get("/health")).json()
            assert health["state"] == "healthy"
            assert health["roles"]["generation"]["model_loaded"] is True
            manifest = (await client.get("/runtime/manifest")).json()
            assert manifest["state"] == "healthy"
            assert manifest["generation_paused"] is True
            assert json.loads(target.read_text())["generation_paused"] is True
            assert (await client.post(
                "/v1/chat/completions", json={"model": "assistant-dev", "messages": []}
            )).status_code == 503
            assert (await client.post(
                "/v1/completions", json={"model": "assistant-dev", "prompt": "hello"}
            )).status_code == 503
            assert (await client.post(
                "/v1/embeddings", json={"model": "embedding-custom", "input": "hello"}
            )).status_code == 200
            pause_ack.set()
            assert (await asyncio.wait_for(pending, 2)).json() == {"quiesced": True}
            assert (await client.get("/health/ready")).status_code == 503
            assert json.loads(target.read_text())["generation_paused"] is True
            pending_resume = asyncio.create_task(client.post("/runtime/admin/generation/resume"))
            await asyncio.wait_for(resume_called.wait(), 2)
            assert not pending_resume.done()
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
            assert json.loads(target.read_text())["generation_paused"] is True
            resume_ack.set()
            assert (await asyncio.wait_for(pending_resume, 2)).json() == {"resumed": True}
            assert (await client.get("/health/ready")).status_code == 200
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is False
            assert json.loads(target.read_text())["generation_paused"] is False
            chat = await client.post("/v1/chat/completions", json={
                "model": "assistant-dev", "messages": [{"role": "user", "content": "hello"}],
            })
            assert chat.status_code == 200
            assert chat.json()["choices"][0]["message"]["content"]
            completion = await client.post("/v1/completions", json={
                "model": "assistant-dev", "prompt": "hello",
            })
            assert completion.status_code == 200
            assert completion.json()["choices"][0]["text"]
            assert appliance.backend is backend
            assert appliance.config is runtime_config
            assert backend.role_info("generation") is generation
            assert runtime_config.model_dump() == config_snapshot

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
def test_managed_engine_failure_is_bounded_and_keeps_admission_closed(config_file, tmp_path, monkeypatch, operation):
    target = tmp_path / "manifest.json"
    appliance = make_appliance(
        config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret",
        SOVEREIGN_RUNTIME_MANIFEST=str(target),
    )

    async def fail():
        appliance.backend.generation_paused = False
        raise RuntimeError("private engine path /weights/secret token=private")

    monkeypatch.setattr(appliance.backend, operation, fail)
    client = TestClient(appliance.app)
    response = client.post(
        f"/runtime/admin/generation/{operation}", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 503
    assert response.json() == {"error": {
        "message": f"engine {operation} was not acknowledged", "type": "server_error",
        "code": f"ENGINE_{operation.upper()}_FAILED",
    }}
    assert appliance.backend.generation_paused is True
    assert client.get("/health/ready").status_code == 503
    assert client.get("/runtime/manifest").json()["generation_paused"] is True
    assert json.loads(target.read_text())["generation_paused"] is True


def test_managed_resume_requires_explicit_backend_ack(config_file, monkeypatch):
    appliance = make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret")
    appliance.backend.generation_paused = True

    async def unacknowledged():
        pass

    monkeypatch.setattr(appliance.backend, "resume", unacknowledged)
    client = TestClient(appliance.app)
    response = client.post(
        "/runtime/admin/generation/resume", headers={"Authorization": "Bearer secret"}
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ENGINE_RESUME_FAILED"
    assert client.get("/health/ready").status_code == 503


@pytest.mark.parametrize("forwarded", [False, True])
@pytest.mark.parametrize("cancel_quiesce", [False, True])
def test_quiesce_drains_streams_and_rechecks_waiting_admission(
    config_file, monkeypatch, forwarded, cancel_quiesce
):
    data = yaml.safe_load(config_file.read_text())
    data["roles"]["generation"]["max_concurrent_requests"] = 1
    config_file.write_text(yaml.safe_dump(data))
    appliance = make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret")

    async def exercise():
        started, finish = asyncio.Event(), asyncio.Event()
        queued, draining = asyncio.Event(), asyncio.Event()
        pause_called, engine_ack = asyncio.Event(), asyncio.Event()
        events = []
        arrivals = 0
        original_slot = Admission.slot
        original_idle = Admission.wait_generation_idle

        @asynccontextmanager
        async def observed_slot(self, role):
            nonlocal arrivals
            if role == "generation":
                arrivals += 1
                if arrivals == 2:
                    queued.set()
            async with original_slot(self, role):
                yield

        async def observed_idle(self):
            draining.set()
            await original_idle(self)

        async def chunks(_body=None):
            events.append("forward")
            yield "data: first\n\n"
            started.set()
            await finish.wait()
            events.append("finished")
            yield "data: [DONE]\n\n"

        class EngineBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                async for chunk in chunks():
                    yield chunk.encode()

            async def aclose(self):
                events.append("closed")

        async def quiesce():
            events.append("pause")
            pause_called.set()
            await engine_ack.wait()

        monkeypatch.setattr(Admission, "slot", observed_slot)
        monkeypatch.setattr(Admission, "wait_generation_idle", observed_idle)
        monkeypatch.setattr(appliance.backend, "quiesce", quiesce)
        monkeypatch.setattr(appliance.backend, "chat_completion_stream", chunks)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(
                200, stream=EngineBody(), headers={"content-type": "text/event-stream"}
            )), base_url="http://engine",
        ) as engine_client:
            if forwarded:
                monkeypatch.setattr(appliance.backend, "role_client", lambda role: engine_client, raising=False)
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=appliance.app), base_url="http://runtime",
                headers={"Authorization": "Bearer secret"},
            ) as client:
                body = {"model": "assistant-dev", "messages": [], "stream": True}
                first = asyncio.create_task(client.post("/v1/chat/completions", json=body))
                await asyncio.wait_for(started.wait(), 2)
                second = asyncio.create_task(client.post("/v1/chat/completions", json=body))
                await asyncio.wait_for(queued.wait(), 2)
                pending = asyncio.create_task(client.post("/runtime/admin/generation/quiesce"))
                await asyncio.wait_for(draining.wait(), 2)
                assert not pause_called.is_set(), "pause(wait) would freeze admitted queued work"
                assert not pending.done()
                assert (await client.get("/health/ready")).status_code == 503
                if cancel_quiesce:
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert not first.done(), "cancelling the admin wait must not cut generation"
                finish.set()
                first_response = await asyncio.wait_for(first, 2)
                assert first_response.status_code == 200
                assert "data: [DONE]" in first_response.text
                assert (await asyncio.wait_for(second, 2)).status_code == 503
                assert events.count("forward") == 1
                if cancel_quiesce:
                    assert not pause_called.is_set()
                else:
                    await asyncio.wait_for(pause_called.wait(), 2)
                    assert events.index("finished") < events.index("pause")
                    if forwarded:
                        assert events.index("closed") < events.index("pause")
                    assert not pending.done(), "completed HTTP responses are not an engine ACK"
                    engine_ack.set()
                    assert (await asyncio.wait_for(pending, 2)).json() == {"quiesced": True}
                assert appliance.backend.generation_paused is True
                assert (await client.get("/health/ready")).status_code == 503

    asyncio.run(exercise())


def test_cancelling_engine_ack_wait_never_reopens_generation(config_file, tmp_path, monkeypatch):
    target = tmp_path / "manifest.json"
    appliance = make_appliance(
        config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret",
        SOVEREIGN_RUNTIME_MANIFEST=str(target),
    )

    async def exercise():
        entered = asyncio.Event()

        async def quiesce():
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(appliance.backend, "quiesce", quiesce)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=appliance.app), base_url="http://runtime",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            pending = asyncio.create_task(client.post("/runtime/admin/generation/quiesce"))
            await asyncio.wait_for(entered.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert appliance.backend.generation_paused is True
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
            assert json.loads(target.read_text())["generation_paused"] is True

    asyncio.run(exercise())


@pytest.mark.parametrize("failed_gate", [True, False])
def test_failed_gate_publication_cannot_acknowledge_resume(config_file, tmp_path, monkeypatch, failed_gate):
    target = tmp_path / "manifest.json"
    appliance = make_appliance(
        config_file, monkeypatch, SOVEREIGN_RUNTIME_API_KEY="secret",
        SOVEREIGN_RUNTIME_MANIFEST=str(target),
    )
    publish = appliance.manifest.write
    failed = False

    def unavailable_once():
        nonlocal failed
        if not failed and appliance.backend.generation_paused is failed_gate:
            failed = True
            raise OSError("manifest storage unavailable")
        publish()

    async def resume():
        appliance.backend.generation_paused = False

    monkeypatch.setattr(appliance.manifest, "write", unavailable_once)
    monkeypatch.setattr(appliance.backend, "resume", resume)
    client = TestClient(appliance.app)
    response = client.post(
        "/runtime/admin/generation/resume", headers={"Authorization": "Bearer secret"},
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ENGINE_RESUME_FAILED"
    assert appliance.backend.generation_paused is True
    assert client.get("/runtime/manifest").json()["generation_paused"] is True
    assert json.loads(target.read_text())["generation_paused"] is True
    assert client.get("/health/ready").status_code == 503
