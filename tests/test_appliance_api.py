"""Appliance integration tests over the fake backend: the full contract
surface without vLLM installed."""

import asyncio
import math

import pytest
from fastapi.testclient import TestClient

from lazarus.appliance.backends.fake import FakeBackend
from lazarus.appliance.launcher import Appliance


def make_appliance(config_path, monkeypatch, **env) -> Appliance:
    for key in ("SOVEREIGN_RUNTIME_API_KEY", "SOVEREIGN_FAKE_FAIL_ROLE", "SOVEREIGN_RUNTIME_MANIFEST"):
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
    assert manifest["topology"] == "single_process_multi_role"
    assert manifest["state"] == "healthy"
    assert manifest["roles"]["embedding"]["dimensions"] == 384
    assert manifest["roles"]["embedding"]["normalization"] == "l2"
    assert manifest["profile"] == "mock"


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


def test_multimodal_embeddings_messages_schema(healthy):
    # Extended schema: `messages` replaces `input` (runtime-contract §embeddings).
    body = {
            "model": "embedding-custom",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
                {"type": "text", "text": "caption"},
            ]}
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
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ]}
        ],
    }
    emb = healthy.post("/v1/embeddings", json=body)
    assert emb.status_code == 400
    assert "data URIs" in emb.json()["error"]["message"]


def test_streaming_ends_with_done(healthy):
    with healthy.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}], "stream": True},
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
    assert client.post(
        "/v1/chat/completions",
        json={"model": "assistant-dev", "messages": [{"role": "user", "content": "hi"}]},
    ).status_code == 200
    assert client.post(
        "/v1/embeddings", json={"model": "embedding-custom", "input": "hello"}
    ).status_code == 404


def test_manifest_written_to_file(config_file, tmp_path, monkeypatch):
    target = tmp_path / "state" / "manifest.json"
    make_appliance(config_file, monkeypatch, SOVEREIGN_RUNTIME_MANIFEST=str(target))
    assert target.is_file()
