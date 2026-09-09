"""Supervisor regressions with subprocess/HTTP boundaries, not real engine execution."""

import asyncio
import json
import os
import signal
import sys
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest
import yaml
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from lazarus.appliance.backends import select_backend
from lazarus.appliance.backends.agent import AgentBackend
from lazarus.appliance.backends.base import BackendStartError
from lazarus.appliance.backends.fake import FakeBackend
from lazarus.appliance.backends.slimserve import SlimServeBackend, _stop_group, child_environment, validate_observation
from lazarus.appliance.backends.slimserve_agent import SlimServeAgentBackend
from lazarus.appliance.backends.vllm_engine import VllmBackend
from lazarus.appliance.config import (
    QUIXICORE_CUDA_COMMIT,
    QUIXICORE_METAL_COMMIT,
    SLIMSERVE_COMMIT,
    ConfigError,
    RuntimeConfig,
    SlimServeFile,
    load_config,
)
from lazarus.appliance.launcher import Appliance

UUIDS = [f"GPU-00000000-0000-0000-0000-{index:012d}" for index in (2, 1, 4, 3)]
APPLE_ID = "apple-platform-integrated-gpu-v1:" + "a" * 64
EXECUTION_FIELDS = (
    "engine_model", "revision", "context_length", "device_count", "tensor_parallel_size",
    "engine_profile_id", "upstream_profile_id", "quant", "max_concurrent_requests", "capabilities",
)


@pytest.fixture(autouse=True)
def isolated_runtime_environment(monkeypatch):
    for name in (
        "SOVEREIGN_ENGINE_BACKEND", "SOVEREIGN_RUNTIME_API_KEY",
        "SOVEREIGN_RUNTIME_MANIFEST", "SOVEREIGN_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def slimserve_document():
    return {
        "schema_version": "1.2",
        "runtime": {"profile": "cuda-x86_64", "api_key_env": "SOVEREIGN_RUNTIME_API_KEY"},
        "roles": {"generation": {
            "enabled": True, "engine": "slimserve", "engine_profile_id": "managed-a100",
            "task": "generate", "source": "local",
            "model": "/models/staged/artifact/" + "a" * 64 + "/managed-a100/model",
            "revision": "b" * 40, "served_model_name": "assistant-large",
            "max_model_len": 2048, "max_concurrent_requests": 1,
            "accelerator_device_ids": UUIDS[:2], "tensor_parallel_size": 2,
            "slimserve": {
                "source_commit": SLIMSERVE_COMMIT, "profile_id": "upstream-a100",
                "variant": "a100", "quant": "bf16",
                "artifacts": [{
                    "role": "model", "repository": "owner/model", "revision": "b" * 40,
                    "files": [{"file": "weights.safetensors", "size_bytes": 6, "sha256": "c" * 64}],
                }],
            },
        }},
    }


@pytest.fixture
def slimserve_config(slimserve_document):
    return RuntimeConfig.model_validate(slimserve_document)


def _write_config(tmp_path, document):
    path = tmp_path / "runtime.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


def _replace(document, path, value):
    target = document
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value


def _metal_config(document):
    document = deepcopy(document)
    document["runtime"]["profile"] = "metal-arm64"
    role = document["roles"]["generation"]
    role.pop("accelerator_device_ids")
    role.pop("tensor_parallel_size")
    role["slimserve"]["variant"] = "metal"
    return RuntimeConfig.model_validate(document)


def _observation(config):
    role = config.roles.generation
    metal = role.slimserve.variant == "metal"
    backend = "metal" if metal else "cuda"
    commit = QUIXICORE_METAL_COMMIT if metal else QUIXICORE_CUDA_COMMIT
    devices = [{
        "identity_kind": "apple_platform", "stable_identifier": APPLE_ID, "platform_id": APPLE_ID,
    }] if metal else [{
        "identity_kind": "nvidia_gpu_uuid", "local_rank": rank,
        "stable_identifier": uuid, "gpu_uuid": uuid,
    } for rank, uuid in enumerate(role.accelerator_device_ids)]
    return {
        "engine": {"name": "slimserve", "version": SLIMSERVE_COMMIT, "adapter": "slimserve-runtime"},
        "kernels": {"library": f"quixicore-{backend}", "backend": backend,
                    "version": f"{commit}+slimserve.{SLIMSERVE_COMMIT}"},
        "engine_model": role.model, "revision": role.revision,
        "engine_profile_id": role.engine_profile_id, "upstream_profile_id": role.slimserve.profile_id,
        "quant": role.slimserve.quant, "context_length": role.max_model_len,
        "max_concurrent_requests": role.max_concurrent_requests,
        "device_count": role.tensor_parallel_size, "tensor_parallel_size": role.tensor_parallel_size,
        "capabilities": ["chat_completions", "completions", "streaming", "text"],
        "accelerator": {"vendor": "apple" if metal else "nvidia", "unified_memory": metal,
                        "device_count": role.tensor_parallel_size, "devices": devices},
    }


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("context,concurrency", [(1, 1), (1048576, 256)])
def test_typed_cuda_config_accepts_supported_bounds(slimserve_document, tmp_path, count, context, concurrency):
    slimserve_document["roles"]["generation"].update(
        accelerator_device_ids=UUIDS[:count], tensor_parallel_size=count,
        max_model_len=context, max_concurrent_requests=concurrency,
    )
    config = load_config(_write_config(tmp_path, slimserve_document))
    assert config.roles.generation.engine == "slimserve"
    assert config.roles.generation.tensor_parallel_size == count
    assert config.roles.generation.max_model_len == context
    assert config.roles.generation.max_concurrent_requests == concurrency


@pytest.mark.parametrize("field,value", [
    ("engine", "unknown"), ("engine", "SlimServe"), ("source", "huggingface"),
    ("model", "relative/model"), ("revision", "main"), ("engine_profile_id", None),
    ("max_model_len", None), ("max_model_len", 0), ("max_model_len", 1048577),
    ("max_concurrent_requests", 0), ("max_concurrent_requests", 257),
    ("served_model_name", "../private"), ("enabled", False),
    ("tensor_parallel_size", True), ("tensor_parallel_size", "2"), ("tensor_parallel_size", 3),
    ("accelerator_device_ids", [UUIDS[0], UUIDS[0]]),
    ("accelerator_device_ids", ["0", "1"]), ("accelerator_device_ids", UUIDS[:1]),
    ("extra_args", ["--trust-remote-code"]), ("env", {"VLLM_PLUGINS": "unapproved"}),
    ("interpreter", "/tmp/python"), ("command", ["other-server"]),
])
def test_typed_config_rejects_unsealed_generation_inputs(slimserve_document, tmp_path, field, value):
    slimserve_document["roles"]["generation"][field] = value
    with pytest.raises(ConfigError):
        load_config(_write_config(tmp_path, slimserve_document))


def test_metal_config_rejects_explicit_empty_accelerator_placement(slimserve_document, tmp_path):
    document = _metal_config(slimserve_document).model_dump(exclude_none=True, exclude_unset=True)
    document["roles"]["generation"]["accelerator_device_ids"] = []
    with pytest.raises(ConfigError, match=r"roles\.generation\.accelerator_device_ids"):
        load_config(_write_config(tmp_path, document))


@pytest.mark.parametrize("field,value", [
    ("source_commit", "d" * 40), ("profile_id", "../profile"), ("variant", "cpu"),
    ("quant", "--arbitrary"), ("flags", ["--download"]), ("env", {"HF_TOKEN": "private"}),
])
def test_typed_slimserve_config_rejects_unknown_policy(slimserve_document, field, value):
    slimserve_document["roles"]["generation"]["slimserve"][field] = value
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(slimserve_document)


@pytest.mark.parametrize("member", [
    "../weights.gguf", "/weights.gguf", "dir/../weights.gguf", "dir//weights.gguf",
    "dir/.hidden", "dir/./weights.gguf", "dir/weights\\file.gguf", "https://host/weights.gguf",
    "a/b/c/d/weights.gguf", "weights.gguf?download=1", "weights.gguf\n", "a" * 193,
])
def test_artifact_member_rejects_unsafe_paths(member):
    with pytest.raises(ValidationError):
        SlimServeFile(file=member, size_bytes=1, sha256="c" * 64)


@pytest.mark.parametrize("size", [True, "1", 0, -1, 1024**4 + 1])
def test_artifact_size_is_a_bounded_strict_integer(size):
    with pytest.raises(ValidationError):
        SlimServeFile(file="model.gguf", size_bytes=size, sha256="c" * 64)


@pytest.mark.parametrize("kind", ["duplicate-file", "duplicate-role", "missing-model", "oversize-closure"])
def test_typed_artifact_closure_rejects_ambiguity_and_overflow(slimserve_document, kind):
    artifacts = slimserve_document["roles"]["generation"]["slimserve"]["artifacts"]
    if kind == "duplicate-file":
        artifacts[0]["files"] *= 2
    elif kind == "duplicate-role":
        artifacts.append(deepcopy(artifacts[0]))
    elif kind == "missing-model":
        artifacts[0]["role"] = "drafter"
    else:
        artifacts[0]["files"] = [
            {"file": f"part-{index}.safetensors", "size_bytes": 1024**4, "sha256": "c" * 64}
            for index in range(3)
        ]
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(slimserve_document)


@pytest.mark.parametrize("name", ["embedding", "vision", "audio", "rerank"])
def test_slimserve_rejects_secondary_runtime_roles(slimserve_document, name):
    slimserve_document["roles"][name] = {"enabled": True}
    with pytest.raises(ValidationError):
        RuntimeConfig.model_validate(slimserve_document)


@pytest.mark.parametrize("selection,expected", [
    (None, SlimServeBackend), ("vllm", SlimServeBackend), ("fake", SlimServeBackend),
    ("slimserve", SlimServeBackend), ("agent", SlimServeAgentBackend),
])
def test_explicit_slimserve_selection_wins_over_legacy_environment(slimserve_config, monkeypatch, selection, expected):
    if selection is not None:
        monkeypatch.setenv("SOVEREIGN_ENGINE_BACKEND", selection)
    assert type(select_backend(slimserve_config)) is expected


@pytest.mark.parametrize("selection,expected", [
    (None, VllmBackend), ("vllm", VllmBackend), ("fake", FakeBackend), ("agent", AgentBackend),
])
@pytest.mark.parametrize("with_config", [False, True])
def test_legacy_selection_is_unchanged(config_file, monkeypatch, selection, expected, with_config):
    if selection is not None:
        monkeypatch.setenv("SOVEREIGN_ENGINE_BACKEND", selection)
    assert type(select_backend(load_config(config_file) if with_config else None)) is expected


def test_environment_alone_cannot_select_slimserve(monkeypatch):
    monkeypatch.setenv("SOVEREIGN_ENGINE_BACKEND", "slimserve")
    with pytest.raises(ValueError):
        select_backend()


@pytest.mark.parametrize("count", [1, 2, 4])
def test_observation_accepts_exact_cuda_workers_and_detaches_normalized_ranks(slimserve_document, count):
    slimserve_document["roles"]["generation"].update(
        accelerator_device_ids=UUIDS[:count], tensor_parallel_size=count,
    )
    config = RuntimeConfig.model_validate(slimserve_document)
    payload = _observation(config)
    payload["accelerator"]["devices"].reverse()
    accepted = validate_observation(config, payload)
    assert [device["gpu_uuid"] for device in accepted["accelerator"]["devices"]] == UUIDS[:count]
    assert [device["local_rank"] for device in accepted["accelerator"]["devices"]] == list(range(count))
    assert payload["accelerator"]["devices"][0]["local_rank"] == count - 1
    payload["accelerator"]["devices"][0]["gpu_uuid"] = "changed"
    payload["capabilities"].clear()
    assert all(device["gpu_uuid"] != "changed" for device in accepted["accelerator"]["devices"])
    assert accepted["capabilities"] == ["chat_completions", "completions", "streaming", "text"]


def test_observation_accepts_native_apple_identity(slimserve_document, tmp_path):
    config = _metal_config(slimserve_document)
    loaded = load_config(_write_config(tmp_path, config.model_dump(exclude_unset=True)))
    payload = _observation(loaded)
    assert validate_observation(loaded, payload) == payload
    assert payload["accelerator"]["devices"][0]["stable_identifier"] == APPLE_ID


@pytest.mark.parametrize("key", [
    "engine", "kernels", "engine_model", "engine_profile_id", "upstream_profile_id", "quant",
    "revision", "context_length", "max_concurrent_requests", "device_count", "tensor_parallel_size",
    "capabilities", "accelerator",
])
def test_observation_rejects_missing_execution_evidence(slimserve_config, key):
    payload = _observation(slimserve_config)
    del payload[key]
    with pytest.raises(BackendStartError):
        validate_observation(slimserve_config, payload)


@pytest.mark.parametrize("path,value", [
    (("engine", "name"), "vllm"), (("engine", "version"), "d" * 40),
    (("engine", "adapter"), "other-adapter"), (("kernels", "library"), "quixicore-metal"),
    (("kernels", "version"), "unobserved"), (("kernels", "backend"), "metal"),
    (("engine_profile_id",), "other-profile"), (("upstream_profile_id",), "other-upstream"),
    (("quant",), "int4"), (("revision",), "d" * 40),
    (("engine_model",), "/private/other-model"), (("engine_model",), []),
    (("context_length",), 1024), (("max_concurrent_requests",), 2),
    (("device_count",), 1), (("tensor_parallel_size",), 1),
    (("accelerator", "vendor"), "apple"), (("accelerator", "unified_memory"), 0),
    (("accelerator", "device_count"), 1), (("accelerator", "devices"), []),
    (("accelerator", "devices", 0, "gpu_uuid"), UUIDS[1]),
    (("accelerator", "devices", 0, "stable_identifier"), UUIDS[1]),
    (("accelerator", "devices", 0, "identity_kind"), "ordinal"),
    (("accelerator", "devices", 0, "local_rank"), 1),
    (("accelerator", "devices", 0, "local_rank"), False),
    (("accelerator", "devices", 0, "local_rank"), "0"),
    (("capabilities",), ["text"]),
    (("capabilities",), ["chat_completions", "completions", "streaming", "streaming"]),
    (("capabilities",), ["chat_completions", "completions", "streaming", {}]),
])
def test_observation_rejects_drift_and_malformed_values(slimserve_config, path, value):
    payload = _observation(slimserve_config)
    _replace(payload, path, value)
    with pytest.raises(BackendStartError):
        validate_observation(slimserve_config, payload)


@pytest.mark.parametrize("path", [
    ("context_length",), ("max_concurrent_requests",), ("device_count",),
    ("tensor_parallel_size",), ("accelerator", "device_count"),
])
@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_observation_cardinality_never_coerces_equal_nonintegers(slimserve_document, path, value):
    slimserve_document["roles"]["generation"].update(
        accelerator_device_ids=UUIDS[:1], tensor_parallel_size=1, max_model_len=1,
    )
    config = RuntimeConfig.model_validate(slimserve_document)
    payload = _observation(config)
    _replace(payload, path, value)
    with pytest.raises(BackendStartError):
        validate_observation(config, payload)


@pytest.mark.parametrize("field,value", [
    ("identity_kind", "nvidia_gpu_uuid"), ("stable_identifier", "apple-gpu-0"),
    ("platform_id", "apple-platform-integrated-gpu-v1:" + "b" * 64),
    ("stable_identifier", "apple-platform-integrated-gpu-v1:" + "A" * 64),
])
def test_observation_rejects_unproven_native_identity(slimserve_document, field, value):
    config = _metal_config(slimserve_document)
    payload = _observation(config)
    payload["accelerator"]["devices"][0][field] = value
    with pytest.raises(BackendStartError):
        validate_observation(config, payload)


@pytest.mark.parametrize("observed", ["model.gguf", "other.gguf", "directory"])
def test_gguf_observation_must_name_a_verified_member(slimserve_document, observed):
    slimserve_document["roles"]["generation"]["slimserve"]["artifacts"][0]["files"][0]["file"] = "model.gguf"
    config = RuntimeConfig.model_validate(slimserve_document)
    payload = _observation(config)
    if observed != "directory":
        payload["engine_model"] += "/" + observed
    if observed == "model.gguf":
        assert validate_observation(config, payload)["engine_model"] == payload["engine_model"]
    else:
        with pytest.raises(BackendStartError):
            validate_observation(config, payload)


@pytest.fixture
def boundaries(monkeypatch, slimserve_config):
    """Model subprocesses and private HTTP; keep every first-party lifecycle method real."""
    control = SimpleNamespace(
        calls=[], events=[], processes=[], requests=[], client_options=[], clients=[],
        check_returncode=0, check_error=None, check_content=None, spawn_error=None, hold_term=False,
        resolved_model=None, chat_status=200,
        drain_error=None, stop_error=None, group_probe_error=None, descendants_survive=False,
        chat_body={"model": "assistant-large", "choices": [{"message": {"content": "OK"}}]},
        observation=_observation(slimserve_config), observation_status=200, observation_content=None,
        availability={
            "name": "slimserve", "version": SLIMSERVE_COMMIT, "adapter": "slimserve-runtime",
            "variants": ["a100", "rtx3090"],
            "kernel_library": {"name": "quixicore-cuda",
                               "version": f"{QUIXICORE_CUDA_COMMIT}+slimserve.{SLIMSERVE_COMMIT}"},
        },
    )

    class Pipe:
        def __init__(self):
            self.data = bytearray()
            self.closed = False

        def write(self, data):
            assert not self.closed
            self.data.extend(data)

        async def drain(self):
            if control.drain_error is not None:
                raise control.drain_error

        def close(self):
            self.closed = True

    class Process:
        def __init__(self, kind):
            self.kind = kind
            self.pid = 40000 + len(control.processes)
            self.returncode = None
            self.group_alive = True
            self.stdin = Pipe()
            self.input = None

        async def communicate(self, data=None):
            self.input = data
            control.events.append(("communicate", self.kind, self.pid))
            if self.kind == "check":
                if control.check_error is not None:
                    raise control.check_error
                self.returncode = control.check_returncode
                if control.check_content is not None:
                    return control.check_content, b""
                model = control.resolved_model or json.loads(data)["roles"]["generation"]["model"]
                return json.dumps({"validated": True, "engine_model": model}).encode(), b""
            assert self.kind == "availability"
            self.returncode = 0
            return json.dumps(control.availability).encode(), b""

        async def wait(self):
            control.events.append(("wait", self.kind, self.pid))
            if self.returncode is None:
                raise asyncio.TimeoutError
            return self.returncode

    async def spawn(*args, **kwargs):
        kind = "check" if "--check" in args else "availability" if "--availability" in args else "generation"
        control.calls.append(SimpleNamespace(kind=kind, args=args, kwargs=kwargs))
        if control.spawn_error == kind:
            raise OSError("boundary-only launch refusal")
        if kind == "generation":
            assert not any(process.kind == kind and process.group_alive for process in control.processes)
        process = Process(kind)
        control.processes.append(process)
        control.events.append(("spawn", kind, process.pid))
        return process

    def killpg(pid, sig):
        process = next(process for process in control.processes if process.pid == pid)
        control.events.append(("signal", pid, sig))
        if process.kind == "generation" and control.stop_error is not None:
            raise control.stop_error
        if sig == 0:
            if process.kind == "generation" and control.group_probe_error is not None:
                raise control.group_probe_error
            if process.group_alive:
                return
            raise ProcessLookupError
        if not process.group_alive:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            process.group_alive = process.kind == "generation" and control.descendants_survive
            if process.returncode is None:
                process.returncode = -sig
        elif not control.hold_term and process.returncode is None:
            process.returncode = -sig

    def respond(request):
        control.requests.append(request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/v1/chat/completions":
            return httpx.Response(control.chat_status, json=control.chat_body)
        if path == "/sovereign/quiesce":
            return httpx.Response(200, json={"quiesced": True})
        if path == "/sovereign/resume":
            return httpx.Response(200, json={"resumed": True})
        assert path == "/sovereign/observation"
        if control.observation_content is not None:
            return httpx.Response(control.observation_status, content=control.observation_content)
        return httpx.Response(control.observation_status, json=control.observation)

    def client(*args, **kwargs):
        control.client_options.append(deepcopy(kwargs))
        result = AsyncClient(*args, **kwargs, transport=httpx.MockTransport(respond))
        control.clients.append(result)
        return result

    monkeypatch.setattr("lazarus.appliance.backends.slimserve.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("lazarus.appliance.backends.slimserve.os.killpg", killpg)
    monkeypatch.setattr("lazarus.appliance.backends.slimserve.httpx.AsyncClient", client)
    return control


def _assert_no_evidence(backend):
    assert backend.observation() == {}
    assert backend.engine_version() is None
    assert backend.accelerator() == {"vendor": "none", "device_count": 0, "unified_memory": False}
    info = backend.role_info("generation")
    assert all(getattr(info, field) is None for field in EXECUTION_FIELDS)


def test_launch_checks_before_start_with_fixed_interpreter_and_isolated_environment(slimserve_config, boundaries, monkeypatch):
    unsafe = {
        "PATH": "/tmp/untrusted-bin", "PYTHONPATH": "/tmp/plugins", "PYTHONHOME": "/tmp/python",
        "VLLM_BACKEND": "cpu", "VLLM_PLUGINS": "unapproved", "VLLM_USE_V1": "0",
        "VLLM_WORKER_MULTIPROC_METHOD": "fork", "VLLM_MODEL_REDIRECT_PATH": "/tmp/model",
        "HF_TOKEN": "secret", "HUGGING_FACE_HUB_TOKEN": "secret", "HF_ENDPOINT": "https://remote.invalid",
        "HF_HOME": "/tmp/cache", "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0",
        "HTTP_PROXY": "http://proxy.invalid", "HTTPS_PROXY": "http://proxy.invalid",
        "SOVEREIGN_SLIMSERVE_INTERPRETER": "/tmp/python", "SOVEREIGN_SLIMSERVE_MODEL": "/tmp/model",
        "SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN": "inherited-secret",
    }
    for key, value in unsafe.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(UUIDS[:2]))
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", ",".join(UUIDS[:2]))
    backend = SlimServeBackend()

    async def exercise():
        states = []
        try:
            await backend.start(slimserve_config, states.append)
            assert states == ["loading"]
            assert backend.role_info("generation").status == "healthy"
            assert backend.observation() == boundaries.observation
            assert [call.kind for call in boundaries.calls] == ["check", "generation"]
            check, generation = boundaries.processes
            assert json.loads(check.input) == slimserve_config.model_dump(mode="json", exclude_none=True, exclude_unset=True)
            assert json.loads(generation.stdin.data) == json.loads(check.input)
            assert generation.stdin.closed
            assert boundaries.events.index(("signal", check.pid, signal.SIGKILL)) < boundaries.events.index(("spawn", "generation", generation.pid))
            for call in boundaries.calls:
                assert call.args[:3] == ("/opt/sovereign-slimserve/bin/python", "-m", "lazarus.appliance.slimserve_launch")
                assert call.args[3:] == (("--check",) if call.kind == "check" else ())
                assert call.kwargs["start_new_session"] is True
                assert call.kwargs["cwd"] == "/"
                environment = call.kwargs["env"]
                assert environment["PATH"].split(":")[0] == "/opt/sovereign-slimserve/bin"
                assert environment["HF_HUB_OFFLINE"] == environment["TRANSFORMERS_OFFLINE"] == "1"
                assert environment["HF_DATASETS_OFFLINE"] == environment["HF_HUB_DISABLE_TELEMETRY"] == "1"
                assert environment["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
                assert environment["VLLM_PLUGINS"] == ""
                assert environment["CUDA_VISIBLE_DEVICES"] == environment["NVIDIA_VISIBLE_DEVICES"] == ",".join(UUIDS[:2])
                assert not any(environment.get(key) == value for key, value in unsafe.items())
            assert "SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN" not in boundaries.calls[0].kwargs["env"]
            token = boundaries.calls[1].kwargs["env"]["SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN"]
            assert token and all(request.headers["x-sovereign-observation-token"] == token for request in boundaries.requests)
            assert boundaries.client_options[0]["trust_env"] is False
            assert boundaries.client_options[0]["base_url"] == "http://127.0.0.1:18001"
            assert [request.url.path for request in boundaries.requests] == ["/health", "/v1/chat/completions", "/sovereign/observation"]
        finally:
            await backend.shutdown()
        assert all(client.is_closed for client in boundaries.clients)
        assert not any(process.group_alive for process in boundaries.processes)

    asyncio.run(exercise())


@pytest.mark.parametrize("outcome", ["shutdown", "startup-exit", "runtime-exit"])
def test_serving_child_output_is_private_without_losing_supervision(
    slimserve_config, tmp_path, monkeypatch, capfd, caplog, outcome,
):
    # Real child descriptors and process groups; only private HTTP is simulated.
    # The emitted marker proves both writes happened before readiness is tested.
    ready = tmp_path / "child-emitted"
    private_paths = {
        "model": "/private/models/PRIVATE_MODEL_SENTINEL",
        "tokenizer": "/private/tokenizers/PRIVATE_TOKENIZER_SENTINEL",
        "drafter": "/private/drafters/PRIVATE_DRAFTER_SENTINEL",
    }
    raw_arguments = f"non-default args: {private_paths!r}"
    program = (
        "import json, os, pathlib, signal, sys\n"
        "config = json.load(sys.stdin)\n"
        "if sys.argv[1] == '--check':\n"
        "    print(json.dumps({'validated': True, 'engine_model': config['roles']['generation']['model']}))\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGUSR1, lambda *_: sys.exit(23))\n"
        f"os.write(1, {('STDOUT_ARGUMENTS_SENTINEL ' + raw_arguments + chr(10)).encode()!r})\n"
        f"os.write(2, {('STDERR_ARGUMENTS_SENTINEL ' + raw_arguments + chr(10)).encode()!r})\n"
        "pathlib.Path(sys.argv[2]).write_text('emitted')\n"
        "if sys.argv[1] == 'startup-exit':\n"
        "    raise SystemExit(23)\n"
        "signal.pause()\n"
    )
    spawn_child = asyncio.create_subprocess_exec
    processes = []
    observation = _observation(slimserve_config)
    backend = SlimServeBackend()
    states = []

    async def spawn(*args, **kwargs):
        mode = "--check" if "--check" in args else outcome
        process = await spawn_child(sys.executable, "-c", program, mode, str(ready), **kwargs)
        if mode != "--check":
            processes.append(process)
        return process

    async def wait_emitted():
        while not ready.is_file():
            await asyncio.sleep(0.01)

    async def respond(request):
        await asyncio.wait_for(wait_emitted(), 5)
        if request.url.path == "/health":
            if outcome == "startup-exit":
                await asyncio.wait_for(processes[-1].wait(), 5)
                return httpx.Response(503)
            return httpx.Response(200)
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={
                "model": slimserve_config.roles.generation.served_model_name,
                "choices": [{"message": {"content": "OK"}}],
            })
        assert request.url.path == "/sovereign/observation"
        return httpx.Response(200, json=observation)

    def on_state(state):
        states.append(state)
        print(f"SlimServe state: {state}", flush=True)

    monkeypatch.setattr("lazarus.appliance.backends.slimserve.asyncio.create_subprocess_exec", spawn)
    monkeypatch.setattr("lazarus.appliance.backends.slimserve.httpx.AsyncClient", lambda *args, **kwargs:
        AsyncClient(*args, **kwargs, transport=httpx.MockTransport(respond)))

    async def exercise():
        try:
            if outcome == "startup-exit":
                with pytest.raises(BackendStartError, match="generation process exited while loading") as error:
                    await asyncio.wait_for(backend.start(slimserve_config, on_state), 10)
                assert error.value.code == "MODEL_LOAD_FAILED"
                assert backend.role_info("generation").status == "unhealthy"
                assert backend.role_info("generation").error_code == "MODEL_LOAD_FAILED"
                assert processes[-1].returncode == 23
                assert states == ["loading"]
                _assert_no_evidence(backend)
            else:
                await asyncio.wait_for(backend.start(slimserve_config, on_state), 10)
                assert ready.is_file()
                assert processes[-1].returncode is None
                assert states == ["loading"]
                assert backend.role_info("generation").status == "healthy"
                assert backend.observation() == observation
                if outcome == "runtime-exit":
                    monitor = backend._monitor
                    os.kill(processes[-1].pid, signal.SIGUSR1)
                    assert await asyncio.wait_for(processes[-1].wait(), 5) == 23
                    await asyncio.wait_for(monitor, 5)
                    assert states == ["loading", "runtime_error"]
                    assert backend.role_info("generation").status == "unhealthy"
                    assert backend.role_info("generation").error_code == "ENGINE_DEAD"
                    assert backend.generation_paused is True
                    _assert_no_evidence(backend)
        finally:
            await asyncio.wait_for(backend.shutdown(), 10)
        assert processes[-1].returncode == (-signal.SIGTERM if outcome == "shutdown" else 23)
        with pytest.raises(ProcessLookupError):
            os.killpg(processes[-1].pid, 0)
        assert backend.role_info("generation").status == "disabled"
        assert backend._withdrawn is True
        _assert_no_evidence(backend)

    asyncio.run(exercise())
    captured = capfd.readouterr()
    assert "SlimServe state: loading" in captured.out
    if outcome == "runtime-exit":
        assert "SlimServe state: runtime_error" in captured.out
    logs = captured.out + captured.err + caplog.text
    for sentinel in (*private_paths.values(), "STDOUT_ARGUMENTS_SENTINEL", "STDERR_ARGUMENTS_SENTINEL"):
        assert sentinel not in logs


def test_child_environment_is_fresh(monkeypatch):
    monkeypatch.setenv("LANG", "C.UTF-8")
    first = child_environment()
    first["LANG"] = "changed"
    assert child_environment()["LANG"] == "C.UTF-8"


@pytest.mark.parametrize("failure,code", [
    ("rejected", "CONFIG_INVALID"), ("timeout", "CONFIG_INVALID"),
    ("missing-interpreter", "ACCELERATOR_UNAVAILABLE"), ("untyped-result", "CONFIG_INVALID"),
    ("missing-resolution", "CONFIG_INVALID"), ("oversize-result", "CONFIG_INVALID"),
])
def test_preflight_failure_never_starts_generation(slimserve_config, boundaries, failure, code):
    if failure == "rejected":
        boundaries.check_returncode = 2
    elif failure == "timeout":
        boundaries.check_error = asyncio.TimeoutError()
    elif failure == "missing-interpreter":
        boundaries.spawn_error = "check"
    elif failure == "untyped-result":
        boundaries.check_content = json.dumps({"validated": 1, "engine_model": slimserve_config.roles.generation.model}).encode()
    elif failure == "missing-resolution":
        boundaries.check_content = b'{"validated":true}'
    else:
        boundaries.check_content = b" " * 4097
    backend = SlimServeBackend()

    async def exercise():
        states = []
        try:
            with pytest.raises(BackendStartError) as error:
                await backend.start(slimserve_config, states.append)
            assert error.value.code == code
            assert states == []
            assert [call.kind for call in boundaries.calls] == ["check"]
            assert boundaries.requests == []
            assert backend.role_info("generation").status != "healthy"
            _assert_no_evidence(backend)
            assert not any(process.group_alive for process in boundaries.processes)
            requests = len(boundaries.requests)
            await backend.quiesce()
            assert backend.generation_paused is True
            assert len(boundaries.requests) == requests
            with pytest.raises(BackendStartError, match="unavailable for resume"):
                await backend.resume()
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", [
    "spawn", "stdin", "smoke-http", "smoke-identity", "observation-http",
    "observation-size", "observation-json", "observation-drift",
])
def test_start_failure_reaps_generation_and_drops_evidence(slimserve_config, boundaries, failure):
    if failure == "spawn":
        boundaries.spawn_error = "generation"
    elif failure == "stdin":
        boundaries.drain_error = BrokenPipeError("candidate exited before config arrived")
    elif failure == "smoke-http":
        boundaries.chat_status = 500
    elif failure == "smoke-identity":
        boundaries.chat_body["model"] = "wrong-alias"
    elif failure == "observation-http":
        boundaries.observation_status = 500
    elif failure == "observation-size":
        boundaries.observation_content = b" " * (64 * 1024 + 1)
    elif failure == "observation-json":
        boundaries.observation_content = b"not-json"
    else:
        boundaries.observation["revision"] = "d" * 40
    backend = SlimServeBackend()

    async def exercise():
        try:
            with pytest.raises(BackendStartError) as error:
                await backend.start(slimserve_config, lambda state: None)
            assert error.value.code == ("SMOKE_TEST_FAILED" if failure == "smoke-identity" else "MODEL_LOAD_FAILED")
            assert backend.role_info("generation").status == "unhealthy"
            _assert_no_evidence(backend)
            assert not any(process.group_alive for process in boundaries.processes)
            assert all(client.is_closed for client in boundaries.clients)
            requests = len(boundaries.requests)
            await backend.quiesce()
            assert backend.generation_paused is True
            assert len(boundaries.requests) == requests
            with pytest.raises(BackendStartError, match="unavailable for resume"):
                await backend.resume()
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


def test_supervisor_rejects_different_declared_gguf_than_independently_resolved_input(slimserve_document, boundaries):
    files = slimserve_document["roles"]["generation"]["slimserve"]["artifacts"][0]["files"]
    files[0]["file"] = "selected.gguf"
    files.append({"file": "other.gguf", "size_bytes": 6, "sha256": "d" * 64})
    config = RuntimeConfig.model_validate(slimserve_document)
    boundaries.resolved_model = config.roles.generation.model + "/selected.gguf"
    boundaries.observation = _observation(config)
    boundaries.observation["engine_model"] += "/other.gguf"
    backend = SlimServeBackend()

    async def exercise():
        try:
            with pytest.raises(BackendStartError, match="independently resolved"):
                await backend.start(config, lambda state: None)
            _assert_no_evidence(backend)
            assert not any(process.group_alive for process in boundaries.processes)
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


def test_second_start_is_rejected_without_validation_or_another_tree(slimserve_config, boundaries):
    backend = SlimServeBackend()

    async def exercise():
        try:
            await backend.start(slimserve_config, lambda state: None)
            observed = backend.observation()
            with pytest.raises(BackendStartError) as error:
                await backend.start(slimserve_config, lambda state: None)
            assert error.value.code == "CONFIG_INVALID"
            assert [call.kind for call in boundaries.calls] == ["check", "generation"]
            assert backend.role_info("generation").status == "healthy"
            assert backend.observation() == observed
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("hold_term", [False, True])
def test_shutdown_reaps_group_and_restart_has_no_old_observation(slimserve_config, boundaries, hold_term):
    backend = SlimServeBackend()

    async def exercise():
        try:
            await backend.start(slimserve_config, lambda state: None)
            old = boundaries.processes[-1]
            boundaries.hold_term = hold_term
            await backend.shutdown()
            signals = [event[2] for event in boundaries.events if event[:2] == ("signal", old.pid)]
            assert signals == [signal.SIGTERM, signal.SIGKILL, 0]
            assert not old.group_alive
            assert backend.role_info("generation").status == "disabled"
            _assert_no_evidence(backend)
            await backend.shutdown()
            boundaries.hold_term = False
            boundaries.check_returncode = 1
            with pytest.raises(BackendStartError):
                await backend.start(slimserve_config, lambda state: None)
            _assert_no_evidence(backend)
            boundaries.check_returncode = 0
            candidate = slimserve_config.model_copy(deep=True)
            candidate.roles.generation.revision = "d" * 40
            candidate.roles.generation.slimserve.artifacts[0].revision = "d" * 40
            boundaries.observation = _observation(candidate)
            await backend.start(candidate, lambda state: None)
            assert backend.role_info("generation").revision == "d" * 40
            assert backend.observation() == boundaries.observation
            assert boundaries.processes[-1].pid != old.pid
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["preflight", "smoke"])
@pytest.mark.parametrize("profile", ["cuda", "metal"])
def test_appliance_failure_is_live_nonready_without_requested_execution(slimserve_document, tmp_path, boundaries, failure, profile, monkeypatch):
    if failure == "preflight":
        boundaries.check_returncode = 1
    else:
        boundaries.chat_status = 500
    monkeypatch.setenv("SOVEREIGN_RUNTIME_API_KEY", "secret")
    if profile == "metal":
        config = _metal_config(slimserve_document)
        slimserve_document = config.model_dump(exclude_none=True, exclude_unset=True)
        boundaries.observation = _observation(config)
    appliance = Appliance(config_path=str(_write_config(tmp_path, slimserve_document)), backend=SlimServeBackend())
    assert type(appliance.backend) is SlimServeBackend
    assert appliance.config is not None and appliance.config_error is None

    async def exercise():
        try:
            await appliance.run_lifecycle()
            expected_calls = ["availability", "check"]
            expected_requests = []
            if failure == "smoke":
                expected_calls.append("generation")
                expected_requests = ["/health", "/v1/chat/completions"]
            assert [call.kind for call in boundaries.calls] == expected_calls
            assert [request.url.path for request in boundaries.requests] == expected_requests
            assert not any(process.group_alive for process in boundaries.processes)
            assert all(client.is_closed for client in boundaries.clients)
            _assert_no_evidence(appliance.backend)
            request_count = len(boundaries.requests)
            assert all(call.kwargs["cwd"] == "/" for call in boundaries.calls)
            async with AsyncClient(transport=ASGITransport(app=appliance.app), base_url="http://runtime", headers={"Authorization": "Bearer secret"}) as client:
                for operation in ("quiesce", "resume"):
                    denied = await client.post(
                        f"/runtime/admin/generation/{operation}",
                        headers={"Authorization": "Bearer incorrect"},
                    )
                    assert denied.status_code == 401
                quiesced = await client.post("/runtime/admin/generation/quiesce")
                assert quiesced.status_code == 200 and quiesced.json() == {"quiesced": True}
                assert appliance.backend.generation_paused is True
                resumed = await client.post("/runtime/admin/generation/resume")
                assert resumed.status_code == 503
                assert resumed.json()["error"]["code"] == "ENGINE_RESUME_FAILED"
                assert (await client.get("/health/live")).status_code == 200
                ready = await client.get("/health/ready")
                assert ready.status_code == 503 and ready.json()["ready"] is False
                errors = (await client.get("/runtime/errors")).json()["errors"]
                assert errors[-1]["code"] == ("CONFIG_INVALID" if failure == "preflight" else "MODEL_LOAD_FAILED")
                assert errors[-1]["role"] == "generation"
                manifest = (await client.get("/runtime/manifest")).json()
                assert manifest["schema_version"] == "1.3"
                assert manifest["state"] == "configuration_error"
                assert manifest["generation_paused"] is True
                assert manifest["available_engines"] == [boundaries.availability]
                assert "engine" not in manifest and "kernels" not in manifest
                assert manifest["health"]["kernels"] == "unknown"
                assert not any(field in manifest["roles"]["generation"] for field in EXECUTION_FIELDS)
                for model, status_code in (
                    (slimserve_document["roles"]["generation"]["served_model_name"], 503),
                    ("unknown-model", 404),
                ):
                    response = await client.post("/v1/chat/completions", json={
                        "model": model, "messages": [{"role": "user", "content": "hello"}],
                    })
                    assert response.status_code == status_code
                assert len(boundaries.requests) == request_count
            assert not any(process.group_alive for process in boundaries.processes)
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["profile-drift", "worker-drift", "http-failure", "process-exit"])
def test_monitor_failure_revokes_api_readiness_and_all_execution_evidence(slimserve_document, tmp_path, boundaries, monkeypatch, failure):
    manifest_path = tmp_path / "state" / "manifest.json"
    monkeypatch.setenv("SOVEREIGN_RUNTIME_MANIFEST", str(manifest_path))
    appliance = Appliance(config_path=str(_write_config(tmp_path, slimserve_document)))

    async def exercise():
        try:
            await appliance.run_lifecycle()
            async with AsyncClient(transport=ASGITransport(app=appliance.app), base_url="http://runtime") as client:
                assert (await client.get("/health/ready")).status_code == 200
                initial = (await client.get("/runtime/manifest")).json()
                assert initial["engine"] == boundaries.observation["engine"]
                assert initial["kernels"] == boundaries.observation["kernels"]
                assert initial["kernels"]["version"] == f"{QUIXICORE_CUDA_COMMIT}+slimserve.{SLIMSERVE_COMMIT}"
                assert initial["accelerator"] == boundaries.observation["accelerator"]
                assert initial["available_engines"] == [boundaries.availability]
                assert all(initial["roles"]["generation"][field] == boundaries.observation[field] for field in EXECUTION_FIELDS)
                process = boundaries.processes[-1]
                if failure == "profile-drift":
                    boundaries.observation["engine_profile_id"] = "other-profile"
                elif failure == "worker-drift":
                    boundaries.observation["accelerator"]["devices"][0]["gpu_uuid"] = UUIDS[2]
                elif failure == "http-failure":
                    boundaries.observation_status = 503
                else:
                    process.returncode = 1
                await asyncio.wait_for(appliance.backend._monitor, timeout=10)
                ready = await client.get("/health/ready")
                assert ready.status_code == 503 and ready.json()["ready"] is False
                assert (await client.get("/health/live")).status_code == 200
                assert appliance.backend.role_info("generation").status == "unhealthy"
                _assert_no_evidence(appliance.backend)
                errors = (await client.get("/runtime/errors")).json()["errors"]
                assert errors[-1]["code"] == "ENGINE_DEAD"
                assert errors[-1]["recoverable"] is False
                manifest = (await client.get("/runtime/manifest")).json()
                assert manifest["state"] == "runtime_error"
                assert "engine" not in manifest and "kernels" not in manifest
                assert not any(field in manifest["roles"]["generation"] for field in EXECUTION_FIELDS)
                assert json.loads(manifest_path.read_text()) == manifest
                assert not process.group_alive
                request_count = len(boundaries.requests)
                for model, status_code in (
                    (slimserve_document["roles"]["generation"]["served_model_name"], 503),
                    ("unknown-model", 404),
                ):
                    response = await client.post("/v1/chat/completions", json={
                        "model": model, "messages": [{"role": "user", "content": "hello"}],
                    })
                    assert response.status_code == status_code
                assert len(boundaries.requests) == request_count
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("field", ["gpu_uuid", "stable_identifier", "identity_kind", "local_rank"])
def test_observation_rejects_missing_cuda_worker_identity(slimserve_config, field):
    payload = _observation(slimserve_config)
    del payload["accelerator"]["devices"][0][field]
    with pytest.raises(BackendStartError):
        validate_observation(slimserve_config, payload)


@pytest.mark.parametrize("field", ["stable_identifier", "identity_kind", "platform_id"])
def test_observation_rejects_missing_native_worker_identity(slimserve_document, field):
    config = _metal_config(slimserve_document)
    payload = _observation(config)
    del payload["accelerator"]["devices"][0][field]
    with pytest.raises(BackendStartError):
        validate_observation(config, payload)


@pytest.mark.parametrize("changed", [False, True])
def test_start_reuses_preflight_only_for_the_identical_config(slimserve_config, boundaries, changed):
    backend = SlimServeBackend()

    async def exercise():
        try:
            await backend.validate(slimserve_config)
            candidate = slimserve_config.model_copy(deep=True)
            if changed:
                candidate.roles.generation.max_model_len = 4096
                boundaries.observation = _observation(candidate)
            await backend.start(candidate, lambda state: None)
            assert [call.kind for call in boundaries.calls] == (
                ["check", "check", "generation"] if changed else ["check", "generation"]
            )
            assert backend.role_info("generation").context_length == candidate.roles.generation.max_model_len
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["signal", "group-query", "descendants"])
def test_failed_cleanup_retains_owned_tree_and_refuses_quiescence(slimserve_config, boundaries, monkeypatch, failure):
    boundaries.chat_status = 500
    if failure == "signal":
        boundaries.stop_error = PermissionError("owned group stop refused")
    elif failure == "group-query":
        boundaries.group_probe_error = PermissionError("owned group absence unknown")
    else:
        boundaries.descendants_survive = True
        original_sleep = asyncio.sleep

        async def expired_cleanup(delay):
            if delay == 0.05:
                raise asyncio.TimeoutError("owned descendants still present")
            await original_sleep(delay)

        monkeypatch.setattr("lazarus.appliance.backends.slimserve.asyncio.sleep", expired_cleanup)
    backend = SlimServeBackend()

    async def exercise():
        try:
            with pytest.raises((PermissionError, asyncio.TimeoutError)):
                await backend.start(slimserve_config, lambda state: None)
            owned = boundaries.processes[-1]
            assert backend._process is owned
            assert backend.role_info("generation").status != "healthy"
            assert backend.observation() == {}
            for operation in (backend.quiesce, backend.resume):
                with pytest.raises(BackendStartError, match="cleanup is incomplete"):
                    await operation()
                assert backend.generation_paused is True
            with pytest.raises(BackendStartError):
                await backend.start(slimserve_config, lambda state: None)
            assert sum(call.kind == "generation" for call in boundaries.calls) == 1
            boundaries.stop_error = boundaries.group_probe_error = None
            boundaries.descendants_survive = False
            await backend.shutdown()
            await backend.quiesce()
            assert not owned.group_alive
            assert backend._process is None
            assert backend.generation_paused is True
        finally:
            boundaries.stop_error = boundaries.group_probe_error = None
            boundaries.descendants_survive = False
            await backend.shutdown()

    asyncio.run(exercise())


def test_failed_start_restoration_stays_closed_until_real_resume(slimserve_config, boundaries):
    backend = SlimServeBackend()

    async def exercise():
        try:
            boundaries.chat_status = 500
            with pytest.raises(BackendStartError):
                await backend.start(slimserve_config, lambda state: None)
            await backend.quiesce()
            await backend.shutdown()
            boundaries.chat_status = 200
            await backend.start(slimserve_config, lambda state: None)
            assert backend.role_info("generation").status == "healthy"
            assert backend.generation_paused is True
            await backend.resume()
            assert boundaries.requests[-1].url.path == "/sovereign/resume"
            assert backend.generation_paused is False
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


def test_cancelled_spawn_without_handle_never_claims_withdrawal(slimserve_config, boundaries, monkeypatch):
    backend = SlimServeBackend()
    original_spawn = asyncio.create_subprocess_exec

    async def cancelled_spawn(*args, **kwargs):
        if "--check" in args:
            return await original_spawn(*args, **kwargs)
        raise asyncio.CancelledError

    monkeypatch.setattr("lazarus.appliance.backends.slimserve.asyncio.create_subprocess_exec", cancelled_spawn)

    async def exercise():
        with pytest.raises(asyncio.CancelledError):
            await backend.start(slimserve_config, lambda state: None)
        assert backend.role_client("generation") is None
        await backend.shutdown()
        with pytest.raises(BackendStartError, match="unavailable for quiescence"):
            await backend.quiesce()
        assert backend.generation_paused is True

    asyncio.run(exercise())


@pytest.mark.parametrize("adapter", [AgentBackend, SlimServeAgentBackend])
def test_native_installed_slimserve_survives_host_manifest_and_inherited_adapter(boundaries, tmp_path, monkeypatch, adapter):
    from lazarus.agent.config import AgentConfig
    from lazarus.agent.server import Agent, build_app

    boundaries.availability.update(
        variants=["metal"],
        kernel_library={"name": "quixicore-metal", "version": f"{QUIXICORE_METAL_COMMIT}+slimserve.{SLIMSERVE_COMMIT}"},
    )
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "host-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_URL", "http://host")
    host = Agent(AgentConfig(roles={}, llama_server=str(tmp_path / "missing" / "llama-server")))

    async def exercise():
        await host.discover_engines()
        assert host.available_engines == [boundaries.availability]
        assert [call.kind for call in boundaries.calls] == ["availability"]
        assert not any(process.group_alive for process in boundaries.processes)
        app = build_app(host)
        monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: AsyncClient(
            *args, transport=ASGITransport(app=app), **kwargs,
        ))
        available = await adapter().available_engines()
        assert available == host.available_engines
        assert available[0]["variants"] == ["metal"]
        assert available[0]["kernel_library"] == boundaries.availability["kernel_library"]

    asyncio.run(exercise())


@pytest.mark.parametrize("adapter", [AgentBackend, SlimServeAgentBackend])
def test_host_variants_are_engine_specific_without_normalizing_identity(monkeypatch, adapter):
    llama = {"name": "llama.cpp", "version": "b9960-a935fbffe", "adapter": "metal-host-agent", "variants": ["metal-arm64"]}
    slimserve = {"name": "slimserve", "version": SLIMSERVE_COMMIT, "adapter": "slimserve-runtime", "variants": ["metal"]}
    available = [
        llama, slimserve,
        {**llama, "variants": ["metal"]}, {**slimserve, "variants": ["metal-arm64"]},
        {**slimserve, "variants": ["a100", "rtx3090"]}, {**slimserve, "version": None},
        {**slimserve, "name": ["slimserve"]},
    ]
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "host-secret")

    def respond(request):
        assert request.headers["Authorization"] == "Bearer host-secret"
        return httpx.Response(200, json={"available_engines": available})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: AsyncClient(
        *args, transport=httpx.MockTransport(respond), **kwargs,
    ))
    assert asyncio.run(adapter().available_engines()) == [llama, slimserve]


def test_native_owned_group_is_reaped_before_withdrawal_acknowledgement():
    # Exercise OS supervision, not a simulated returncode or successful signal.
    program = (
        "import signal, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "def stop(signum, frame):\n"
        "    child.wait(timeout=5)\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "print('ready', flush=True)\n"
        "signal.pause()\n"
    )

    async def exercise():
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-c", program, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            assert await asyncio.wait_for(process.stdout.readline(), 5) == b"ready\n"
            await asyncio.wait_for(_stop_group(process), 10)
            assert process.returncode == 0
            with pytest.raises(ProcessLookupError):
                os.killpg(process.pid, 0)
        finally:
            if process.returncode is None:
                await _stop_group(process)

    asyncio.run(exercise())


@pytest.mark.parametrize("uncertain_cleanup", [False, True])
def test_host_recovery_drains_handlers_then_requires_verified_withdrawal(
    slimserve_document, boundaries, tmp_path, monkeypatch, uncertain_cleanup,
):
    from lazarus.agent.config import AgentConfig
    from lazarus.agent.server import Agent, build_app

    config = _metal_config(slimserve_document)
    boundaries.chat_status = 500
    boundaries.observation = _observation(config)
    if uncertain_cleanup:
        boundaries.group_probe_error = PermissionError("cannot prove native group absence")
    monkeypatch.setenv("SOVEREIGN_AGENT_TOKEN", "host-secret")
    monkeypatch.setenv("SOVEREIGN_AGENT_URL", "http://host")
    backend = SlimServeBackend()
    host = Agent(AgentConfig(roles={}), tmp_path / "agent.yaml")
    host.generation_backend = backend

    async def exercise():
        try:
            with pytest.raises(PermissionError if uncertain_cleanup else BackendStartError):
                await backend.start(config, lambda state: None)
            handler_wait = asyncio.Event()

            class HandlerDrain(asyncio.Event):
                async def wait(self):
                    handler_wait.set()
                    return await super().wait()

            host.generation_idle = HandlerDrain()
            host.generation_requests = 1
            app = build_app(host)
            monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: AsyncClient(
                *args, transport=ASGITransport(app=app), **kwargs,
            ))
            remote = SlimServeAgentBackend()
            pending = asyncio.create_task(remote.quiesce())
            await asyncio.wait_for(handler_wait.wait(), 2)
            assert host.generation_admission_paused is True
            assert remote.generation_paused is True
            assert not pending.done(), "withdrawal cannot bypass admitted handler completion"
            host.generation_requests = 0
            host.generation_idle.set()
            if uncertain_cleanup:
                with pytest.raises(httpx.HTTPStatusError) as error:
                    await asyncio.wait_for(pending, 2)
                assert error.value.response.status_code == 503
                assert backend._process is not None
            else:
                await asyncio.wait_for(pending, 2)
                assert backend._process is None
            with pytest.raises(httpx.HTTPStatusError) as error:
                await remote.resume()
            assert error.value.response.status_code == 503
            assert remote.generation_paused is True
            assert host.generation_admission_paused is True
            boundaries.group_probe_error = None
            await backend.shutdown()
            await remote.quiesce()
            assert remote.generation_paused is True
        finally:
            boundaries.group_probe_error = None
            await backend.shutdown()

    asyncio.run(exercise())


def test_missing_client_with_owned_process_is_not_withdrawal(slimserve_config, boundaries):
    backend = SlimServeBackend()

    async def exercise():
        try:
            await backend.start(slimserve_config, lambda state: None)
            await backend._client.aclose()
            backend._client = None
            assert backend._process.returncode is None
            with pytest.raises(BackendStartError, match="unavailable for quiescence"):
                await backend.quiesce()
            assert backend.generation_paused is True
        finally:
            await backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("backend_type", [SlimServeBackend, VllmBackend])
def test_missing_client_without_established_startup_cleanup_is_not_proof(backend_type):
    backend = backend_type()

    async def exercise():
        assert backend.role_client("generation") is None
        await backend.shutdown()
        with pytest.raises(BackendStartError):
            await backend.quiesce()
        assert backend.generation_paused is True

    asyncio.run(exercise())


def test_native_preflight_cleanup_preserves_first_start_but_never_clears_pause(slimserve_config, boundaries):
    backend = SlimServeBackend()

    async def exercise():
        try:
            await backend.validate(slimserve_config)
            await backend.shutdown()
            await backend.start(slimserve_config, lambda state: None)
            assert backend.generation_paused is False
            await backend.quiesce()
            await backend.shutdown()
            await backend.start(slimserve_config, lambda state: None)
            assert backend.generation_paused is True
            await backend.resume()
            assert backend.generation_paused is False
        finally:
            await backend.shutdown()

    asyncio.run(exercise())
