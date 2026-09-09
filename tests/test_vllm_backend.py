import argparse
import asyncio
import json
import socket
import sys
import time
import threading
from types import ModuleType, SimpleNamespace

import httpx
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lazarus.appliance.backends.base import BackendStartError
from lazarus.appliance.backends.vllm_engine import (
    VllmBackend,
    _canonical_torch_uuid,
    _validate_local_cuda_bundle,
)
from lazarus.appliance.config import load_config
from lazarus.appliance.launcher import Appliance


def test_engine_construction_does_not_block_api_loop(monkeypatch):
    backend = VllmBackend()

    def slow_constructor(_name, _role):
        time.sleep(0.08)
        return "built"

    monkeypatch.setattr(backend, "_construct_role_engine_sync", slow_constructor)

    async def exercise():
        task = asyncio.create_task(backend._construct_role_engine("generation", None))
        await asyncio.sleep(0.01)
        assert not task.done(), "synchronous model loading blocked the API event loop"
        return await task

    assert asyncio.run(exercise()) == "built"


@pytest.mark.parametrize("count", [1, 2, 4])
@pytest.mark.parametrize("eager", [False, True])
def test_managed_cuda_launch_uses_structured_profile(
    managed_cuda_config_file, monkeypatch, count, eager
):
    monkeypatch.setenv("VLLM_BACKEND", "cuda")
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    generation = data["roles"]["generation"]
    generation["accelerator_device_ids"] = [
        f"GPU-00000000-0000-0000-0000-{rank:012x}" for rank in range(count, 0, -1)
    ]
    generation["tensor_parallel_size"] = count
    generation["enforce_eager"] = eager
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    role = load_config(managed_cuda_config_file).roles.generation
    argv = VllmBackend()._role_argv("generation", role)
    position = argv.index("--tensor-parallel-size")
    assert argv[position + 1] == str(count)
    assert argv.count("--tensor-parallel-size") == 1
    assert argv.count("--enforce-eager") == int(eager)
    assert argv[argv.index("--model") + 1] == generation["model"]
    assert argv[argv.index("--revision") + 1] == generation["revision"]
    assert argv[argv.index("--max-model-len") + 1] == "2048"
    assert argv[argv.index("--tool-call-parser") + 1] == "gemma4_native"
    assert "--reasoning-parser" not in argv
    assert not set(generation["accelerator_device_ids"]).intersection(argv)
    assert "--hf-config-path" not in argv
    assert "--tokenizer" not in argv
    assert "--trust-remote-code" not in argv


def test_torch_uuid_normalization_preserves_physical_identity():
    expected = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    assert _canonical_torch_uuid(expected) == expected
    assert _canonical_torch_uuid(bytes.fromhex("aaaaaaaabbbbccccddddeeeeeeeeeeee")) == expected
    assert _canonical_torch_uuid("0") is None


@pytest.fixture()
def prepared_bundle(tmp_path):
    root = tmp_path / "private-models" / "prepared" / "model"
    root.mkdir(parents=True)
    (root / "model.safetensors").write_bytes(b"test checkpoint")
    (root / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ):
        (root / filename).write_text("{}")
    return root


@pytest.fixture()
def local_cuda_config_file(managed_cuda_config_file, prepared_bundle):
    data = yaml.safe_load(managed_cuda_config_file.read_text())
    data["roles"]["generation"]["model"] = str(prepared_bundle)
    data["startup"]["smoke_test_on_start"] = False
    managed_cuda_config_file.write_text(yaml.safe_dump(data))
    return managed_cuda_config_file


@pytest.fixture()
def stub_vllm(monkeypatch):
    """Stub only the external engine seams; exercise the real runtime adapter."""
    observed = SimpleNamespace(engine_args=[], downloads=[], network=[], model=None, shutdowns=0)
    monkeypatch.setenv("VLLM_BACKEND", "cuda")
    monkeypatch.setenv("SOVEREIGN_PROFILE", "cuda-x86_64")
    monkeypatch.delenv("SOVEREIGN_RUNTIME_API_KEY", raising=False)

    def forbid_network(*args, **kwargs):
        observed.network.append((args, kwargs))
        raise AssertionError("local model preparation attempted network access")

    def snapshot_download(*args, **kwargs):
        observed.downloads.append((args, kwargs))
        raise AssertionError("local model preparation attempted a Hub snapshot")

    def make_arg_parser(parser):
        for flag in (
            "--model",
            "--served-model-name",
            "--revision",
            "--tool-call-parser",
            "--reasoning-parser",
            "--gpu-memory-utilization",
        ):
            parser.add_argument(flag)
        parser.add_argument("--max-model-len", type=int)
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--max-num-seqs", type=int)
        for flag in [
            *VllmBackend.APPLIANCE_DEFAULT_FLAGS,
            "--enforce-eager",
            "--enable-auto-tool-choice",
        ]:
            parser.add_argument(flag, action="store_true")
        return parser

    def shutdown():
        observed.shutdowns += 1

    def from_engine_args(args):
        observed.engine_args.append(args)
        return SimpleNamespace(
            model_config=SimpleNamespace(
                model=observed.model or args.model, max_model_len=args.max_model_len
            ),
            shutdown=shutdown,
        )

    async def init_app_state(engine, state, args, supported_tasks):
        state.engine = engine

    modules = {
        "vllm": {"__version__": "0.25.0"},
        "vllm.engine.arg_utils": {
            "AsyncEngineArgs": SimpleNamespace(from_cli_args=lambda args: args)
        },
        "vllm.entrypoints.openai.api_server": {
            "build_app": lambda *args: FastAPI(),
            "init_app_state": init_app_state,
        },
        "vllm.entrypoints.openai.cli_args": {"make_arg_parser": make_arg_parser},
        "vllm.utils.argparse_utils": {"FlexibleArgumentParser": argparse.ArgumentParser},
        "vllm.v1.engine.async_llm": {
            "AsyncLLM": SimpleNamespace(from_engine_args=from_engine_args)
        },
        "huggingface_hub": {"snapshot_download": snapshot_download},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    return observed


def _cuda_appliance(config_file, monkeypatch):
    config = load_config(config_file)
    backend = VllmBackend()
    devices = config.roles.generation.accelerator_device_ids
    monkeypatch.setattr(
        backend,
        "accelerator",
        lambda: {
            "vendor": "nvidia",
            "device_count": len(devices),
            "unified_memory": False,
            "devices": [{"gpu_uuid": device} for device in devices],
        },
    )
    return Appliance(config_path=str(config_file), backend=backend)


def test_local_directory_launch_has_no_snapshot_or_network_fallback(
    local_cuda_config_file, prepared_bundle, stub_vllm, monkeypatch
):
    appliance = _cuda_appliance(local_cuda_config_file, monkeypatch)

    async def exercise():
        try:
            await appliance.run_lifecycle()
            # Even a direct call to the download seam must leave a local role alone.
            await appliance.backend._download(appliance.config.roles.generation)
        finally:
            await appliance.backend.shutdown()

    asyncio.run(exercise())
    assert appliance.state.state == "healthy"
    assert len(stub_vllm.engine_args) == 1
    args = stub_vllm.engine_args[0]
    assert args.model == str(prepared_bundle)
    assert args.served_model_name == "assistant-large"
    assert args.revision == "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
    assert args.tensor_parallel_size == 2
    assert args.enforce_eager is True
    assert stub_vllm.downloads == []
    assert stub_vllm.network == []


def test_local_manifest_observes_engine_path_without_public_identity_leak(
    local_cuda_config_file, prepared_bundle, stub_vllm, monkeypatch
):
    # The engine observation is intentionally distinct from requested input.
    stub_vllm.model = str(prepared_bundle.parent / "engine-observed")
    appliance = _cuda_appliance(local_cuda_config_file, monkeypatch)
    asyncio.run(appliance.run_lifecycle())
    try:
        client = TestClient(appliance.app)
        generation = client.get("/runtime/manifest").json()["roles"]["generation"]
        assert generation["engine_model"] == stub_vllm.model
        assert generation["served_model_name"] == "assistant-large"
        assert "model" not in generation
        public_models = client.get("/v1/models").json()
        assert public_models["data"] == [
            {"id": "assistant-large", "object": "model", "owned_by": "sovereign"}
        ]
        assert str(prepared_bundle.parent) not in json.dumps(public_models)
        assert str(prepared_bundle.parent) not in client.get("/health").text
    finally:
        asyncio.run(appliance.backend.shutdown())


@pytest.mark.parametrize(
    "missing",
    ["config.json", "tokenizer.json", "tokenizer_config.json", "processor_config.json"],
)
def test_missing_local_metadata_stays_alive_without_engine_or_download(
    local_cuda_config_file, prepared_bundle, stub_vllm, monkeypatch, missing
):
    (prepared_bundle / missing).unlink()
    appliance = _cuda_appliance(local_cuda_config_file, monkeypatch)
    asyncio.run(appliance.run_lifecycle())
    client = TestClient(appliance.app)
    assert appliance.state.state == "configuration_error"
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    errors = client.get("/runtime/errors").json()["errors"]
    assert errors[0]["code"] == "CONFIG_INVALID"
    assert str(prepared_bundle.parent) not in json.dumps(errors)
    assert stub_vllm.engine_args == []
    assert stub_vllm.downloads == []
    assert stub_vllm.network == []


@pytest.mark.parametrize(
    "extra", ["artifact", "model.pt", "model.bin", "adapter.safetensors", "unapproved.json"]
)
def test_local_bundle_rejects_extra_files(prepared_bundle, extra):
    (prepared_bundle / extra).write_bytes(b"unapproved")
    with pytest.raises(BackendStartError) as error:
        _validate_local_cuda_bundle(str(prepared_bundle))
    assert error.value.code == "CONFIG_INVALID"
    assert extra not in str(error.value)
    assert str(prepared_bundle) not in str(error.value)


@pytest.mark.parametrize("filename", ["model.safetensors", "tokenizer.json"])
@pytest.mark.parametrize("kind", ["empty", "directory", "symlink"])
def test_local_bundle_requires_nonempty_regular_files(prepared_bundle, filename, kind):
    target = prepared_bundle / filename
    target.unlink()
    if kind == "empty":
        target.touch()
    elif kind == "directory":
        target.mkdir()
    else:
        target.symlink_to(prepared_bundle / "config.json")
    with pytest.raises(BackendStartError, match="nonempty regular files"):
        _validate_local_cuda_bundle(str(prepared_bundle))


def test_local_cuda_rejects_primary_file_and_directory_symlink(prepared_bundle):
    with pytest.raises(BackendStartError, match="prepared directory"):
        _validate_local_cuda_bundle(str(prepared_bundle / "model.safetensors"))
    alias = prepared_bundle.parent / "linked-model"
    alias.symlink_to(prepared_bundle, target_is_directory=True)
    with pytest.raises(BackendStartError, match="prepared directory"):
        _validate_local_cuda_bundle(str(alias))


@pytest.mark.parametrize("exists", [False, True])
def test_local_cuda_requires_primary_safetensors(prepared_bundle, exists):
    (prepared_bundle / "model.safetensors").unlink()
    if exists:
        (prepared_bundle / "artifact").write_bytes(b"extensionless checkpoint")
    with pytest.raises(BackendStartError):
        _validate_local_cuda_bundle(str(prepared_bundle))


@pytest.mark.parametrize("contents", ["private-invalid-content", "[]", "null"])
def test_local_config_errors_do_not_echo_contents(prepared_bundle, contents):
    (prepared_bundle / "config.json").write_text(contents)
    with pytest.raises(BackendStartError) as error:
        _validate_local_cuda_bundle(str(prepared_bundle))
    assert error.value.code == "CONFIG_INVALID"
    assert contents not in str(error.value)
    assert str(prepared_bundle) not in str(error.value)


@pytest.mark.parametrize("excess", [0, 1])
def test_local_support_file_size_boundary(prepared_bundle, excess):
    with (prepared_bundle / "tokenizer.json").open("r+b") as support:
        support.truncate(64 * 1024 * 1024 + excess)
    if excess:
        with pytest.raises(BackendStartError, match="size limit"):
            _validate_local_cuda_bundle(str(prepared_bundle))
    else:
        _validate_local_cuda_bundle(str(prepared_bundle))


def test_local_support_total_size_is_bounded(prepared_bundle):
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ):
        with (prepared_bundle / filename).open("r+b") as support:
            support.truncate(64 * 1024 * 1024)
    with pytest.raises(BackendStartError, match="size limit"):
        _validate_local_cuda_bundle(str(prepared_bundle))


@pytest.mark.parametrize(
    "weight_map",
    [None, {}, [], {"weight": "missing.safetensors"}, {"weight": "../model.safetensors"}],
)
def test_local_bundle_rejects_stale_or_external_weight_index(prepared_bundle, weight_map):
    (prepared_bundle / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with pytest.raises(BackendStartError, match="weight index"):
        _validate_local_cuda_bundle(str(prepared_bundle))


def test_local_bundle_accepts_index_matching_sole_primary(prepared_bundle):
    (prepared_bundle / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"first": "model.safetensors", "second": "model.safetensors"}})
    )
    _validate_local_cuda_bundle(str(prepared_bundle))


@pytest.mark.parametrize(
    "missing", [None, "preprocessor_config.json", "video_preprocessor_config.json"]
)
def test_qwen_local_processor_assets_are_required(prepared_bundle, missing):
    (prepared_bundle / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
    for filename in ("preprocessor_config.json", "video_preprocessor_config.json"):
        (prepared_bundle / filename).write_text("{}")
    if missing:
        (prepared_bundle / missing).unlink()
        with pytest.raises(BackendStartError, match="processor metadata"):
            _validate_local_cuda_bundle(str(prepared_bundle))
    else:
        # Curated Fast omits its upstream stale index; direct discovery is valid.
        _validate_local_cuda_bundle(str(prepared_bundle))


def test_local_bundle_is_rechecked_before_engine_construction(
    local_cuda_config_file, prepared_bundle, stub_vllm
):
    role = load_config(local_cuda_config_file).roles.generation
    backend = VllmBackend()
    (prepared_bundle / "tokenizer.json").unlink()
    with pytest.raises(BackendStartError, match="missing required"):
        backend._construct_role_engine_sync("generation", role)
    assert stub_vllm.engine_args == []
    assert stub_vllm.downloads == []


@pytest.mark.parametrize(
    "flag",
    ["--model", "--served-model-name", "--revision", "--tensor-parallel-size", "--enforce-eager"],
)
def test_required_local_launch_flags_cannot_be_silently_dropped(
    local_cuda_config_file, stub_vllm, monkeypatch, flag
):
    module = sys.modules["vllm.entrypoints.openai.cli_args"]
    original = module.make_arg_parser

    def without_required_flag(parser):
        result = original(parser)
        del result._option_string_actions[flag]
        return result

    monkeypatch.setattr(module, "make_arg_parser", without_required_flag)
    role = load_config(local_cuda_config_file).roles.generation
    with pytest.raises(BackendStartError, match="required structured launch arguments"):
        VllmBackend()._construct_role_engine_sync("generation", role)
    assert stub_vllm.engine_args == []


@pytest.mark.parametrize("filename", ["model.gguf", "artifact"])
def test_legacy_metal_local_file_bypasses_cuda_bundle_validation(
    local_cuda_config_file, stub_vllm, monkeypatch, tmp_path, filename
):
    checkpoint = tmp_path / filename
    checkpoint.write_bytes(b"GGUF")
    data = yaml.safe_load(local_cuda_config_file.read_text())
    data["runtime"]["profile"] = "metal-arm64"
    generation = data["roles"]["generation"]
    generation["model"] = str(checkpoint)
    del generation["accelerator_device_ids"]
    del generation["tensor_parallel_size"]
    local_cuda_config_file.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("VLLM_BACKEND", "metal")
    role = load_config(local_cuda_config_file).roles.generation
    VllmBackend()._construct_role_engine_sync("generation", role)
    assert stub_vllm.engine_args[0].model == str(checkpoint)
    assert stub_vllm.downloads == []


def test_missing_local_directory_is_recoverable_before_engine_import(
    local_cuda_config_file, prepared_bundle, stub_vllm, monkeypatch
):
    data = yaml.safe_load(local_cuda_config_file.read_text())
    missing = prepared_bundle.parent / "unavailable-default-model"
    data["roles"]["generation"]["model"] = str(missing)
    local_cuda_config_file.write_text(yaml.safe_dump(data))
    monkeypatch.setitem(sys.modules, "vllm", None)
    appliance = _cuda_appliance(local_cuda_config_file, monkeypatch)
    asyncio.run(appliance.run_lifecycle())
    client = TestClient(appliance.app)
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503
    assert client.get("/runtime/manifest").json()["state"] == "configuration_error"
    errors = client.get("/runtime/errors").json()["errors"]
    assert errors[0]["code"] == "CONFIG_INVALID"
    assert str(missing) not in json.dumps(errors)
    assert stub_vllm.engine_args == []
    assert stub_vllm.downloads == []
    assert stub_vllm.network == []


def test_quiesce_awaits_engine_idle_future_before_checking_paused():
    backend = VllmBackend()

    async def exercise():
        pause_called, idle_ack = asyncio.Event(), asyncio.Event()
        calls = []

        async def pause_generation(*, mode, clear_cache):
            calls.append(("pause", mode, clear_cache))
            pause_called.set()
            await idle_ack.wait()

        async def is_paused():
            calls.append("is_paused")
            return True

        backend._apps["generation"] = SimpleNamespace(engine=SimpleNamespace(
            pause_generation=pause_generation, is_paused=is_paused,
        ))
        pending = asyncio.create_task(backend.quiesce())
        await asyncio.wait_for(pause_called.wait(), 2)
        assert backend.generation_paused is True
        assert not pending.done()
        assert calls == [("pause", "wait", False)], "paused state alone is not the idle ACK"
        idle_ack.set()
        await asyncio.wait_for(pending, 2)
        assert calls == [("pause", "wait", False), "is_paused"]
        assert backend.generation_paused is True

    asyncio.run(exercise())


def test_resume_reopens_only_after_engine_future_and_unpaused_ack():
    backend = VllmBackend()
    backend.generation_paused = True

    async def exercise():
        resume_called, resume_ack = asyncio.Event(), asyncio.Event()
        calls = []

        async def resume_generation():
            calls.append("resume")
            resume_called.set()
            await resume_ack.wait()

        async def is_paused():
            calls.append("is_paused")
            return False

        backend._apps["generation"] = SimpleNamespace(engine=SimpleNamespace(
            resume_generation=resume_generation, is_paused=is_paused,
        ))
        pending = asyncio.create_task(backend.resume())
        await asyncio.wait_for(resume_called.wait(), 2)
        assert backend.generation_paused is True
        assert not pending.done()
        assert calls == ["resume"]
        resume_ack.set()
        await asyncio.wait_for(pending, 2)
        assert calls == ["resume", "is_paused"]
        assert backend.generation_paused is False

    asyncio.run(exercise())


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
@pytest.mark.parametrize("ack", [None, "true", 0, 1, "opposite"])
def test_managed_engine_rejects_unknown_or_wrong_paused_ack(operation, ack):
    backend = VllmBackend()

    async def completed(**kwargs):
        return None

    async def is_paused():
        return operation == "resume" if ack == "opposite" else ack

    backend._apps["generation"] = SimpleNamespace(engine=SimpleNamespace(
        pause_generation=completed, resume_generation=completed, is_paused=is_paused,
    ))
    with pytest.raises(BackendStartError) as error:
        asyncio.run(getattr(backend, operation)())
    assert error.value.code == f"ENGINE_{operation.upper()}_FAILED"
    assert backend.generation_paused is True


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
@pytest.mark.parametrize("engine_state", ["missing", "legacy", "failed", "ack_failed"])
def test_managed_engine_fails_closed_without_usable_ack(operation, engine_state):
    backend = VllmBackend()

    async def fail(**kwargs):
        raise RuntimeError("private engine exception /weights/secret")

    async def completed(**kwargs):
        return None

    if engine_state == "missing":
        backend._native_lifetime_started = True
    if engine_state != "missing":
        engine = SimpleNamespace()
        if engine_state != "legacy":
            engine.pause_generation = fail if engine_state == "failed" else completed
            engine.resume_generation = fail if engine_state == "failed" else completed
            engine.is_paused = fail
        backend._apps["generation"] = SimpleNamespace(engine=engine)
    with pytest.raises(BackendStartError) as error:
        asyncio.run(getattr(backend, operation)())
    suffix = "UNAVAILABLE" if engine_state == "missing" else "FAILED"
    assert error.value.code == f"ENGINE_{operation.upper()}_{suffix}"
    assert "private" not in str(error.value)
    assert backend.generation_paused is True


@pytest.mark.parametrize("operation", ["quiesce", "resume"])
def test_cancelling_engine_control_never_invents_idle_or_resumed_state(operation):
    backend = VllmBackend()

    async def exercise():
        entered = asyncio.Event()
        cancelled = asyncio.Event()

        async def pending_ack(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        backend._apps["generation"] = SimpleNamespace(engine=SimpleNamespace(
            pause_generation=pending_ack, resume_generation=pending_ack,
        ))
        pending = asyncio.create_task(getattr(backend, operation)())
        await asyncio.wait_for(entered.wait(), 2)
        assert backend.generation_paused is True
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert cancelled.is_set()
        assert backend.generation_paused is True

    asyncio.run(exercise())


def test_selected_vllm_profile_applies_actual_scheduler_limit(local_cuda_config_file, stub_vllm):
    role = load_config(local_cuda_config_file).roles.generation.model_copy(update={
        "engine_profile_id": "selected-vllm-profile", "max_concurrent_requests": 7,
    })
    backend = VllmBackend()
    backend._construct_role_engine_sync("generation", role)
    assert stub_vllm.engine_args[0].max_num_seqs == 7
    assert "--max-num-seqs" not in backend._role_argv(
        "generation", role.model_copy(update={"engine_profile_id": None})
    )


def test_selected_vllm_profile_cannot_drop_scheduler_limit(local_cuda_config_file, stub_vllm, monkeypatch):
    module = sys.modules["vllm.entrypoints.openai.cli_args"]
    original = module.make_arg_parser

    def without_scheduler_limit(parser):
        result = original(parser)
        del result._option_string_actions["--max-num-seqs"]
        return result

    monkeypatch.setattr(module, "make_arg_parser", without_scheduler_limit)
    role = load_config(local_cuda_config_file).roles.generation.model_copy(update={
        "engine_profile_id": "selected-vllm-profile",
    })
    with pytest.raises(BackendStartError, match="required structured launch arguments"):
        VllmBackend()._construct_role_engine_sync("generation", role)
    assert stub_vllm.engine_args == []


@pytest.fixture()
def selected_generation(local_cuda_config_file, monkeypatch):
    role = load_config(local_cuda_config_file).roles.generation.model_copy(update={
        "engine_profile_id": "selected-vllm-profile",
    })
    model = SimpleNamespace(
        model=role.model, max_model_len=role.max_model_len,
        served_model_name=[role.served_model_name], quantization=None, dtype="torch.bfloat16",
    )
    config = SimpleNamespace(
        model_config=model,
        scheduler_config=SimpleNamespace(max_num_seqs=role.max_concurrent_requests),
        parallel_config=SimpleNamespace(tensor_parallel_size=role.tensor_parallel_size),
    )
    observed = SimpleNamespace(tasks=("generate",), shutdowns=0)

    async def get_supported_tasks():
        return observed.tasks

    async def shutdown():
        observed.shutdowns += 1

    engine = SimpleNamespace(
        model_config=model, vllm_config=config, get_supported_tasks=get_supported_tasks,
        shutdown=shutdown,
    )
    app = FastAPI()

    async def completion():
        return {}

    app.add_api_route("/v1/chat/completions", completion, methods=["POST"])
    app.add_api_route("/v1/completions", completion, methods=["POST"])

    async def init_app_state(*args):
        return None

    async def construct(name, requested_role):
        return (
            engine, SimpleNamespace(tensor_parallel_size=requested_role.tensor_parallel_size),
            lambda *args: app, init_app_state,
        )

    backend = VllmBackend()
    monkeypatch.setattr(backend, "_construct_role_engine", construct)
    return SimpleNamespace(
        backend=backend, role=role, engine=engine, config=config, model=model,
        app=app, observed=observed,
    )


@pytest.mark.parametrize("quantization,dtype,expected", [
    (None, "torch.bfloat16", "bfloat16"),
    (None, "torch.float16", "float16"),
    (None, "float32", "float32"),
    ("awq", "torch.float16", "awq"),
])
@pytest.mark.parametrize("alias_list", [False, True])
def test_selected_vllm_profile_publishes_only_loaded_execution(
    selected_generation, quantization, dtype, expected, alias_list
):
    selected = selected_generation
    selected.model.quantization = quantization
    selected.model.dtype = dtype
    selected.model.served_model_name = (
        [selected.role.served_model_name] if alias_list else selected.role.served_model_name
    )

    async def exercise():
        try:
            await selected.backend._start_role("generation", selected.role)
            info = selected.backend.role_info("generation")
            assert info.status == "healthy"
            assert info.engine_profile_id == selected.role.engine_profile_id
            assert info.engine_model == selected.model.model
            assert info.context_length == selected.model.max_model_len
            assert info.tensor_parallel_size == selected.config.parallel_config.tensor_parallel_size
            assert info.max_concurrent_requests == selected.config.scheduler_config.max_num_seqs
            assert info.quant == expected
            assert info.capabilities == ["chat_completions", "completions", "streaming", "text"]
            assert info.upstream_profile_id is None
            assert selected.backend.role_client("generation") is not None
        finally:
            await selected.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("owner,field,value", [
    ("model", "model", None),
    ("model", "model", "/different-loaded-model"),
    ("model", "max_model_len", None),
    ("model", "max_model_len", 999),
    ("model", "max_model_len", True),
    ("model", "served_model_name", None),
    ("model", "served_model_name", ["wrong-alias"]),
    ("model", "dtype", None),
    ("model", "dtype", "auto"),
    ("model", "quantization", ""),
    ("model", "quantization", 4),
    ("scheduler", "max_num_seqs", None),
    ("scheduler", "max_num_seqs", 999),
    ("scheduler", "max_num_seqs", True),
    ("parallel", "tensor_parallel_size", None),
    ("parallel", "tensor_parallel_size", 999),
    ("parallel", "tensor_parallel_size", True),
    ("config", "scheduler_config", None),
    ("config", "parallel_config", None),
    ("engine", "vllm_config", None),
    ("engine", "model_config", None),
])
def test_selected_vllm_profile_rejects_wrong_or_missing_loaded_facts(selected_generation, owner, field, value):
    selected = selected_generation
    owners = {
        "model": selected.model, "scheduler": selected.config.scheduler_config,
        "parallel": selected.config.parallel_config, "config": selected.config,
        "engine": selected.engine,
    }
    setattr(owners[owner], field, value)
    asyncio.run(selected.backend._start_role("generation", selected.role))
    info = selected.backend.role_info("generation")
    assert info.status == "unhealthy"
    assert info.error_code == "MODEL_LOAD_FAILED"
    assert info.engine_profile_id is None
    assert info.quant is None
    assert info.max_concurrent_requests is None
    assert info.capabilities is None
    assert selected.backend.role_client("generation") is None
    assert selected.observed.shutdowns == 1


@pytest.mark.parametrize("missing", ["tasks", "generate", "chat", "completion", "post"])
def test_selected_vllm_profile_requires_observed_generation_apis(selected_generation, missing):
    selected = selected_generation
    if missing == "tasks":
        selected.observed.tasks = None
    elif missing == "generate":
        selected.observed.tasks = ("embed",)
    elif missing == "post":
        for route in selected.app.routes:
            if route.path == "/v1/chat/completions":
                route.methods = {"GET"}
    else:
        path = "/v1/chat/completions" if missing == "chat" else "/v1/completions"
        selected.app.router.routes[:] = [route for route in selected.app.routes if route.path != path]
    asyncio.run(selected.backend._start_role("generation", selected.role))
    info = selected.backend.role_info("generation")
    assert info.status == "unhealthy"
    assert info.engine_profile_id is None
    assert info.capabilities is None
    assert selected.observed.shutdowns == 1


def test_unselected_vllm_retains_legacy_observation_behavior(selected_generation):
    selected = selected_generation
    selected.role = selected.role.model_copy(update={"engine_profile_id": None})
    selected.engine.vllm_config = None
    selected.observed.tasks = None
    selected.model.quantization = None
    selected.model.dtype = None

    async def exercise():
        try:
            await selected.backend._start_role("generation", selected.role)
            info = selected.backend.role_info("generation")
            assert info.status == "healthy"
            assert info.engine_profile_id is None
            assert info.max_concurrent_requests is None
            assert info.capabilities is None
            assert info.quant is None
        finally:
            await selected.backend.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("failure", ["local-input", "required-flag", "constructor", "app", "cleanup"])
def test_failed_candidate_api_quiesces_only_before_native_ownership(
    local_cuda_config_file, prepared_bundle, stub_vllm, monkeypatch, failure,
):
    monkeypatch.setenv("SOVEREIGN_RUNTIME_API_KEY", "secret")
    if failure == "local-input":
        (prepared_bundle / "tokenizer.json").unlink()
    elif failure == "required-flag":
        module = sys.modules["vllm.entrypoints.openai.cli_args"]
        original_parser = module.make_arg_parser

        def without_placement(parser):
            parser = original_parser(parser)
            del parser._option_string_actions["--tensor-parallel-size"]
            return parser

        monkeypatch.setattr(module, "make_arg_parser", without_placement)
    elif failure == "constructor":
        def failed_constructor(args):
            raise RuntimeError("core may have spawned before constructor failed")

        monkeypatch.setattr(sys.modules["vllm.v1.engine.async_llm"].AsyncLLM, "from_engine_args", failed_constructor)
    else:
        async def failed_app(*args):
            raise RuntimeError("app initialization failed after owned engine construction")

        monkeypatch.setattr(sys.modules["vllm.entrypoints.openai.api_server"], "init_app_state", failed_app)
        if failure == "cleanup":
            factory = sys.modules["vllm.v1.engine.async_llm"].AsyncLLM
            original_constructor = factory.from_engine_args

            def failed_cleanup():
                stub_vllm.shutdowns += 1
                raise RuntimeError("owned worker shutdown failed")

            def construct(args):
                engine = original_constructor(args)
                engine.shutdown = failed_cleanup
                return engine

            monkeypatch.setattr(factory, "from_engine_args", construct)
    appliance = _cuda_appliance(local_cuda_config_file, monkeypatch)

    async def exercise():
        await appliance.run_lifecycle()
        assert appliance.backend.role_client("generation") is None
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=appliance.app), base_url="http://runtime",
            headers={"Authorization": "Bearer secret"},
        ) as client:
            before = (await client.get("/runtime/manifest")).json()
            assert before["state"] == ("runtime_error" if failure == "cleanup" else "configuration_error")
            response = await client.post("/runtime/admin/generation/quiesce")
            no_owned_engine = failure in ("local-input", "required-flag")
            assert response.status_code == (200 if no_owned_engine else 503)
            if no_owned_engine:
                assert response.json() == {"quiesced": True}
            else:
                assert response.json()["error"]["code"] == "ENGINE_QUIESCE_FAILED"
            assert (await client.get("/runtime/manifest")).json()["generation_paused"] is True
            assert (await client.get("/health/live")).status_code == 200
            assert (await client.get("/health/ready")).status_code == 503
            assert (await client.post("/runtime/admin/generation/resume")).status_code == 503
            assert (await client.post("/v1/chat/completions", json={
                "model": appliance.config.roles.generation.served_model_name, "messages": [],
            })).status_code == 503
        if failure in ("app", "cleanup"):
            assert stub_vllm.shutdowns == 1
            assert "generation" in appliance.backend._engines
        else:
            assert stub_vllm.shutdowns == 0

    asyncio.run(exercise())


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("role_name", ["generation", "embedding"])
def test_partial_app_failure_cleans_every_returned_engine(selected_generation, monkeypatch, selected, role_name):
    sample = selected_generation
    role = sample.role.model_copy(update={"engine_profile_id": sample.role.engine_profile_id if selected else None})

    async def failed_app(*args):
        raise BackendStartError("MODEL_LOAD_FAILED", "post-constructor app failure")

    async def construct(name, requested_role):
        return sample.engine, SimpleNamespace(tensor_parallel_size=role.tensor_parallel_size), lambda *args: sample.app, failed_app

    monkeypatch.setattr(sample.backend, "_construct_role_engine", construct)

    async def exercise():
        with pytest.raises(BackendStartError, match="post-constructor app failure"):
            await sample.backend._start_role(role_name, role)
        assert sample.observed.shutdowns == 1
        assert sample.backend.role_client(role_name) is None
        assert sample.backend._engines[role_name] is sample.engine
        # Successful pinned cleanup is still not a worker-descendant receipt.
        with pytest.raises(BackendStartError):
            await sample.backend.quiesce()
        assert sample.backend.generation_paused is True

    asyncio.run(exercise())


def test_cancelled_offloop_constructor_is_collected_and_cleaned(local_cuda_config_file, stub_vllm, monkeypatch):
    backend = VllmBackend()
    role = load_config(local_cuda_config_file).roles.generation
    entered, release = threading.Event(), threading.Event()
    factory = sys.modules["vllm.v1.engine.async_llm"].AsyncLLM
    original_constructor = factory.from_engine_args

    def blocked_constructor(args):
        entered.set()
        assert release.wait(5)
        return original_constructor(args)

    monkeypatch.setattr(factory, "from_engine_args", blocked_constructor)

    async def exercise():
        pending = asyncio.create_task(backend._start_role("generation", role))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            pending.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            with pytest.raises(BackendStartError, match="construction has not settled"):
                await backend.quiesce()
            assert backend.generation_paused is True
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, 2)
        assert stub_vllm.shutdowns == 1
        assert backend.role_client("generation") is None
        assert "generation" in backend._engines
        with pytest.raises(BackendStartError):
            await backend.quiesce()

    asyncio.run(exercise())


def test_shutdown_failure_retains_every_engine_and_attempts_other_roles():
    backend = VllmBackend()
    calls = []

    def broken():
        calls.append("generation")
        raise RuntimeError("generation worker cleanup failed")

    def completed():
        calls.append("embedding")

    backend._engines = {"generation": SimpleNamespace(shutdown=broken), "embedding": SimpleNamespace(shutdown=completed)}
    backend._native_lifetime_started = True

    async def exercise():
        with pytest.raises(BackendStartError, match="cleanup is unconfirmed"):
            await backend.shutdown()
        assert calls == ["generation", "embedding"]
        assert set(backend._engines) == {"generation", "embedding"}
        with pytest.raises(BackendStartError, match="cleanup is unconfirmed"):
            await backend.quiesce()
        assert backend.generation_paused is True

    asyncio.run(exercise())


def test_failed_start_waits_for_shutdown_future_without_claiming_descendant_exit(selected_generation, monkeypatch):
    sample = selected_generation
    sample.model.max_model_len += 1

    async def exercise():
        cleanup = asyncio.get_running_loop().create_future()
        entered = asyncio.Event()

        def shutdown():
            entered.set()
            return cleanup

        monkeypatch.setattr(sample.engine, "shutdown", shutdown)
        pending = asyncio.create_task(sample.backend._start_role("generation", sample.role))
        await asyncio.wait_for(entered.wait(), 2)
        assert not pending.done()
        with pytest.raises(BackendStartError, match="cleanup is unconfirmed"):
            await sample.backend.quiesce()
        cleanup.set_result(None)
        await asyncio.wait_for(pending, 2)
        assert sample.backend.role_client("generation") is None
        with pytest.raises(BackendStartError, match="cleanup is unconfirmed"):
            await sample.backend.quiesce()
        assert sample.backend.generation_paused is True

    asyncio.run(exercise())
