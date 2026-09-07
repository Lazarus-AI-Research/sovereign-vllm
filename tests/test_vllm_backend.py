import argparse
import asyncio
import json
import socket
import sys
import time
from types import ModuleType, SimpleNamespace

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
    observed = SimpleNamespace(engine_args=[], downloads=[], network=[], model=None)
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
        for flag in [
            *VllmBackend.APPLIANCE_DEFAULT_FLAGS,
            "--enforce-eager",
            "--enable-auto-tool-choice",
        ]:
            parser.add_argument(flag, action="store_true")
        return parser

    def from_engine_args(args):
        observed.engine_args.append(args)
        return SimpleNamespace(
            model_config=SimpleNamespace(
                model=observed.model or args.model, max_model_len=args.max_model_len
            )
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
