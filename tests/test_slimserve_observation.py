"""Producer tests using only external engine/OS doubles, not fake first-party producers.

Native PyCFunction doubles test the observation boundary, not accelerator performance.
"""

import asyncio
import hashlib
import importlib
import importlib.machinery
import importlib.util
import json
import plistlib
import sys
import sysconfig
from pathlib import Path
from types import BuiltinFunctionType, ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import Response

from lazarus.appliance import slimserve_observation as observation
from lazarus.appliance import slimserve_provenance as provenance
from lazarus.appliance.config import RuntimeConfig

TOKEN = "private-test-token-" + "x" * 32
REVISION = "a" * 40
GPU_IDS = [f"GPU-00000000-0000-0000-0000-{rank + 1:012x}" for rank in range(4)]


@pytest.fixture()
def installation(tmp_path, monkeypatch):
    for name in tuple(sys.modules):
        if name in ("vllm", "slimserve", "lazarus.appliance.slimserve_worker") or name.startswith(("vllm.", "slimserve.")):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delattr(sys.modules["lazarus.appliance"], "slimserve_worker", raising=False)
    root = (tmp_path / "site-packages").resolve()
    root.mkdir()
    metadata = tmp_path / "provenance.json"
    files = {}

    def module(name, source=""):
        parts = name.split(".")
        for index in range(1, len(parts)):
            parent = ".".join(parts[:index])
            if parent not in sys.modules:
                package = root.joinpath(*parts[:index], "__init__.py")
                package.parent.mkdir(parents=True, exist_ok=True)
                package.write_text("")
                spec = importlib.util.spec_from_file_location(parent, package)
                value = importlib.util.module_from_spec(spec)
                monkeypatch.setitem(sys.modules, parent, value)
                files[package.relative_to(root).as_posix()] = ""
        path = root.joinpath(*parts).with_suffix(".py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, value)
        if len(parts) > 1:
            monkeypatch.setattr(sys.modules[".".join(parts[:-1])], parts[-1], value, raising=False)
        exec(compile(source, str(path), "exec"), value.__dict__)
        files[path.relative_to(root).as_posix()] = source
        return value

    profiles = root / "slimserve/profiles.json"
    profiles.parent.mkdir()
    profiles.write_text("{}")
    files["slimserve/profiles.json"] = "{}"
    extension_path = root / "vllm" / ("_quixicore_C" + importlib.machinery.EXTENSION_SUFFIXES[0])
    extension_path.parent.mkdir()
    extension_path.write_bytes(b"external native extension fixture")
    extension = ModuleType(provenance.KERNEL_MODULE)
    extension.__file__ = str(extension_path)
    loader = importlib.machinery.ExtensionFileLoader(provenance.KERNEL_MODULE, str(extension_path))
    extension.__spec__ = importlib.util.spec_from_file_location(
        provenance.KERNEL_MODULE, extension_path, loader=loader,
    )
    monkeypatch.setitem(sys.modules, provenance.KERNEL_MODULE, extension)
    native = Mock(spec=BuiltinFunctionType)
    native.__module__ = provenance.KERNEL_MODULE
    native.__name__ = "qgemm"
    native.side_effect = lambda tensor: tensor
    extension.qgemm = native
    debug = Mock(spec=BuiltinFunctionType)
    debug.__module__ = provenance.KERNEL_MODULE
    debug.__name__ = "dsv4_hash_router_debug"
    extension.dsv4_hash_router_debug = debug

    def seal(backend="cuda"):
        repository, commit = provenance.KERNEL_SOURCES[backend]
        artifacts = [{
            "path": extension_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
        }]
        if backend == "metal":
            library = extension_path.with_name("quixicore_metal.metallib")
            library.write_bytes(b"external Metal library fixture")
            artifacts.append({
                "path": library.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            })
        document = {
            "schema_version": 1,
            "slimserve": {
                "source_repository": "https://github.com/QuixiAI/SlimServe",
                "source_commit": provenance.SOURCE_COMMIT,
                "package_version": "test-external-engine",
                "files": [
                    {"path": name, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
                    for name in files
                ],
            },
            "kernels": {
                "library": f"quixicore-{backend}",
                "source_repository": f"https://github.com/QuixiAI/{repository}",
                "source_commit": commit,
                "modifications_commit": provenance.SOURCE_COMMIT,
                "version": f"{commit}+slimserve.{provenance.SOURCE_COMMIT}",
                "module_name": provenance.KERNEL_MODULE,
                "source_files": ["csrc/quixicore/reviewed-test-input.cu"],
                "artifacts": artifacts,
            },
        }
        metadata.write_text(json.dumps(document))
        provenance.installed_provenance.cache_clear()
        return document

    original_get_path = sysconfig.get_path
    monkeypatch.setattr(
        sysconfig, "get_path",
        lambda name, *a, **kw: str(root) if name == "purelib" else original_get_path(name, *a, **kw),
    )
    monkeypatch.setattr(provenance, "PROVENANCE_PATH", metadata)
    seal()
    yield SimpleNamespace(
        root=root, metadata=metadata, module=module, seal=seal, extension=extension,
        extension_path=extension_path, native=native, debug=debug,
    )
    provenance.installed_provenance.cache_clear()


def _config(count=1, backend="cuda"):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            model="/models/staged/test/" + REVISION + "/engine-profile/model",
            tokenizer="/models/staged/test/" + REVISION + "/engine-profile/model",
            revision=REVISION, served_model_name=["assistant"], max_model_len=2048,
            quantization=None, dtype="torch.float16",
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=count, pipeline_parallel_size=1, data_parallel_size=1,
            enable_expert_parallel=False,
            prefill_context_parallel_size=1, decode_context_parallel_size=1, world_size=count,
            worker_cls="lazarus.appliance.slimserve_worker." + (
                "MetalWorker" if backend == "metal" else "CudaWorker"
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        load_config=SimpleNamespace(load_format="safetensors"),
        compilation_config=SimpleNamespace(mode=0, cudagraph_mode="NONE", max_cudagraph_capture_size=0),
        attention_config=SimpleNamespace(use_trtllm_attention=False),
        speculative_config=None,
    )


@pytest.fixture()
def engine_workers(installation, monkeypatch):
    state = SimpleNamespace(rank=0, count=1, backend="cuda", invoke=True, capture=False)

    class Tensor:
        def __init__(self, device):
            self.device = SimpleNamespace(type="mps", index=0) if device.type == "mps" else device

        def numel(self):
            return 8

    torch = ModuleType("torch")
    torch.Tensor = Tensor
    torch.state = state
    torch.version = SimpleNamespace(hip=None)
    torch.cuda = SimpleNamespace(
        current_device=lambda: state.rank,
        device_count=lambda: state.count,
        is_available=lambda: True,
        get_device_capability=lambda rank: (8, 0),
        get_device_properties=lambda device: SimpleNamespace(
            uuid=GPU_IDS[getattr(device, "index", device)], name="NVIDIA A100",
        ),
        is_current_stream_capturing=lambda: state.capture,
        synchronize=Mock(),
    )
    torch.mps = SimpleNamespace(synchronize=Mock())
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    torch.distributed = SimpleNamespace(
        get_rank=lambda: state.rank, get_world_size=lambda: state.count,
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    installation.module("vllm.distributed", """
import torch
from types import SimpleNamespace

def get_tp_group():
    return SimpleNamespace(world_size=torch.state.count, rank_in_group=torch.state.rank)
""")
    installation.module("vllm.v1.worker.gpu_model_runner", """
import torch
import vllm._quixicore_C as native

class LoadedModel:
    def __init__(self, device):
        self.tensor = torch.Tensor(device)
    def parameters(self):
        return iter([self.tensor])
    def buffers(self):
        return iter(())

class GPUModelRunner:
    def __init__(self, device):
        self.device = device
        self.model = None
    def load_model(self, *, load_dummy_weights=False):
        self.model = LoadedModel(self.device)
        if torch.state.invoke:
            native.qgemm(self.model.tensor)
    def get_model(self):
        return self.model
""")
    installation.module("vllm.v1.worker.gpu_worker", """
import torch
from types import SimpleNamespace
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

class Worker:
    def __init__(self, vllm_config, local_rank, rank, distributed_init_method):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.local_rank = local_rank
        self.rank = rank
        self.device = SimpleNamespace(type=torch.state.backend, index=local_rank)
        if self.device.type == 'metal':
            self.device = SimpleNamespace(type='mps', index=None)
        self.model_runner = GPUModelRunner(self.device)
    def init_device(self):
        return None
    def load_model(self, *, load_dummy_weights=False):
        self.model_runner.load_model(load_dummy_weights=load_dummy_weights)
""")
    installation.module("vllm.v1.worker.metal_worker", """
from vllm.v1.worker.gpu_worker import Worker
class MetalWorker(Worker):
    pass
""")
    installation.seal()
    monkeypatch.delitem(sys.modules, "lazarus.appliance.slimserve_worker", raising=False)
    workers = importlib.import_module("lazarus.appliance.slimserve_worker")

    def make(count=1, backend="cuda", config=None):
        state.count, state.backend = count, backend
        config = config or _config(count, backend)
        result = []
        for rank in range(count):
            state.rank = rank
            cls = workers.MetalWorker if backend == "metal" else workers.CudaWorker
            worker = cls(config, rank, rank, "test-external-group")
            worker.load_model()
            result.append(worker)
        return config, result

    def rows(result):
        output = []
        for rank, worker in enumerate(result):
            state.rank = rank
            output.append(worker.sovereign_observation())
        return output

    yield SimpleNamespace(module=workers, make=make, rows=rows, state=state, torch=torch)
    workers.metal_platform_uuid.cache_clear()


def test_loaded_python_and_compiled_extension_are_digest_bound(installation):
    api = installation.module("vllm.audit_api", "def loaded_api():\n    return 1\n")
    installation.seal()
    installed = provenance.installed_provenance()
    installed.verify_api(api.loaded_api)
    assert installed.verify_extension(installation.extension, "cuda")["backend"] == "cuda"
    Path(api.__file__).write_text("def loaded_api():\n    return 200\n")
    with pytest.raises(provenance.ProvenanceError):
        installed.verify_api(api.loaded_api)


@pytest.mark.parametrize("mutation", ["missing", "digest", "source", "loader", "python"])
def test_wrong_or_missing_kernel_provenance_fails_closed(installation, mutation):
    document = installation.seal()
    if mutation == "missing":
        document["kernels"]["artifacts"] = []
    elif mutation == "digest":
        document["kernels"]["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "source":
        document["kernels"]["source_commit"] = "0" * 40
    elif mutation == "loader":
        installation.extension.__spec__.loader = None
    else:
        installation.extension.__file__ = str(installation.root / "vllm/fake.py")
    installation.metadata.write_text(json.dumps(document))
    with pytest.raises(provenance.ProvenanceError):
        provenance.installed_provenance().verify_extension(installation.extension, "cuda")


def test_missing_metal_library_cannot_be_observed(installation):
    installation.seal("metal")
    installation.extension_path.with_name("quixicore_metal.metallib").unlink()
    with pytest.raises((OSError, provenance.ProvenanceError)):
        provenance.installed_provenance().verify_extension(installation.extension, "metal")


def test_worker_records_execution_then_restores_native_callable(installation, engine_workers):
    config, workers = engine_workers.make()
    rows = engine_workers.rows(workers)
    assert rows[0]["ready"] is True
    assert rows[0]["kernel_entrypoint"] == "qgemm"
    assert rows[0]["config"] == observation.engine_config_facts(config)
    assert installation.extension.qgemm is installation.native
    assert installation.extension.dsv4_hash_router_debug is installation.debug
    installation.native.assert_called_once()
    engine_workers.torch.cuda.synchronize.assert_called_once()


def test_import_or_debug_call_is_not_native_inference_evidence(installation, engine_workers):
    engine_workers.state.invoke = False
    _, workers = engine_workers.make()
    installation.extension.dsv4_hash_router_debug()
    assert engine_workers.rows(workers) == [{"ready": False}]
    installation.native.assert_not_called()
    workers[0]._sovereign_kernel.restore()


def test_failed_kernel_call_never_records_success(installation, engine_workers):
    installation.native.side_effect = RuntimeError("external native launch failure")
    with pytest.raises(RuntimeError, match="external native"):
        engine_workers.make()
    assert installation.extension.qgemm is installation.native


def test_captured_dispatch_is_not_completed_execution(installation, engine_workers):
    engine_workers.state.capture = True
    _, workers = engine_workers.make()
    assert engine_workers.rows(workers) == [{"ready": False}]
    engine_workers.torch.cuda.synchronize.assert_not_called()
    engine_workers.state.capture = False
    installation.extension.qgemm(workers[0].model_runner.model.tensor)
    assert engine_workers.rows(workers)[0]["ready"] is True
    assert installation.extension.qgemm is installation.native


def test_replaced_model_is_nonready(installation, engine_workers):
    _, workers = engine_workers.make()
    workers[0].model_runner.model = SimpleNamespace()
    assert engine_workers.rows(workers) == [{"ready": False}]


@pytest.mark.parametrize("count", [1, 2, 4])
def test_exact_worker_count_order_and_placement(installation, engine_workers, count):
    config, workers = engine_workers.make(count)
    installed = provenance.installed_provenance()
    role = SimpleNamespace(tensor_parallel_size=count, accelerator_device_ids=GPU_IDS[:count])
    result = observation._workers(
        engine_workers.rows(workers), observation.engine_config_facts(config), role,
        installed, installed.verify_extension(installation.extension, "cuda"), "cuda",
    )
    assert [device["gpu_uuid"] for device in result["devices"]] == GPU_IDS[:count]


@pytest.mark.parametrize("mutation", [
    "missing", "extra", "order", "rank", "local_rank", "tensor_parallel_rank",
    "world_size", "visible_device_count", "model", "kernel", "debug", "identity", "bool_rank",
])
def test_worker_drift_is_nonready(installation, engine_workers, mutation):
    config, workers = engine_workers.make(2)
    rows = engine_workers.rows(workers)
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(rows[0])
    elif mutation == "order":
        rows.reverse()
    elif mutation in {"rank", "local_rank", "tensor_parallel_rank"}:
        rows[0][mutation] = 1
    elif mutation in {"world_size", "visible_device_count"}:
        rows[0][mutation] = 1
    elif mutation == "model":
        rows[0]["config"]["model"] = "/other/model"
    elif mutation == "kernel":
        rows[0]["kernels"]["version"] = "unverified"
    elif mutation == "debug":
        rows[0]["kernel_entrypoint"] = "dsv4_hash_router_debug"
    elif mutation == "identity":
        rows[1]["device"]["stable_identifier"] = GPU_IDS[0]
    else:
        rows[0]["rank"] = False
    installed = provenance.installed_provenance()
    with pytest.raises(provenance.ProvenanceError):
        observation._workers(
            rows, observation.engine_config_facts(config),
            SimpleNamespace(tensor_parallel_size=2, accelerator_device_ids=GPU_IDS[:2]),
            installed, installed.verify_extension(installation.extension, "cuda"), "cuda",
        )


def test_metal_identity_is_probed_not_inherited(installation, engine_workers, monkeypatch):
    installation.seal("metal")
    monkeypatch.setattr(sys, "platform", "darwin")
    host_uuid = "01234567-89AB-CDEF-0123-456789ABCDEF"
    os_probe = Mock(return_value=SimpleNamespace(stdout=plistlib.dumps([{"IOPlatformUUID": host_uuid}])))
    monkeypatch.setattr(engine_workers.module.subprocess, "run", os_probe)
    monkeypatch.setenv("SOVEREIGN_ACCELERATOR_DEVICE_IDS", "forged-device")
    config, workers = engine_workers.make(backend="metal")
    row = engine_workers.rows(workers)[0]
    expected = "apple-platform-integrated-gpu-v1:" + hashlib.sha256(host_uuid.lower().encode()).hexdigest()
    assert row["ready"] is True
    assert row["device"] == {
        "identity_kind": "apple_platform", "stable_identifier": expected, "platform_id": expected,
    }
    assert os_probe.call_args.args[0] == [
        "/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice", "-a",
    ]
    installed = provenance.installed_provenance()
    accelerator = observation._workers(
        [row], observation.engine_config_facts(config),
        SimpleNamespace(tensor_parallel_size=1, accelerator_device_ids=[]), installed,
        installed.verify_extension(installation.extension, "metal"), "metal",
    )
    assert accelerator["devices"][0]["platform_id"] == expected
    engine_workers.torch.mps.synchronize.assert_called_once()


@pytest.mark.parametrize("method,path", [
    ("GET", "/health"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("GET", "/v1/models"),
    ("GET", "/metrics"),
    ("GET", observation.OBSERVATION_PATH),
    ("POST", "/sovereign/quiesce"),
    ("POST", "/sovereign/resume"),
    ("GET", "/another-upstream-route"),
])
@pytest.mark.parametrize("headers", [
    pytest.param({}, id="missing"),
    pytest.param({"x-sovereign-observation-token": "forged-token-" + "z" * 32}, id="wrong"),
    pytest.param({"x-sovereign-observation-token": "short"}, id="malformed"),
    pytest.param({"x-sovereign-observation-token": "x" * 257}, id="oversized"),
    pytest.param([("x-sovereign-observation-token", TOKEN)] * 2, id="duplicate"),
    pytest.param([
        ("x-sovereign-observation-token", TOKEN),
        ("X-Sovereign-Observation-Token", "forged-token-" + "z" * 32),
    ], id="duplicate-mixed-case"),
])
def test_every_child_http_request_requires_private_token(monkeypatch, method, path, headers):
    monkeypatch.setenv(observation.TOKEN_ENV, TOKEN)
    app = FastAPI()
    app.middleware("http")(observation.observe_request)
    calls = []

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def upstream(request: Request):
        calls.append(request.url.path)
        return Response(b"upstream must not be reached")

    with TestClient(app) as client:
        response = client.request(method, path, headers=headers)
        assert response.status_code == 401
        assert response.content == b'{"error":"unauthorized"}'
        assert calls == []


@pytest.mark.parametrize("method,path,payload", [
    ("GET", "/health", b""),
    ("POST", "/v1/chat/completions", b'{"model":"private","messages":[]}'),
    ("POST", "/v1/completions", b'{"model":"private","prompt":"unchanged"}'),
    ("GET", "/v1/models", b""),
    ("GET", "/metrics", b""),
])
def test_authenticated_child_http_preserves_ordinary_serving(monkeypatch, method, path, payload):
    monkeypatch.setenv(observation.TOKEN_ENV, TOKEN)
    app = FastAPI()
    app.middleware("http")(observation.observe_request)
    calls = []

    async def upstream(request: Request):
        calls.append((request.method, request.url.path, request.url.query, request.headers["content-type"]))
        return Response(await request.body(), headers={"x-upstream": "unchanged"})

    app.add_api_route(path, upstream, methods=[method])
    with TestClient(app, headers={"x-sovereign-observation-token": TOKEN}) as client:
        # Role clients retain their private default credential when raw forwarding
        # supplies Content-Type; ordinary routes keep upstream body/query semantics.
        response = client.request(
            method, path + "?upstream=unchanged", content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 200
        assert response.content == payload
        assert response.headers["x-upstream"] == "unchanged"
        assert calls == [(method, path, "upstream=unchanged", "application/json")]


def test_authenticated_private_route_contracts(monkeypatch):
    monkeypatch.setenv(observation.TOKEN_ENV, TOKEN)
    app = FastAPI()
    app.middleware("http")(observation.observe_request)

    with TestClient(app, headers={"x-sovereign-observation-token": TOKEN}) as client:
        response = client.post(observation.OBSERVATION_PATH)
        assert response.status_code == 405
        assert response.headers["Allow"] == "GET"
        assert client.get(observation.OBSERVATION_PATH).status_code == 503
        for path in ("/sovereign/quiesce", "/sovereign/resume"):
            response = client.get(path)
            assert response.status_code == 405
            assert response.headers["Allow"] == "POST"
            assert client.post(path, json={"mode": "abort"}).status_code == 400
            assert client.post(path + "?mode=abort").status_code == 400


@pytest.fixture()
def resolved_profile(installation, engine_workers, tmp_path, monkeypatch):
    import platform

    from lazarus.appliance.slimserve_launch import resolve_local

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_IDS[0])
    root = (tmp_path / "prepared" / "model").resolve()
    root.mkdir(parents=True)
    contents = {
        "config.json": json.dumps({
            "model_type": "qwen4_exp", "dtype": "bfloat16",
            "text_config": {
                "model_type": "qwen4_exp_text", "head_dim": 256,
                "max_position_embeddings": 4096, "indexer_n_heads": 4,
                "indexer_head_dim": 128, "indexer_kv_heads": 1,
                "indexer_budget": 2048, "indexer_compress_ratio": 4,
                "rope_parameters": {
                    "rope_type": "default", "partial_rotary_factor": 0.25,
                    "mrope_section": [11, 11, 10], "mrope_interleaved": True,
                },
            },
        }).encode(),
        "tokenizer.json": b'{}',
        "tokenizer_config.json": b'{}',
        "model.safetensors": b'external checkpoint fixture',
    }
    for name, value in contents.items():
        (root / name).write_bytes(value)
    members = [
        {"file": name, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
        for name, value in contents.items()
    ]
    config = RuntimeConfig.model_validate({
        "schema_version": "1.2", "runtime": {"profile": "cuda-x86_64"},
        "roles": {"generation": {
            "enabled": True, "engine": "slimserve", "engine_profile_id": "engine-profile",
            "task": "generate", "source": "local", "model": str(root), "revision": REVISION,
            "served_model_name": "assistant", "max_model_len": 2048,
            "max_concurrent_requests": 2, "enforce_eager": True,
            "accelerator_device_ids": GPU_IDS[:1], "tensor_parallel_size": 1,
            "slimserve": {
                "source_commit": provenance.SOURCE_COMMIT,
                "profile_id": "reviewed-profile", "variant": "a100", "quant": "NVFP4",
                "artifacts": [{
                    "role": "model", "repository": "reviewed/model", "revision": REVISION,
                    "files": members,
                }],
            },
        }},
    })
    # Reduced external registry/engine doubles, not a runnable catalog tuple or
    # real checkpoint. Keep the Qwen source shape so the real resolver checks
    # the sealed indexer/RoPE inputs instead of taking the non-Qwen branch.
    registry = installation.module("slimserve.registry", """
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Plan:
    profile_id: str
    platform: str
    gpus: int
    source_key: str
    quant: object
    source: dict
    engine: dict
    env: dict
    speculative: bool
    speculative_overrides: dict
    variant_speculator: object = None
    @property
    def speculator(self):
        return self.variant_speculator
    @property
    def entry_file(self):
        return Path(self.source['local_dir'])

blocked = False
def profile_blocked(profile_id, platform):
    return 'blocked' if blocked else None
def resolve(profile_id, platform, gpus, quant, **kwargs):
    if (profile_id, platform, gpus, quant) != ('reviewed-profile', 'a100', 1, 'NVFP4'):
        raise ValueError('unknown external profile')
    return selected_plan
""")
    registry.selected_plan = registry.Plan(
        "reviewed-profile", "a100", 1, "qwen38-flash-next-nvfp4",
        SimpleNamespace(
            name="NVFP4", assembly=None, min_host_ram_bytes={},
            files=[{"path": entry["file"], "bytes": entry["size_bytes"], "sha256": entry["sha256"]}
                   for entry in members if entry["file"] == "model.safetensors"],
        ),
        {"repo": "reviewed/model", "revision": REVISION, "local_dir": "unused", "format": "safetensors"},
        {"tensor_parallel_size": 1, "max_model_len": 4096, "max_num_seqs": 2,
         "dtype": "bfloat16"}, {}, False, {},
    )
    installation.module("slimserve.engine", """
import json
def serve_argv(plan, host, port):
    argv = ['--model', str(plan.entry_file), '--host', host, '--port', str(port)]
    for key, value in plan.engine.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)])
    return argv
""")
    installation.module("slimserve.hardware", """
from types import SimpleNamespace
def detect():
    return SimpleNamespace(platform='a100', count=1, memory_bytes=0, host_ram_bytes=1 << 40)
def _classify(name):
    return 'a100'
""")
    installation.seal()
    local = resolve_local(config)
    identity = {
        "engine_profile_id": "engine-profile", "profile_id": "reviewed-profile",
        "variant": "a100", "quant": "NVFP4", "source_commit": provenance.SOURCE_COMMIT,
        "config": config.model_dump(exclude_none=True),
    }
    monkeypatch.setenv(observation.IDENTITY_ENV, json.dumps(identity))
    monkeypatch.setenv(observation.TOKEN_ENV, TOKEN)
    return SimpleNamespace(config=config, local=local, identity=identity, registry=registry)


@pytest.fixture()
def serving(installation, engine_workers, resolved_profile):
    local = resolved_profile.local
    config = _config()
    config.model_config.model = local.model_path
    config.model_config.tokenizer = local.tokenizer_path
    config.model_config.dtype = "torch.bfloat16"
    client_module = installation.module("vllm.v1.engine.async_llm", """
import torch

class AsyncLLM:
    def __init__(self, config, workers):
        self.vllm_config = config
        self.workers = workers
        self.tasks = ('generate',)
        self.paused = False
        self.pause_args = None
        self.pause_ack = None
        self.bad_ack = False
        self.pause_started = None
        self.rows_override = None
    async def get_supported_tasks(self):
        return self.tasks
    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        assert (method, timeout, args, kwargs) == ('sovereign_observation', 5, (), None)
        if self.rows_override is not None:
            return self.rows_override
        result = []
        for rank, worker in enumerate(self.workers):
            torch.state.rank = rank
            result.append(worker.sovereign_observation())
        return result
    async def pause_generation(self, *, mode='abort', clear_cache=True):
        self.pause_args = (mode, clear_cache)
        if self.pause_started is not None:
            self.pause_started.set()
        if self.pause_ack is not None:
            await self.pause_ack.wait()
        if not self.bad_ack:
            self.paused = True
    async def resume_generation(self):
        if not self.bad_ack:
            self.paused = False
    async def is_paused(self):
        return self.paused
""")
    server = installation.module("vllm.entrypoints.openai.api_server", """
import importlib
import inspect
from fastapi import FastAPI
def build_app(args):
    app = FastAPI()
    app.state.args = args
    for name in args.middleware:
        path, attribute = name.rsplit('.', 1)
        function = getattr(importlib.import_module(path), attribute)
        assert inspect.iscoroutinefunction(function)
        app.middleware('http')(function)
    return app
async def init_app_state(client, state):
    state.engine_client = client
    state.vllm_config = client.vllm_config
""")
    routes = []
    for package, function, path in (
        ("chat_completion", "create_chat_completion", "/v1/chat/completions"),
        ("completion", "create_completion", "/v1/completions"),
    ):
        route = installation.module(f"vllm.entrypoints.openai.{package}.api_router", f"""
from fastapi import APIRouter
router = APIRouter()
class Handler:
    async def {function}(self):
        return None
@router.post({path!r})
async def {function}():
    return {{'external': 'generation route'}}
""")
        routes.append(route)
    installation.seal()
    _, workers = engine_workers.make(config=config)
    client = client_module.AsyncLLM(config, workers)
    args = SimpleNamespace(**local.engine, middleware=[
        "lazarus.appliance.slimserve_observation.observe_request",
    ])
    args.served_model_name = ["assistant"]
    app = server.build_app(args)
    asyncio.run(server.init_app_state(client, app.state))
    app.state.openai_serving_chat = routes[0].Handler()
    app.state.openai_serving_completion = routes[1].Handler()
    for route in routes:
        app.include_router(route.router)
    return SimpleNamespace(
        app=app, client=client, config=config, workers=workers,
        profile=resolved_profile, headers={"x-sovereign-observation-token": TOKEN},
    )


def test_private_route_produces_only_corroborated_facts(serving):
    with TestClient(serving.app) as client:
        response = client.get(observation.OBSERVATION_PATH, headers=serving.headers)
        assert response.status_code == 200, response.text
        value = response.json()
        assert value["engine"]["version"] == provenance.SOURCE_COMMIT
        assert value["engine_profile_id"] == "engine-profile"
        assert value["upstream_profile_id"] == "reviewed-profile"
        assert value["engine_model"] == serving.profile.local.model_path
        assert value["quant"] == "NVFP4"
        assert value["capabilities"] == ["chat_completions", "completions", "streaming", "text"]
        assert value["accelerator"]["devices"][0]["gpu_uuid"] == GPU_IDS[0]
        assert set(value) == {
            "engine", "kernels", "engine_profile_id", "upstream_profile_id", "quant",
            "engine_model", "revision", "context_length", "max_concurrent_requests",
            "device_count", "tensor_parallel_size", "accelerator", "capabilities",
        }


@pytest.mark.parametrize("mutation", ["profile", "blocked", "config", "tasks", "handler", "rows", "digest"])
def test_private_route_missing_or_inconsistent_facts_is_503(serving, installation, monkeypatch, mutation):
    if mutation == "profile":
        identity = dict(serving.profile.identity, profile_id="forged-profile")
        monkeypatch.setenv(observation.IDENTITY_ENV, json.dumps(identity))
    elif mutation == "blocked":
        serving.profile.registry.blocked = True
    elif mutation == "config":
        serving.config.model_config.model = "/forged/model"
    elif mutation == "tasks":
        serving.client.tasks = ("embed",)
    elif mutation == "handler":
        serving.app.state.openai_serving_chat = None
    elif mutation == "rows":
        serving.client.rows_override = []
    else:
        installation.extension_path.write_bytes(b"replaced extension")
    with TestClient(serving.app) as client:
        response = client.get(observation.OBSERVATION_PATH, headers=serving.headers)
        assert response.status_code == 503
        assert response.json() == {"error": "observation_unavailable"}


def test_profile_query_parameters_do_not_override_resolved_identity(serving):
    with TestClient(serving.app) as client:
        response = client.get(
            observation.OBSERVATION_PATH + "?profile_id=forged&quant=forged", headers=serving.headers,
        )
        assert response.status_code == 200, response.text
        assert response.json()["upstream_profile_id"] == "reviewed-profile"
        assert response.json()["quant"] == "NVFP4"


def test_private_quiescence_requires_actual_wait_ack_and_explicit_resume(serving):
    import httpx

    async def exercise():
        serving.client.pause_ack = asyncio.Event()
        serving.client.pause_started = asyncio.Event()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=serving.app), base_url="http://private") as client:
            pending = asyncio.create_task(client.post("/sovereign/quiesce", headers=serving.headers))
            # The external engine method itself exposes when it received pause;
            # no first-party control or producer implementation is replaced.
            await asyncio.wait_for(serving.client.pause_started.wait(), timeout=5)
            assert serving.client.pause_args == ("wait", False)
            assert not pending.done()
            serving.client.pause_ack.set()
            result = await pending
            assert result.status_code == 200
            assert result.json() == {"quiesced": True}
            assert serving.client.paused is True
            result = await client.post("/sovereign/resume", headers=serving.headers)
            assert result.status_code == 200
            assert result.json() == {"resumed": True}
            assert serving.client.paused is False

    asyncio.run(exercise())


def test_false_engine_pause_ack_is_nonready(serving):
    serving.client.bad_ack = True
    with TestClient(serving.app) as client:
        response = client.post("/sovereign/quiesce", headers=serving.headers)
        assert response.status_code == 503
        assert serving.client.pause_args == ("wait", False)


def test_loaded_config_mutation_cannot_relabel_existing_weights(engine_workers):
    config, workers = engine_workers.make()
    config.model_config.model = "/different/checkpoint"
    assert engine_workers.rows(workers) == [{"ready": False}]


def test_mps_unindexed_worker_matches_actual_index_zero_tensor(installation, engine_workers):
    installation.seal("metal")
    _, workers = engine_workers.make(backend="metal")
    worker = workers[0]
    assert worker.device.index is None
    assert worker.model_runner.get_model().tensor.device.index == 0
    assert worker._sovereign_kernel.evidence()["kernel_entrypoint"] == "qgemm"


def test_worker_artifact_digest_mismatch_is_rejected(installation, engine_workers):
    config, workers = engine_workers.make()
    rows = engine_workers.rows(workers)
    rows[0]["kernel_artifacts"][rows[0]["kernel_module"]] = "0" * 64
    installed = provenance.installed_provenance()
    with pytest.raises(provenance.ProvenanceError):
        observation._workers(
            rows, observation.engine_config_facts(config),
            SimpleNamespace(tensor_parallel_size=1, accelerator_device_ids=GPU_IDS[:1]),
            installed, installed.verify_extension(installation.extension, "cuda"), "cuda",
        )


def test_speculative_snapshot_excludes_unrelated_computed_objects():
    from lazarus.appliance.slimserve_launch import _SPEC_KEYS

    config = _config()
    selected = dict.fromkeys(_SPEC_KEYS)
    selected.update(method="mtp", num_speculative_tokens=1)
    config.speculative_config = SimpleNamespace(
        **selected, model=config.model_config.model, revision=None,
        draft_model_config=SimpleNamespace(model=config.model_config.model, revision=REVISION),
        draft_load_config=object(), target_model_config=object(),
    )
    facts = observation.engine_config_facts(config)
    assert facts["speculative_config"]["resolved_draft_model"] == config.model_config.model
    assert facts["speculative_config"]["resolved_draft_revision"] == REVISION
    assert "draft_load_config" not in facts["speculative_config"]
    assert "target_model_config" not in facts["speculative_config"]
    json.dumps(facts)


def test_loaded_alias_snapshot_does_not_follow_mutable_config(engine_workers):
    config, workers = engine_workers.make()
    config.model_config.served_model_name.append("forged-alias")
    assert engine_workers.rows(workers) == [{"ready": False}]


@pytest.mark.parametrize("mode,graph", [(3, "NONE"), (0, "FULL_DECODE_ONLY")])
def test_requested_eager_requires_actual_compilation_and_graph_disable(serving, mode, graph):
    serving.config.compilation_config.mode = mode
    serving.config.compilation_config.cudagraph_mode = graph
    with TestClient(serving.app) as client:
        response = client.get(observation.OBSERVATION_PATH, headers=serving.headers)
        assert response.status_code == 503


@pytest.mark.parametrize("module_name", provenance.FORBIDDEN_OPTIONAL_MODULES)
def test_optional_flashinfer_importability_rejected_without_import(installation, monkeypatch, module_name):
    class OptionalFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == module_name:
                return importlib.machinery.ModuleSpec(fullname, self)
            return None

        def create_module(self, spec):
            raise AssertionError("optional native package must never be imported")

        def exec_module(self, module):
            raise AssertionError("optional native package must never be executed")

    monkeypatch.setattr(sys, "meta_path", [OptionalFinder(), *sys.meta_path])
    with pytest.raises(provenance.ProvenanceError, match="optional native"):
        provenance.installed_provenance()
    assert module_name not in sys.modules


def test_live_provenance_rejects_later_optional_import(installation, monkeypatch):
    installed = provenance.installed_provenance()
    monkeypatch.setitem(sys.modules, "flashinfer", ModuleType("flashinfer"))
    with pytest.raises(provenance.ProvenanceError, match="optional native"):
        installed.verify_extension(installation.extension, "cuda")


@pytest.mark.parametrize("value", [True, None, 0])
def test_actual_attention_policy_must_be_explicit_false(serving, value):
    serving.config.attention_config.use_trtllm_attention = value
    with TestClient(serving.app) as client:
        response = client.get(observation.OBSERVATION_PATH, headers=serving.headers)
        assert response.status_code == 503
