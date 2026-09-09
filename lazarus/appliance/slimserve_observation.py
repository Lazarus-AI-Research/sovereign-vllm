"""Authenticated private observations for the pinned upstream API server.

SlimServe's --middleware accepts this coroutine directly via app.middleware.
Ordinary traffic is passed through without reading or modifying its payload.
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import inspect
import os
import re
import sys
from enum import Enum
from pathlib import Path

from starlette.responses import JSONResponse

from lazarus.appliance.config import RuntimeConfig
from lazarus.appliance.slimserve_provenance import (
    KERNEL_ENTRYPOINTS,
    KERNEL_MODULE,
    SOURCE_COMMIT,
    ProvenanceError,
    bounded_json,
    installed_provenance,
)

OBSERVATION_PATH = "/sovereign/observation"
TOKEN_ENV = "SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN"
IDENTITY_ENV = "SOVEREIGN_SLIMSERVE_IDENTITY"
_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}\Z")
_CUDA_ID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_METAL_ID = re.compile(r"apple-platform-integrated-gpu-v1:[0-9a-f]{64}\Z")


def _integer(value, minimum=1, maximum=1048576) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ProvenanceError("missing or invalid applied integer")
    return value


def _boolean(value) -> bool:
    if type(value) is not bool:
        raise ProvenanceError("missing or invalid applied boolean")
    return value


def _plain(value):
    if isinstance(value, Enum):
        return _plain(value.value)
    if value is None or type(value) in (str, int, bool, float):
        return value
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _plain(item) for key, item in value.items()}
    raise ProvenanceError("unknown applied configuration value")


def _matches(actual, expected) -> bool:
    if isinstance(expected, dict):
        return all(
            _matches(actual.get(key) if isinstance(actual, dict) else getattr(actual, key), value)
            for key, value in expected.items()
        )
    if isinstance(actual, Enum):
        return expected in (actual.name, actual.value)
    if isinstance(expected, (list, tuple)):
        return isinstance(actual, (list, tuple)) and len(actual) == len(expected) and all(
            _matches(left, right) for left, right in zip(actual, expected)
        )
    if isinstance(expected, bool):
        return actual is expected
    return actual == expected


def engine_config_facts(config) -> dict:
    """Snapshot the same applied VllmConfig in the API process and each worker."""
    model, parallel = config.model_config, config.parallel_config
    compilation = config.compilation_config
    graph_mode = compilation.cudagraph_mode
    graph_mode = graph_mode.name if isinstance(graph_mode, Enum) else graph_mode
    if graph_mode not in {"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}:
        raise ProvenanceError("unknown applied graph mode")
    speculative = config.speculative_config
    spec = None
    if speculative is not None:
        from lazarus.appliance.slimserve_launch import _SPEC_KEYS

        spec = {key: _plain(getattr(speculative, key)) for key in _SPEC_KEYS | {"model", "revision"}}
        # AttentionBackendEnum.value is a class path; profile flags use its name.
        attention = speculative.attention_backend
        if isinstance(attention, Enum):
            spec["attention_backend"] = attention.name
        draft = speculative.draft_model_config
        if draft is None:
            raise ProvenanceError("speculation has no loaded draft model configuration")
        spec["resolved_draft_model"] = draft.model
        spec["resolved_draft_revision"] = draft.revision
    aliases = model.served_model_name
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list) or not aliases or not all(isinstance(v, str) for v in aliases):
        raise ProvenanceError("missing applied serving aliases")
    return {
        "model": model.model,
        "tokenizer": model.tokenizer,
        "revision": model.revision,
        "served_model_name": list(aliases),
        "max_model_len": _integer(model.max_model_len),
        "max_num_seqs": _integer(config.scheduler_config.max_num_seqs, maximum=256),
        "tensor_parallel_size": _integer(parallel.tensor_parallel_size, maximum=4),
        "pipeline_parallel_size": _integer(parallel.pipeline_parallel_size),
        "data_parallel_size": _integer(parallel.data_parallel_size),
        "enable_expert_parallel": _boolean(parallel.enable_expert_parallel),
        "prefill_context_parallel_size": _integer(parallel.prefill_context_parallel_size),
        "decode_context_parallel_size": _integer(parallel.decode_context_parallel_size),
        "world_size": _integer(parallel.world_size, maximum=4),
        "worker_cls": parallel.worker_cls,
        "compilation_mode": _integer(_plain(compilation.mode), minimum=0, maximum=3),
        "cudagraph_mode": graph_mode,
        "max_cudagraph_capture_size": _integer(
            compilation.max_cudagraph_capture_size, minimum=0,
        ),
        "quantization": model.quantization,
        "dtype": str(model.dtype).removeprefix("torch."),
        "load_format": _plain(config.load_config.load_format),
        "use_trtllm_attention": _boolean(config.attention_config.use_trtllm_attention),
        "speculative_config": spec,
    }


def _selection():
    from slimserve import registry

    from lazarus.appliance.slimserve_launch import resolve_local

    provenance = installed_provenance()
    provenance.verify_module(registry)
    provenance.verify_api(registry.resolve)
    provenance.verify_api(registry.profile_blocked)
    provenance.verify_file(provenance.root / "slimserve/profiles.json")
    identity = bounded_json(os.environ.get(IDENTITY_ENV, ""), 1024 * 1024)
    config = RuntimeConfig.model_validate(identity.get("config"))
    role = config.roles.generation
    selected = role.slimserve
    if not role.enabled or role.engine != "slimserve" or selected is None:
        raise ProvenanceError("no selected generation profile")
    expected = {
        "engine_profile_id": role.engine_profile_id,
        "profile_id": selected.profile_id,
        "variant": selected.variant,
        "quant": selected.quant,
        "source_commit": SOURCE_COMMIT,
    }
    if set(identity) != set(expected) | {"config"} or any(
        identity.get(key) != value for key, value in expected.items()
    ):
        raise ProvenanceError("private profile identity disagrees with resolved configuration")
    if registry.profile_blocked(selected.profile_id, selected.variant):
        raise ProvenanceError("selected upstream profile is blocked")
    local = resolve_local(config, verify_files=False)
    plan = local.plan
    if (
        plan.profile_id != selected.profile_id
        or plan.platform != selected.variant
        or plan.quant.name != selected.quant
        or type(plan.gpus) is not int
        or plan.gpus != role.tensor_parallel_size
    ):
        raise ProvenanceError("resolved upstream tuple disagrees with selected profile")
    return role, local, provenance


def _verify_applied(state, role, local, facts):
    backend = "metal" if role.slimserve.variant == "metal" else "cuda"
    worker_name = "MetalWorker" if backend == "metal" else "CudaWorker"
    expected = {
        "model": local.model_path,
        "tokenizer": local.tokenizer_path,
        "revision": role.revision,
        "served_model_name": [role.served_model_name],
        "max_model_len": role.max_model_len,
        "max_num_seqs": role.max_concurrent_requests,
        "tensor_parallel_size": role.tensor_parallel_size,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "enable_expert_parallel": local.engine.get("enable_expert_parallel", False),
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "world_size": role.tensor_parallel_size,
        "worker_cls": f"lazarus.appliance.slimserve_worker.{worker_name}",
        "use_trtllm_attention": False,
    }
    if any(not _matches(facts.get(key), value) for key, value in expected.items()):
        raise ProvenanceError("loaded engine configuration differs from the resolved plan")
    # This source pin has no ModelConfig.enforce_eager field. The first-party
    # launch translates that runtime request to the documented compilation
    # mode NONE plus CUDA graph mode NONE, and workers report those facts.
    if role.enforce_eager and (
        facts["compilation_mode"] != 0 or facts["cudagraph_mode"] != "NONE"
        or facts["max_cudagraph_capture_size"] != 0
    ):
        raise ProvenanceError("requested eager execution was not applied")
    authored_compilation = local.engine.get("compilation_config", {})
    for key in ("mode", "cudagraph_mode"):
        if key in authored_compilation and not _matches(
            getattr(state.vllm_config.compilation_config, key), authored_compilation[key],
        ):
            raise ProvenanceError("effective compilation mode differs from reviewed profile")
    for key, expected_value in local.engine.items():
        actual = getattr(state.args, key)
        if key == "served_model_name" and isinstance(expected_value, str):
            expected_value = [expected_value]
        if not _matches(actual, expected_value):
            raise ProvenanceError("reviewed profile override was not applied")
    for key in ("quantization", "load_format", "dtype"):
        if key in local.engine and local.engine[key] != "auto" and facts[key] != local.engine[key]:
            raise ProvenanceError("resolved model format differs from profile")
    spec = facts["speculative_config"]
    if local.speculative_config is None:
        if spec is not None:
            raise ProvenanceError("unexpected speculative engine")
    else:
        if spec is None or any(
            not _matches(spec.get(key), value) for key, value in local.speculative_config.items()
        ):
            raise ProvenanceError("speculative engine settings differ from the resolved plan")
        if spec["resolved_draft_model"] != local.draft_path:
            raise ProvenanceError("resolved drafter path differs from loaded draft")
    if backend == "metal" and role.tensor_parallel_size != 1:
        raise ProvenanceError("Metal requires exactly one worker")
    return backend


async def _capabilities(app, client, provenance) -> list[str]:
    provenance.verify_api(client.get_supported_tasks)
    tasks = await asyncio.wait_for(client.get_supported_tasks(), timeout=5)
    if not isinstance(tasks, (tuple, list)) or "generate" not in tasks:
        raise ProvenanceError("engine has no generation task")
    routes = (
        ("/v1/chat/completions", "chat_completion", "create_chat_completion", "openai_serving_chat"),
        ("/v1/completions", "completion", "create_completion", "openai_serving_completion"),
    )
    for path, package, function, attribute in routes:
        module = importlib.import_module(f"vllm.entrypoints.openai.{package}.api_router")
        endpoint = getattr(module, function)
        provenance.verify_api(endpoint)
        matching = [route for route in app.routes if getattr(route, "path", None) == path]
        if (
            len(matching) != 1
            or "POST" not in matching[0].methods
            or inspect.unwrap(matching[0].endpoint) is not inspect.unwrap(endpoint)
        ):
            raise ProvenanceError("audited generation route is not registered")
        handler = getattr(app.state, attribute)
        if handler is None:
            raise ProvenanceError("generation route has no initialized handler")
        provenance.verify_api(getattr(handler, function))
        provenance.verify_api(type(handler))
    # Both pinned routes return StreamingResponse for their streaming generator;
    # their registered implementations and engine generation task were verified.
    return ["chat_completions", "completions", "streaming", "text"]


def _workers(rows, facts, role, provenance, kernels, backend):
    count = role.tensor_parallel_size
    if not isinstance(rows, list) or len(rows) != count or count not in (1, 2, 4):
        raise ProvenanceError("incomplete worker RPC response")
    devices = []
    module_path = Path(sys.modules[KERNEL_MODULE].__file__).relative_to(provenance.root).as_posix()
    for rank, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("ready") is not True:
            raise ProvenanceError("worker is not ready")
        for key in ("rank", "local_rank", "tensor_parallel_rank"):
            if _integer(row.get(key), minimum=0, maximum=3) != rank:
                raise ProvenanceError("worker rank order drift")
        if (
            _integer(row.get("world_size"), maximum=4) != count
            or _integer(row.get("visible_device_count"), maximum=4) != count
            or row.get("engine") != provenance.engine
            or row.get("kernels") != kernels
            or row.get("kernel_entrypoint") not in KERNEL_ENTRYPOINTS
            or row.get("kernel_module") != module_path
            or row.get("kernel_artifacts") != provenance.artifacts
            or row.get("config") != facts
        ):
            raise ProvenanceError("worker implementation or applied configuration drift")
        device = row.get("device")
        if not isinstance(device, dict):
            raise ProvenanceError("worker has no physical device identity")
        identity = device.get("stable_identifier")
        if not isinstance(identity, str):
            raise ProvenanceError("worker device identity is missing")
        if backend == "cuda":
            if (
                not _CUDA_ID.fullmatch(identity)
                or device.get("identity_kind") != "nvidia_gpu_uuid"
                or device.get("gpu_uuid") != identity
                or type(device.get("local_rank")) is not int
                or device["local_rank"] != rank
                or set(device) != {"identity_kind", "stable_identifier", "gpu_uuid", "local_rank"}
            ):
                raise ProvenanceError("invalid CUDA physical identity")
        elif (
            count != 1
            or not _METAL_ID.fullmatch(identity)
            or device.get("identity_kind") != "apple_platform"
            or device.get("platform_id") != identity
            or set(device) != {"identity_kind", "stable_identifier", "platform_id"}
        ):
            raise ProvenanceError("invalid native Metal physical identity")
        devices.append(device)
    identities = [device["stable_identifier"] for device in devices]
    if len(set(identities)) != count or (
        backend == "cuda" and identities != role.accelerator_device_ids
    ):
        raise ProvenanceError("worker physical placement differs from selected ordered devices")
    return {
        "vendor": "apple" if backend == "metal" else "nvidia",
        "device_count": count,
        "unified_memory": backend == "metal",
        "devices": devices,
    }


async def collect_observation(app) -> dict:
    role, local, provenance = await asyncio.to_thread(_selection)
    state, client = app.state, app.state.engine_client
    if state.vllm_config is not client.vllm_config:
        raise ProvenanceError("API and engine configuration differ")
    server = sys.modules.get("__main__")
    if getattr(getattr(server, "__spec__", None), "name", None) != "vllm.entrypoints.openai.api_server":
        server = importlib.import_module("vllm.entrypoints.openai.api_server")
    provenance.verify_api(server.build_app)
    provenance.verify_api(server.init_app_state)
    provenance.verify_api(client.collective_rpc)
    facts = engine_config_facts(client.vllm_config)
    backend = _verify_applied(state, role, local, facts)
    extension = importlib.import_module(KERNEL_MODULE)
    kernels = provenance.verify_extension(extension, backend)
    capabilities = await _capabilities(app, client, provenance)
    rows = await asyncio.wait_for(
        client.collective_rpc("sovereign_observation", timeout=5, args=(), kwargs=None),
        timeout=6,
    )
    accelerator = _workers(rows, facts, role, provenance, kernels, backend)
    await asyncio.to_thread(provenance.verify_loaded_modules)
    return {
        "engine": provenance.engine,
        "kernels": kernels,
        "engine_profile_id": role.engine_profile_id,
        "upstream_profile_id": local.plan.profile_id,
        "quant": local.plan.quant.name,
        "engine_model": facts["model"],
        "revision": facts["revision"],
        "context_length": facts["max_model_len"],
        "max_concurrent_requests": facts["max_num_seqs"],
        "device_count": accelerator["device_count"],
        "tensor_parallel_size": facts["tensor_parallel_size"],
        "accelerator": accelerator,
        "capabilities": capabilities,
    }


async def _control(app, pause: bool) -> dict:
    # The outer Runtime gate must first drain every admitted response stream.
    # Upstream mode='wait' alone freezes queued work rather than draining it.
    role, local, provenance = await asyncio.to_thread(_selection)
    client = app.state.engine_client
    if app.state.vllm_config is not client.vllm_config:
        raise ProvenanceError("API and engine configuration differ")
    _verify_applied(app.state, role, local, engine_config_facts(client.vllm_config))
    method = client.pause_generation if pause else client.resume_generation
    provenance.verify_api(method)
    provenance.verify_api(client.is_paused)
    lock = getattr(app.state, "_sovereign_control_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        app.state._sovereign_control_lock = lock
    async with lock:
        if pause:
            await method(mode="wait", clear_cache=False)
        else:
            await method()
        if await client.is_paused() is not pause:
            raise ProvenanceError("engine did not acknowledge the requested scheduler state")
    return {"quiesced": True} if pause else {"resumed": True}


async def observe_request(request, call_next):
    token = os.environ.get(TOKEN_ENV, "")
    supplied = request.headers.getlist("x-sovereign-observation-token")
    if (
        not _TOKEN.fullmatch(token)
        or len(supplied) != 1
        or not _TOKEN.fullmatch(supplied[0])
        or not hmac.compare_digest(token, supplied[0])
    ):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    path = request.url.path
    if path not in (OBSERVATION_PATH, "/sovereign/quiesce", "/sovereign/resume"):
        return await call_next(request)
    method = "GET" if path == OBSERVATION_PATH else "POST"
    if request.method != method:
        return JSONResponse(
            {"error": "method_not_allowed"}, status_code=405, headers={"Allow": method},
        )
    if method == "POST" and (
        request.headers.get("content-length", "0") != "0"
        or "transfer-encoding" in request.headers
        or request.url.query
    ):
        return JSONResponse({"error": "body_not_allowed"}, status_code=400)
    try:
        if path == OBSERVATION_PATH:
            observation = await collect_observation(request.app)
        else:
            observation = await _control(request.app, pause=path == "/sovereign/quiesce")
    except Exception:
        return JSONResponse(
            {"error": "observation_unavailable"}, status_code=503,
            headers={"Cache-Control": "no-store"},
        )
    return JSONResponse(observation, headers={"Cache-Control": "no-store"})
