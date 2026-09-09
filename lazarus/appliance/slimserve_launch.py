"""Offline, provenance-bound adapter for the reviewed SlimServe distribution.

Only the isolated interpreter imports SlimServe. Control supplies sealed local
inputs, never engine arguments, environment, downloads, or assembly work.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import platform
import re
import stat
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from lazarus.appliance.backends.vllm_engine import (
    LOCAL_SUPPORT_FILES as _VLLM_SUPPORT_FILES,
    LOCAL_SUPPORT_FILE_LIMIT,
    LOCAL_SUPPORT_TOTAL_LIMIT,
    _canonical_torch_uuid,
)
from lazarus.appliance.config import RuntimeConfig, SlimServeArtifact
from lazarus.appliance.slimserve_provenance import (
    SOURCE_COMMIT,
    bounded_json,
    installed_provenance,
    reject_optional_flashinfer,
)

if TYPE_CHECKING:
    from slimserve.registry import Plan

PORT = 18001
PYTHON = "/opt/sovereign-slimserve/bin/python"
KV_ROOT = Path("/var/lib/sovereign-slimserve/kv")
_INPUT_LIMIT = 1024 * 1024
_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\Z")
_MEMBER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_REQUIRED_CONFIG = {"config.json", "tokenizer.json", "tokenizer_config.json"}
# SlimServe ModelOpt consumes this additional quantization metadata file.
LOCAL_SUPPORT_FILES = _VLLM_SUPPORT_FILES | {"hf_quant_config.json"}
# DefaultModelLoader selects weights by extension; get_quant_config selects
# JSON. These exact repository documents have no runtime loader role. Keep
# them in upstream Plan/provenance, never admit them to the executable tree.
_REFERENCE_ONLY_FILES = frozenset({".gitattributes", "README.md", "LICENSE"})
# Audited keys at SOURCE_COMMIT, not an extensible caller flag surface.
_ENGINE_KEYS = frozenset({
    "attention_backend", "attention_config", "block_size", "compilation_config",
    "data_parallel_size", "default_chat_template_kwargs", "disable_custom_all_reduce",
    "dtype", "enable_auto_tool_choice", "enable_chunked_prefill", "enable_expert_parallel",
    "enable_prefix_caching", "gpu_memory_utilization", "kernel_config", "kv_cache_dtype",
    "kv_cache_memory_bytes", "kv_transfer_config", "language_model_only", "limit_mm_per_prompt",
    "linear_backend", "mamba_cache_dtype", "mamba_ssm_cache_dtype", "max_model_len",
    "max_num_batched_tokens", "max_num_seqs", "moe_backend", "override_generation_config",
    "reasoning_parser", "served_model_name", "tensor_parallel_size", "tokenizer_mode",
    "tool_call_parser", "trust_remote_code",
})
_NESTED_KEYS = {
    "attention_config": {"backend", "sparse_mla_force_mqa"},
    "compilation_config": {"cudagraph_mode", "max_cudagraph_capture_size"},
    "kernel_config": {"moe_backend", "linear_backend"},
    "default_chat_template_kwargs": {"thinking", "enable_thinking", "reasoning_effort", "preserve_thinking"},
    "override_generation_config": {"thinking_token_budget"},
    "limit_mm_per_prompt": {"vision_chunk", "image"},
    "kv_transfer_config": {"kv_connector", "kv_role", "kv_connector_extra_config"},
    "kv_connector_extra_config": {
        "host_tier_gb_per_rank", "nvme_tier_gb_per_rank", "enable_cross_layers_blocks",
        "main_kv_host_resident", "main_kv_gpu_rows", "main_kv_sub_blocks",
        "main_kv_tier_gb_per_rank", "kv_pool_deep_requests",
    },
}
_SPEC_KEYS = frozenset({
    "method", "num_speculative_tokens", "attention_backend", "kv_cache_dtype", "quantization",
    "disable_draft_cudagraphs", "num_speculative_tokens_per_batch_size",
    "index_share_for_mtp_iteration", "use_local_argmax_reduction",
})
_ENV_VALUES = {
    "NCCL_P2P_LEVEL": {"SYS"},
    "PYTORCH_CUDA_ALLOC_CONF": {"expandable_segments:True"},
    "VLLM_ADMISSION_MAX_CONCURRENT": {"96"},
    "VLLM_DSV4_ALIGNED_Q8": {"1"},
    "VLLM_DSV4_AUX_STREAMS": {"0"},
    "VLLM_DSV4_MHC_SCHEDULE": {"async"},
    "VLLM_DSV4_W1_QWARP8": {"1"},
    "VLLM_GDN_DECODE_KERNEL": {"triton"},
    "VLLM_GGUF_DSV4_REPACK_IQ2": {"0"},
    "VLLM_GGUF_DSV4_REPACK_Q2K": {"0"},
    "VLLM_METAL_ASYNC_SCHED": {"1"},
    "VLLM_QC_MUSE": {"1"},
    "VLLM_QWEN4_EXP_PLE_HOST": {"1"},
    "VLLM_QWEN4_EXP_SKINNY_GEMM": {"1"},
    "VLLM_QWEN4_EXP_SKINNY_W8": {"1"},
    "VLLM_SD_ADAPT_THROTTLE": {"1"},
    "VLLM_SSM_CONV_STATE_LAYOUT": {"DS"},
    "VLLM_USE_V2_MODEL_RUNNER": {"1"},
}
_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "SLIMSERVE_KV_TIER_DIR": str(KV_ROOT),
    "VLLM_PLUGINS": "",
}


class LocalPlanError(ValueError):
    """Sanitized rejection safe to return through the supervising process."""


@dataclass(frozen=True)
class LocalPlan:
    plan: Plan
    argv: list[str]
    engine: dict
    model_path: str
    tokenizer_path: str
    draft_path: str | None
    speculative_config: dict | None


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise LocalPlanError(reason)


def _modules():
    reject_optional_flashinfer()
    provenance = installed_provenance()
    registry = importlib.import_module("slimserve.registry")
    engine = importlib.import_module("slimserve.engine")
    hardware = importlib.import_module("slimserve.hardware")
    for module in (registry, engine, hardware):
        provenance.verify_module(module)
    return registry, engine, hardware


def _qc():
    return importlib.import_module("vllm._quixicore_C")


def _hardware(config, hardware):
    role = config.roles.generation
    variant = role.slimserve.variant
    count = role.tensor_parallel_size
    _require(count in (1, 2, 4), "unsupported device count")
    machine = hardware.detect()
    if variant == "metal":
        _require(platform.system() == "Darwin" and platform.machine() == "arm64",
                 "native Metal platform required")
        _require(count == 1 and not role.accelerator_device_ids and machine.platform == "metal"
                 and machine.count == 1 and machine.memory_bytes > 0,
                 "Metal device or memory unavailable")
        import torch
        _require(torch.backends.mps.is_available(), "Metal device unavailable")
        return machine.memory_bytes, machine.memory_bytes
    _require(platform.system() == "Linux" and platform.machine() == "x86_64",
             "managed CUDA platform required")
    ids = role.accelerator_device_ids
    _require(len(ids) == count and len(set(ids)) == count and all(_UUID.fullmatch(v) for v in ids),
             "invalid managed device identities")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == ",".join(ids), "device visibility mismatch")
    import torch
    _require(not getattr(torch.version, "hip", None) and torch.cuda.is_available()
             and torch.cuda.device_count() == count, "CUDA device count mismatch")
    names = []
    expected_capability = (8, 0) if variant == "a100" else (8, 6)
    for rank, identity in enumerate(ids):
        properties = torch.cuda.get_device_properties(rank)
        name = str(properties.name)
        _require(hardware._classify(name) == variant
                 and tuple(torch.cuda.get_device_capability(rank)) == expected_capability
                 and _canonical_torch_uuid(getattr(properties, "uuid", None)) == identity,
                 "CUDA device identity or architecture mismatch")
        names.append(name)
    _require(len(set(names)) == 1, "CUDA devices must be homogeneous")
    return 0, machine.host_ram_bytes


def _keys(value: dict, allowed, reason: str) -> None:
    _require(isinstance(value, dict) and not value.keys() - allowed, reason)


def _settings(plan) -> None:
    _keys(plan.engine, _ENGINE_KEYS, "unreviewed engine setting")
    for key, value in plan.engine.items():
        if isinstance(value, dict):
            _require(key in _NESTED_KEYS, "unreviewed nested engine setting")
            _keys(value, _NESTED_KEYS[key], "unreviewed nested engine setting")
            for inner, item in value.items():
                if isinstance(item, dict):
                    _require(inner in _NESTED_KEYS, "unreviewed nested engine setting")
                    _keys(item, _NESTED_KEYS[inner], "unreviewed nested engine setting")
                    _require(all(type(v) in (str, int, float, bool) for v in item.values()),
                             "unreviewed nested engine value")
                else:
                    _require(type(item) in (str, int, float, bool), "unreviewed engine value")
        else:
            _require(type(value) in (str, int, float, bool), "unreviewed engine value")
    _require(type(plan.engine.get("data_parallel_size", 1)) is int
             and plan.engine.get("data_parallel_size", 1) == 1,
             "data parallel profiles are unsupported")
    _require(type(plan.engine.get("enable_expert_parallel", False)) is bool,
             "invalid expert parallel setting")
    _require(isinstance(plan.env, dict) and all(
        key in _ENV_VALUES and value in _ENV_VALUES[key] for key, value in plan.env.items()
    ), "unreviewed profile environment")
    _keys(plan.speculative_overrides, _SPEC_KEYS, "unreviewed speculative setting")
    if plan.speculative:
        _require(isinstance(plan.speculator, dict), "missing profile drafter")
        _keys(plan.speculator, {"repo", "revision", "base_url", "local_dir", "file", "bytes", "engine", "note"},
              "unreviewed speculative source setting")
        if plan.speculator.get("file"):
            _keys(plan.speculator["file"], {"path", "bytes", "sha256"},
                  "unreviewed speculative file setting")
        _keys(plan.speculator.get("engine"), _SPEC_KEYS, "unreviewed speculative setting")
    elif plan.speculative_overrides:
        raise LocalPlanError("unused speculative settings")


def _open_directory(path: Path) -> int:
    """Open every ancestor without following a symlink, including the root."""
    _require(path.is_absolute() and not any(p in (".", "..") for p in path.parts),
             "invalid prepared root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _role_directories(root: Path, roles) -> None:
    descriptor = _open_directory(root.parent)
    seen = set()
    try:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                _require(entry.name in roles and stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode),
                         "unexpected prepared role root")
                seen.add(entry.name)
    finally:
        os.close(descriptor)
    _require(seen == set(roles), "missing prepared role root")


def _member(name: str) -> bool:
    parts = name.split("/")
    return 0 < len(name) <= 192 and len(parts) <= 4 and all(_MEMBER.fullmatch(p) for p in parts)


def _read_member(directory: int, name: str, expected, verify_files: bool) -> bytes | None:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size == expected.size_bytes,
                 "artifact member size or type mismatch")
        metadata = expected.file in LOCAL_SUPPORT_FILES
        contents = bytearray() if metadata else None
        digest = hashlib.sha256() if verify_files else None
        if verify_files or metadata:
            remaining = expected.size_bytes
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                _require(bool(chunk), "artifact member changed during verification")
                remaining -= len(chunk)
                if digest is not None:
                    digest.update(chunk)
                if contents is not None:
                    contents.extend(chunk)
            _require(not source.read(1), "artifact member changed during verification")
        after = os.fstat(source.fileno())
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 "artifact member changed during verification")
        if digest is not None:
            _require(digest.hexdigest() == expected.sha256, "artifact member digest mismatch")
        return bytes(contents) if contents is not None else None


def _closure(root: Path, artifact: SlimServeArtifact, verify_files: bool) -> dict[str, dict]:
    expected = {entry.file: entry for entry in artifact.files}
    _require(len(expected) == len(artifact.files) and 1 <= len(expected) <= 256
             and all(_member(name) for name in expected), "invalid artifact closure")
    _require(all(name in LOCAL_SUPPORT_FILES or name.endswith((".safetensors", ".gguf"))
                 or re.fullmatch(r"[A-Za-z0-9._/-]+\.gguf\.part-[0-9]{2}-of-[0-9]{2}", name)
                 for name in expected), "unsupported artifact member")
    metadata = {name: entry for name, entry in expected.items() if name in LOCAL_SUPPORT_FILES}
    _require(all(entry.size_bytes <= LOCAL_SUPPORT_FILE_LIMIT for entry in metadata.values()),
             "artifact metadata exceeds limit")
    directories = {str(parent) for name in expected for parent in Path(name).parents if str(parent) != "."}
    seen = set()
    documents = {}

    def visit(descriptor: int, prefix: str = ""):
        with os.scandir(descriptor) as entries:
            for entry in entries:
                name = prefix + entry.name
                info = entry.stat(follow_symlinks=False)
                if name in directories:
                    _require(stat.S_ISDIR(info.st_mode), "artifact directory type mismatch")
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                                    dir_fd=descriptor)
                    try:
                        visit(child, name + "/")
                    finally:
                        os.close(child)
                else:
                    _require(name in expected and stat.S_ISREG(info.st_mode),
                             "unexpected artifact member")
                    contents = _read_member(descriptor, entry.name, expected[name], verify_files)
                    seen.add(name)
                    if contents is not None and name.endswith(".json"):
                        documents[name] = bounded_json(contents.decode("utf-8"), LOCAL_SUPPORT_FILE_LIMIT)
    descriptor = _open_directory(root)
    try:
        visit(descriptor)
    finally:
        os.close(descriptor)
    _require(seen == expected.keys(), "missing artifact member")
    if "model.safetensors.index.json" in documents:
        weights = documents["model.safetensors.index.json"].get("weight_map")
        supplied = {name for name in expected if name.endswith(".safetensors")}
        _require(isinstance(weights, dict) and bool(weights)
                 and all(isinstance(value, str) and "/" not in value and value in supplied
                         for value in weights.values())
                 and set(weights.values()) == supplied, "invalid safetensors weight index")
    return documents


def _match(artifact, entries) -> set[str]:
    supplied = {entry.file: entry for entry in artifact.files}
    names = set()
    for entry in entries:
        name = entry.get("path")
        _require(isinstance(name, str) and _member(name) and name not in names,
                 "invalid upstream artifact member")
        names.add(name)
        actual = supplied.get(name)
        _require(actual is not None and actual.size_bytes == entry.get("bytes"),
                 "registered artifact member mismatch")
        if digest := entry.get("sha256"):
            _require(actual.sha256 == digest, "registered artifact digest mismatch")
    return names


def _checkpoint(artifact, documents, tokenizer: bool):
    names = {entry.file for entry in artifact.files}
    required = _REQUIRED_CONFIG if tokenizer else {"config.json"}
    weights = {name for name in names if name.endswith(".safetensors") and "/" not in name}
    _require(required <= names and bool(weights), "incomplete local checkpoint")
    _require(names <= LOCAL_SUPPORT_FILES | weights, "unsupported checkpoint member")
    _require(len(weights) == 1 or "model.safetensors.index.json" in documents,
             "sharded checkpoint requires a complete weight index")


def _qwen_qsa_inputs(plan, documents: dict) -> None:
    """Exclude the pinned Qwen4Exp indexer's FlashInfer reference branch.

    These source checkpoints use the same text config for target and MTP.
    Only explicit, locally sealed constructor inputs are interpreted; missing
    Transformer defaults and alternate RoPE implementations are not invented.
    """
    if plan.source_key not in {"qwen38-flash-next-fp8", "qwen38-flash-next-nvfp4"}:
        return
    document = documents.get("config.json", {})
    _require(document.get("model_type") in ("qwen4_exp", "qwen4_exp_text"),
             "Qwen local architecture does not match the reviewed source")
    text = document.get("text_config", document)
    _require(isinstance(text, dict) and text.get("model_type") in ("qwen4_exp", "qwen4_exp_text"),
             "Qwen local text configuration is unavailable")
    fields = ("indexer_n_heads", "indexer_kv_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio")
    _require(all(type(text.get(key)) is int and text[key] > 0 for key in fields),
             "Qwen indexer configuration is incomplete")
    ratio = text["indexer_compress_ratio"]
    budget = text["indexer_budget"]
    _require(text["indexer_head_dim"] == 128 and text["indexer_kv_heads"] == 1
             and ratio > 1 and ratio & (ratio - 1) == 0
             and budget % ratio == 0 and budget // ratio in (512, 2048),
             "Qwen indexer requires an unavailable kernel path")
    head = text.get("head_dim")
    rope = text.get("rope_parameters")
    _require(type(head) is int and head > 0 and isinstance(rope, dict)
             and rope.get("rope_type") == "default", "Qwen rotary configuration is unreviewed")
    rotary = rope.get("rope_dim")
    if not rotary:
        factor = rope.get("partial_rotary_factor")
        _require(type(factor) in (int, float) and 0 < factor <= 1,
                 "Qwen rotary factor is unavailable")
        rotary = int(head * factor)
    _require(type(rotary) is int and rotary == 64, "Qwen rotary dimension requires an unavailable kernel")
    sections = rope.get("mrope_section")
    _require(sections is None or isinstance(sections, list) and (
        not sections or len(sections) == 3
        and all(type(value) is int and value >= 0 for value in sections)
        and sum(sections) == 32 and rope.get("mrope_interleaved") is True
    ), "Qwen rotary layout requires an unavailable kernel")
    dtype = plan.engine.get("dtype", "auto")
    if dtype == "auto":
        dtype = document.get("dtype", document.get("torch_dtype", text.get("dtype", text.get("torch_dtype"))))
    _require(dtype == "bfloat16", "Qwen indexer requires explicit BF16 model metadata")


def _capacity(plan, host_ram: int, verify_capacity: bool) -> None:
    minimum = plan.quant.min_host_ram_bytes.get(plan.platform, 0)
    transfer = plan.engine.get("kv_transfer_config")
    if transfer:
        _require(transfer.get("kv_connector") == "HostTierConnector"
                 and transfer.get("kv_role") == "kv_both", "unreviewed KV connector")
        extra = transfer.get("kv_connector_extra_config", {})
        host_gb = extra.get("host_tier_gb_per_rank", 0)
        main_gb = extra.get("main_kv_tier_gb_per_rank", 0)
        _require(type(host_gb) in (int, float) and host_gb > 0
                 and type(main_gb) in (int, float) and main_gb >= 0, "invalid host tier capacity")
        minimum = max(minimum, int((host_gb + main_gb) * (1 << 30)) * plan.gpus)
        nvme_gb = extra.get("nvme_tier_gb_per_rank", 0)
        _require(type(nvme_gb) in (int, float) and nvme_gb >= 0, "invalid NVMe tier capacity")
        if nvme_gb and verify_capacity:
            descriptor = _open_directory(KV_ROOT)
            try:
                usage = os.fstatvfs(descriptor)
                free = usage.f_bavail * usage.f_frsize
                _require(free >= int(nvme_gb * (1 << 30)) * plan.gpus,
                         "insufficient NVMe tier capacity")
            finally:
                os.close(descriptor)
    _require(not minimum or host_ram > 0 and host_ram >= minimum,
             "host memory is unknown or insufficient")


def _context(documents: dict) -> int:
    config = documents.get("config.json", {})
    config = config.get("text_config", config)
    _require(isinstance(config, dict), "invalid local model configuration")
    values = [config.get(key) for key in (
        "max_position_embeddings", "n_positions", "max_seq_len", "seq_length", "model_max_length",
    )]
    ceilings = [value for value in values if type(value) is int and value > 0]
    _require(bool(ceilings), "local context ceiling is unknown")
    return min(ceilings)


def resolve_local(config: RuntimeConfig, verify_files: bool = True) -> LocalPlan:
    """Resolve pinned APIs against a sealed, completely local input closure.

    Observation skips weight hashes and NVMe free-space reservation checks
    after allocation. Shape, metadata, profile, placement and physical-memory
    checks remain identical; replay grants no new launch authority.
    """
    try:
        return _resolve_local(config, verify_files)
    except LocalPlanError:
        raise
    except Exception:
        raise LocalPlanError("SlimServe local configuration is unavailable or invalid") from None


def _resolve_local(config: RuntimeConfig, verify_files: bool) -> LocalPlan:
    # Revalidate even an in-process model_copy/model_construct caller.
    config = RuntimeConfig.model_validate(config.model_dump(exclude_unset=True))
    role = config.roles.generation
    _require(role.engine == "slimserve" and role.slimserve is not None, "SlimServe generation required")
    selected = role.slimserve
    registry, engine, hardware = _modules()
    _require(not registry.profile_blocked(selected.profile_id, selected.variant), "upstream profile is blocked")
    memory, host_ram = _hardware(config, hardware)
    plan = registry.resolve(selected.profile_id, selected.variant, role.tensor_parallel_size,
                            selected.quant, memory_bytes=memory, host_ram_bytes=host_ram)
    _require(plan.gpus == role.tensor_parallel_size
             and plan.engine.get("tensor_parallel_size") == role.tensor_parallel_size,
             "upstream profile requires a different device count")
    _settings(plan)
    _capacity(plan, host_ram, verify_files)
    artifacts = {artifact.role: artifact for artifact in selected.artifacts}
    model = artifacts["model"]
    _require(model.repository == plan.source["repo"] and model.revision == role.revision
             and (not plan.source.get("revision") or model.revision == plan.source["revision"]),
             "model repository or revision mismatch")
    _require(sum(entry.size_bytes for artifact in artifacts.values() for entry in artifact.files
                 if entry.file in LOCAL_SUPPORT_FILES) <= LOCAL_SUPPORT_TOTAL_LIMIT,
             "artifact metadata exceeds aggregate limit")
    root = Path(role.model)
    _require(root.name == "model", "prepared model root required")
    _role_directories(root, artifacts)
    roots = {name: root if name == "model" else root.parent / name for name in artifacts}
    documents = {name: _closure(roots[name], artifact, verify_files) for name, artifact in artifacts.items()}
    _qwen_qsa_inputs(plan, documents["model"])
    shared = plan.source.get("shared") or []
    _require(all(entry.get("repo", model.repository) == model.repository
                 and entry.get("base_url", plan.source.get("base_url")) == plan.source.get("base_url")
                 for entry in shared), "cross-source shared inputs are unsupported")
    registered = _match(model, shared)
    if plan.quant.assembly:
        assembly = plan.quant.assembly
        digest = assembly.get("sha256_patched") if assembly.get("patch") else assembly.get("sha256_published")
        _require(bool(digest), "unsealed assembly output")
        registered |= _match(model, [{"path": assembly["output"], "bytes": assembly["bytes"], "sha256": digest}])
        # Published assemblies may intentionally discard their source parts.
        # Check any retained parts, but never demand or assemble missing ones.
        supplied = {entry.file for entry in model.files}
        registered |= _match(model, [entry for entry in plan.quant.files if entry["path"] in supplied])
    else:
        registered |= _match(model, [entry for entry in plan.quant.files
                                     if entry["path"] not in _REFERENCE_ONLY_FILES])
    names = {entry.file for entry in model.files}
    _require(names <= registered | LOCAL_SUPPORT_FILES, "unregistered model member")
    checkpoint = plan.source.get("format") == "safetensors"
    used = {"model"}
    if checkpoint:
        _checkpoint(model, documents["model"], tokenizer=True)
        tokenizer_root = root
    else:
        _require("tokenizer" in artifacts, "GGUF requires an explicit local tokenizer closure")
        tokenizer = artifacts["tokenizer"]
        tokenizer_names = {entry.file for entry in tokenizer.files}
        _require(_REQUIRED_CONFIG <= tokenizer_names and tokenizer_names <= LOCAL_SUPPORT_FILES,
                 "incomplete local tokenizer closure")
        tokenizer_root = roots["tokenizer"]
        used.add("tokenizer")
    speculator = None
    draft_path = None
    if plan.speculative:
        original = plan.speculator
        same = (original.get("repo") == plan.source["repo"]
                and original.get("local_dir") == plan.source["local_dir"] and not original.get("file")
                and original["engine"].get("method") in ("mtp", "qwen3_5_mtp"))
        if same:
            _require(checkpoint, "MTP requires a local checkpoint")
            draft_root = root
        else:
            _require("drafter" in artifacts and bool(re.fullmatch(r"[0-9a-f]{40}", original.get("revision", ""))),
                     "separate drafter requires an immutable upstream pin")
            draft = artifacts["drafter"]
            _require(draft.repository == original["repo"] and draft.revision == original["revision"],
                     "drafter repository or revision mismatch")
            draft_root = roots["drafter"]
            used.add("drafter")
            if original.get("file"):
                required = _match(draft, [original["file"]])
                _require({entry.file for entry in draft.files} == required, "unexpected drafter member")
            else:
                _checkpoint(draft, documents["drafter"], tokenizer=False)
        speculator = {**copy.deepcopy(original), "local_dir": str(draft_root)}
        draft_path = str(draft_root / original["file"]["path"]) if original.get("file") else str(draft_root)
    _require(used == artifacts.keys(), "unused artifact role")
    applied = copy.deepcopy(plan.engine)
    ceiling = applied.get("max_model_len")
    if ceiling is None:
        ceiling = _context(documents["model" if checkpoint else "tokenizer"])
    _require(type(ceiling) is int and 0 < role.max_model_len <= ceiling, "context exceeds reviewed ceiling")
    # Pinned EngineArgs.get_batch_defaults: A100/OpenAI server = 256.
    _require(role.max_concurrent_requests <= applied.get("max_num_seqs", 256),
             "concurrency exceeds reviewed ceiling")
    for parser in ("tool_call_parser", "reasoning_parser"):
        requested = getattr(role, parser)
        _require(requested is None or requested == applied.get(parser), "parser differs from reviewed profile")
    # Pinned TRTLLM discovery otherwise probes NVIDIA before checking SM.
    # None of SM80, SM86, or native Metal supports that kernel path.
    applied.setdefault("attention_config", {})["use_trtllm_attention"] = False
    if role.enforce_eager:
        # This pin has no --enforce-eager flag. CompilationMode.NONE is
        # fully eager; graph NONE independently disables graph replay.
        applied.setdefault("compilation_config", {}).update(
            mode=0, cudagraph_mode="NONE", max_cudagraph_capture_size=0,
        )
    applied.update(served_model_name=role.served_model_name, max_model_len=role.max_model_len,
                   max_num_seqs=role.max_concurrent_requests, revision=role.revision,
                   trust_remote_code=False, tokenizer=str(tokenizer_root),
                   hf_config_path=str(tokenizer_root))
    local = replace(plan, source={**copy.deepcopy(plan.source), "local_dir": str(root)},
                    variant_speculator=speculator, engine=applied)
    argv = engine.serve_argv(local, "127.0.0.1", PORT)
    speculative = None
    if "--speculative-config" in argv:
        speculative = json.loads(argv[argv.index("--speculative-config") + 1])
        _require(speculative.get("model") == draft_path and "revision" not in speculative,
                 "nonlocal speculative fallback rejected")
    _require(bool(speculative) == bool(plan.speculative), "missing speculative configuration")
    worker = "MetalWorker" if selected.variant == "metal" else "CudaWorker"
    argv.extend(["--worker-cls", f"lazarus.appliance.slimserve_worker.{worker}",
                 "--middleware", "lazarus.appliance.slimserve_observation.observe_request"])
    return LocalPlan(local, argv, applied, str(local.entry_file), str(tokenizer_root), draft_path, speculative)


def _availability() -> dict:
    reject_optional_flashinfer()
    provenance = installed_provenance()
    backend = "metal" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cuda"
    kernels = provenance.verify_extension(_qc(), backend)
    return {**provenance.engine, "variants": ["metal"] if backend == "metal" else ["a100", "rtx3090"],
            "kernel_library": {"name": kernels["library"], "version": kernels["version"]}}


def main() -> int:
    try:
        _require(sys.executable == PYTHON, "isolated SlimServe interpreter required")
        _require(sys.argv[1:] in ([], ["--check"], ["--availability"]), "unsupported helper arguments")
        os.environ.update(_OFFLINE_ENV)
        if sys.argv[1:] == ["--availability"]:
            with redirect_stdout(sys.stderr):
                availability = _availability()
            print(json.dumps(availability))
            return 0
        raw = sys.stdin.buffer.read(_INPUT_LIMIT + 1)
        document = bounded_json(raw.decode("utf-8"), _INPUT_LIMIT)
        config = RuntimeConfig.model_validate(document)
        local = resolve_local(config)
        if sys.argv[1:] == ["--check"]:
            print(json.dumps({"validated": True, "engine_model": local.model_path}))
            return 0
        environment = dict(os.environ)
        environment.update(local.plan.env)
        environment.update(_OFFLINE_ENV)
        environment["SOVEREIGN_SLIMSERVE_IDENTITY"] = json.dumps({
            "engine_profile_id": config.roles.generation.engine_profile_id,
            "profile_id": local.plan.profile_id, "variant": local.plan.platform,
            "quant": local.plan.quant.name, "source_commit": SOURCE_COMMIT,
            "config": config.model_dump(exclude_none=True, exclude_unset=True),
        }, separators=(",", ":"))
        os.execve(sys.executable, [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *local.argv], environment)
        return 0
    except Exception:
        # Never emit upstream errors, private paths, configuration, or tracebacks.
        print("SlimServe local launch validation failed", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
