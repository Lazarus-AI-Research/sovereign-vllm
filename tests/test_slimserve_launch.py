"""Offline contracts using the actual pinned registry/engine and tiny artifacts.

Embedded Apache-2.0 upstream sources are unmodified. Model sizes/digests in the
in-memory registry are reduced only for closure tests, never serving evidence.
"""

import base64
import hashlib
import importlib
import io
import json
import socket
import sys
import zlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lazarus.appliance import slimserve_launch as launch
from lazarus.appliance.config import RuntimeConfig
from lazarus.appliance.slimserve_provenance import InstalledProvenance, ProvenanceError, SOURCE_COMMIT

# Relevant constructor inputs from the audited Qwen4Exp checkpoint shape.
_QWEN_QSA_CONFIG = {
    "model_type": "qwen4_exp", "dtype": "bfloat16",
    "text_config": {
        "model_type": "qwen4_exp_text", "head_dim": 256,
        "max_position_embeddings": 262144, "indexer_n_heads": 4,
        "indexer_head_dim": 128, "indexer_kv_heads": 1,
        "indexer_budget": 2048, "indexer_compress_ratio": 4,
        "rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.25,
                            "mrope_section": [11, 11, 10], "mrope_interleaved": True},
    },
}


@pytest.fixture()
def pinned(tmp_path, monkeypatch):
    installation = tmp_path.resolve() / "installed"
    package = installation / "slimserve"
    package.mkdir(parents=True)
    entries = []
    for name, (digest, encoded) in _PINNED.items():
        contents = zlib.decompress(base64.b64decode(encoded))
        assert hashlib.sha256(contents).hexdigest() == digest
        (package / name).write_bytes(contents)
        entries.append({"path": "slimserve/" + name, "sha256": digest})
    provenance_path = installation / "provenance.json"
    provenance_path.write_text(json.dumps({
        "schema_version": 1,
        "slimserve": {"source_repository": "https://github.com/QuixiAI/SlimServe",
                      "source_commit": SOURCE_COMMIT, "package_version": "test-fixture", "files": entries},
        "kernels": {"artifacts": entries},
    }))
    provenance = InstalledProvenance(provenance_path, installation)
    monkeypatch.setattr(launch, "installed_provenance", lambda: provenance)
    for name in tuple(sys.modules):
        if name == "slimserve" or name.startswith("slimserve."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(installation))
    registry = importlib.import_module("slimserve.registry")
    engine = importlib.import_module("slimserve.engine")
    hardware = importlib.import_module("slimserve.hardware")
    for name in ("slimserve", "slimserve.registry", "slimserve.engine", "slimserve.hardware", "slimserve.term"):
        loaded = sys.modules.pop(name)
        monkeypatch.setitem(sys.modules, name, loaded)
    def forbid_network(*args, **kwargs):
        raise AssertionError("offline launch attempted networking")
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    return SimpleNamespace(registry=registry, engine=engine, hardware=hardware,
                           provenance=provenance, package=package)


@pytest.fixture()
def prepared(pinned, tmp_path, monkeypatch):
    def make(profile_id="dsv4-q4ktail-2", variant="a100"):
        record = pinned.registry._registry()["profiles"][profile_id]
        source = pinned.registry._registry()["sources"][record["source"]]
        quant = record["variants"][variant]["default_quant"]
        raw_quant = source["quants"][quant]
        count = record["gpus"]
        root = tmp_path.resolve() / "staged" / profile_id / "model"
        root.mkdir(parents=True)
        weights = [e["path"] for e in raw_quant["files"] if e["path"].endswith(".safetensors")]
        all_contents = {}
        for entry in raw_quant["files"] + source.get("shared", []):
            name = entry["path"]
            if name in {".gitattributes", "README.md", "LICENSE"}:
                continue
            if name == "config.json":
                contents = b'{"model_type":"qwen3_5","max_position_embeddings":262144}'
                if record["source"] in {"qwen38-flash-next-fp8", "qwen38-flash-next-nvfp4"}:
                    contents = json.dumps(_QWEN_QSA_CONFIG).encode()
            elif name == "model.safetensors.index.json":
                contents = json.dumps({"weight_map": {str(i): name for i, name in enumerate(weights)}}).encode()
            elif name.endswith(".json"):
                contents = b"{}"
            else:
                contents = ("offline fixture " + name).encode()
            entry.update(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
            all_contents[name] = contents
        role_contents = {"model": all_contents}
        revision = "1" * 40
        repositories = {"model": (source["repo"], revision)}
        if source.get("format") != "safetensors":
            role_contents["tokenizer"] = {"config.json": b'{"max_position_embeddings":262144}',
                                          "tokenizer.json": b"{}", "tokenizer_config.json": b"{}"}
            repositories["tokenizer"] = ("reviewed/tokenizer", "2" * 40)
        spec = record["variants"][variant].get("speculator") or source.get("speculator")
        if record.get("speculative") and spec:
            same = spec["repo"] == source["repo"] and spec["local_dir"] == source["local_dir"] and not spec.get("file")
            if not same:
                if spec.get("file"):
                    contents = b"offline drafter fixture"
                    spec["file"].update(bytes=len(contents), sha256=hashlib.sha256(contents).hexdigest())
                    role_contents["drafter"] = {spec["file"]["path"]: contents}
                else:
                    role_contents["drafter"] = {"config.json": b"{}", "model.safetensors": b"draft weights"}
                repositories["drafter"] = (spec["repo"], spec.get("revision", "3" * 40))
        artifacts = []
        for role, contents in role_contents.items():
            directory = root if role == "model" else root.parent / role
            directory.mkdir(exist_ok=True)
            members = []
            for name, value in contents.items():
                path = directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(value)
                members.append({"file": name, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()})
            repository, pin = repositories[role]
            artifacts.append({"role": role, "repository": repository, "revision": pin, "files": members})
        ids = [f"GPU-00000000-0000-0000-0000-{rank:012x}" for rank in range(count, 0, -1)]
        properties = [SimpleNamespace(name="NVIDIA GeForce RTX 3090" if variant == "rtx3090" else "NVIDIA A100-SXM4-80GB", uuid=value) for value in ids]
        torch = ModuleType("torch")
        torch.version = SimpleNamespace(hip=None)
        torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
        torch.cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: len(properties),
                                     get_device_properties=lambda rank: properties[rank],
                                     get_device_capability=lambda rank: (8, 6) if variant == "rtx3090" else (8, 0))
        monkeypatch.setitem(sys.modules, "torch", torch)
        machine = SimpleNamespace(platform=variant, count=count, memory_bytes=1024**4, host_ram_bytes=1024**4)
        monkeypatch.setattr(pinned.hardware, "detect", lambda: machine)
        monkeypatch.setattr(launch.platform, "system", lambda: "Darwin" if variant == "metal" else "Linux")
        monkeypatch.setattr(launch.platform, "machine", lambda: "arm64" if variant == "metal" else "x86_64")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(ids))
        generation = {"enabled": True, "engine": "slimserve", "engine_profile_id": "sealed-profile",
                      "task": "generate", "source": "local", "model": str(root), "revision": revision,
                      "served_model_name": "assistant-large", "max_model_len": 2048,
                      "max_concurrent_requests": 4, "enforce_eager": True, "tensor_parallel_size": count,
                      "slimserve": {"source_commit": SOURCE_COMMIT, "profile_id": profile_id,
                                    "variant": variant, "quant": quant, "artifacts": artifacts}}
        if variant != "metal":
            generation["accelerator_device_ids"] = ids
        data = {"schema_version": "1.2", "runtime": {"profile": "metal-arm64" if variant == "metal" else "cuda-x86_64"},
                "roles": {"generation": generation}}
        return SimpleNamespace(config=lambda: RuntimeConfig.model_validate(data), data=data, root=root,
                               generation=generation, record=record, source=source, quant=raw_quant,
                               spec=spec, artifacts=artifacts, machine=machine, properties=properties,
                               torch=torch, pinned=pinned)
    return make


def artifact(sample, role="model"):
    return next(item for item in sample.artifacts if item["role"] == role)


def reseal(sample, name, content, role="model"):
    root = sample.root if role == "model" else sample.root.parent / role
    (root / name).write_bytes(content)
    member = next(item for item in artifact(sample, role)["files"] if item["file"] == name)
    member.update(size_bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    return member


@pytest.mark.parametrize("profile_id,variant", [("dsv4-q4ktail-2", "a100"), ("dsv4-q4ktail-4", "a100"), ("dsv4-xxs-1", "metal"), ("qwen38-nvfp4-1", "metal"), ("qwen38-nvfp4-1-tq", "metal")])
def test_real_pinned_resolution_has_only_prepared_inputs(prepared, monkeypatch, profile_id, variant):
    sample = prepared(profile_id, variant)
    monkeypatch.setenv("SLIMSERVE_CACHE", "/untrusted/cache")
    result = launch.resolve_local(sample.config())
    assert isinstance(result.plan, sample.pinned.registry.Plan)
    assert Path(result.plan.source["local_dir"]) == sample.root
    assert result.plan.variant_speculator["local_dir"] == str(sample.root.parent / "drafter")
    assert result.model_path.startswith(str(sample.root))
    assert result.speculative_config["model"] == result.draft_path
    assert "revision" not in result.speculative_config
    assert result.engine["max_num_seqs"] == 4 and result.engine["max_model_len"] == 2048
    assert result.engine["trust_remote_code"] is False
    assert result.argv[:6] == ["--model", result.model_path, "--host", "127.0.0.1", "--port", "18001"]
    assert result.argv[result.argv.index("--tokenizer") + 1] == result.tokenizer_path
    assert result.argv[result.argv.index("--hf-config-path") + 1] == result.tokenizer_path
    assert "--trust-remote-code" not in result.argv
    assert result.argv[-4:] == ["--worker-cls", "lazarus.appliance.slimserve_worker." + ("MetalWorker" if variant == "metal" else "CudaWorker"), "--middleware", "lazarus.appliance.slimserve_observation.observe_request"]
    assert "slimserve.cli" not in sys.modules and "slimserve.fetch" not in sys.modules


def test_unmodified_registry_blocking_is_not_resolve(pinned):
    assert pinned.registry.profile_blocked("glm52-xxs-1", "metal")
    assert pinned.registry.resolve("glm52-xxs-1", "metal", 1, None, memory_bytes=1024**4).platform == "metal"


@pytest.mark.parametrize("role", ["model", "drafter", "tokenizer"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "size", "digest", "symlink"])
def test_exact_closure_is_required(prepared, role, mutation):
    sample = prepared()
    entry = artifact(sample, role)["files"][0]
    directory = sample.root if role == "model" else sample.root.parent / role
    path = directory / entry["file"]
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (directory / "unexpected.py").write_text("raise RuntimeError")
    elif mutation == "size":
        path.write_bytes(path.read_bytes() + b"!")
    elif mutation == "digest":
        path.write_bytes(b"X" + path.read_bytes()[1:])
    else:
        target = directory.parent / "outside"
        path.rename(target)
        path.symlink_to(target)
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("level", ["root", "parent"])
def test_root_and_ancestor_symlinks_fail(prepared, level):
    sample = prepared()
    path = sample.root if level == "root" else sample.root.parent
    moved = path.with_name(path.name + "-moved")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=True)
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


def test_observation_skips_hashes_but_not_shape(prepared):
    sample = prepared()
    path = sample.root / artifact(sample)["files"][0]["file"]
    path.write_bytes(b"X" + path.read_bytes()[1:])
    assert launch.resolve_local(sample.config(), verify_files=False).model_path == str(path)
    with pytest.raises(launch.LocalPlanError, match="digest"):
        launch.resolve_local(sample.config())
    path.unlink()
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config(), verify_files=False)


@pytest.mark.parametrize("field,value", [("engine", {"worker_cls": "unreviewed.Worker"}), ("engine", {"compilation_config": {"unreviewed": True}}), ("env", {"PYTHONPATH": "/unreviewed"}), ("env", {"VLLM_DSV4_ALIGNED_Q8": "0"}), ("speculative_overrides", {"model": "unreviewed/repository"}), ("engine", {"enable_expert_parallel": "true"}), ("engine", {"data_parallel_size": 2})])
def test_unreviewed_settings_fail_closed(prepared, field, value):
    sample = prepared()
    sample.record["variants"]["a100"].setdefault(field, {}).update(value)
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


def test_profile_blocked_is_checked_before_loading(prepared):
    sample = prepared()
    sample.record["variants"]["a100"]["status"] = "in-progress"
    with pytest.raises(launch.LocalPlanError, match="blocked"):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("mutation", ["count", "order", "name", "capability", "visibility", "duplicate", "hip"])
def test_every_cuda_rank_is_checked(prepared, monkeypatch, mutation):
    sample = prepared()
    if mutation == "count":
        sample.properties.append(sample.properties[0])
    elif mutation == "order":
        sample.properties.reverse()
    elif mutation == "name":
        sample.properties[1].name = "NVIDIA H100"
    elif mutation == "capability":
        sample.torch.cuda.get_device_capability = lambda rank: (8, 0) if rank == 0 else (8, 6)
    elif mutation == "visibility":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    elif mutation == "duplicate":
        sample.properties[1].uuid = sample.properties[0].uuid
    else:
        sample.torch.version.hip = "unexpected"
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("count", [1, 4])
def test_exact_profile_count_not_upstream_minimum(prepared, monkeypatch, count):
    sample = prepared()
    sample.generation["tensor_parallel_size"] = count
    ids = [f"GPU-00000000-0000-0000-0000-{rank:012x}" for rank in range(count, 0, -1)]
    sample.generation["accelerator_device_ids"] = ids
    sample.properties[:] = [SimpleNamespace(name="NVIDIA A100-SXM4-80GB", uuid=value) for value in ids]
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(ids))
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("mutation", ["platform", "memory", "unavailable"])
def test_metal_requires_actual_platform_device_and_memory(prepared, monkeypatch, mutation):
    sample = prepared("qwen38-nvfp4-1", "metal")
    if mutation == "platform":
        monkeypatch.setattr(launch.platform, "system", lambda: "Linux")
    elif mutation == "memory":
        sample.machine.memory_bytes = 0
    else:
        sample.torch.backends.mps.is_available = lambda: False
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("amount", [0, 1])
def test_host_capacity_cannot_use_upstream_unknown_bypass(prepared, amount):
    sample = prepared()
    sample.machine.host_ram_bytes = amount
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("mutation", ["missing", "insufficient"])
def test_nvme_capacity_is_not_rereserved_by_observer(prepared, monkeypatch, tmp_path, mutation):
    sample = prepared()
    sample.record["variants"]["a100"]["engine"]["kv_transfer_config"]["kv_connector_extra_config"]["nvme_tier_gb_per_rank"] = 256
    tier = tmp_path.resolve() / "kv-tier"
    monkeypatch.setattr(launch, "KV_ROOT", tier)
    if mutation == "insufficient":
        tier.mkdir()
        monkeypatch.setattr(launch.os, "fstatvfs", lambda descriptor: SimpleNamespace(f_bavail=1, f_frsize=4096))
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())
    assert launch.resolve_local(sample.config(), verify_files=False).plan.profile_id == "dsv4-q4ktail-2"


@pytest.mark.parametrize("field,value", [("max_model_len", 262145), ("max_concurrent_requests", 257), ("tool_call_parser", "off"), ("reasoning_parser", "unreviewed")])
def test_runtime_overrides_are_bounded(prepared, field, value):
    sample = prepared()
    sample.generation[field] = value
    with pytest.raises(ValueError):
        launch.resolve_local(sample.config())


def test_profile_context_and_concurrency_ceilings_apply(prepared):
    sample = prepared("qwen38-nvfp4-1", "metal")
    sample.generation["max_concurrent_requests"] = 9
    with pytest.raises(launch.LocalPlanError, match="concurrency"):
        launch.resolve_local(sample.config())
    sample.generation["max_concurrent_requests"] = 4
    sample.generation["max_model_len"] = 32769
    with pytest.raises(launch.LocalPlanError, match="context"):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("role", ["model", "drafter"])
@pytest.mark.parametrize("field", ["repository", "revision"])
def test_exact_repository_and_revision_required(prepared, role, field):
    sample = prepared()
    artifact(sample, role)[field] = "other/repository" if field == "repository" else "a" * 40
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


def test_upstream_digest_agreement_is_separate_from_local_hash(prepared):
    sample = prepared()
    member = artifact(sample)["files"][0]
    reseal(sample, member["file"], b"changed and sealed bytes")
    with pytest.raises(launch.LocalPlanError, match="registered artifact"):
        launch.resolve_local(sample.config())


def test_missing_drafter_directory_never_becomes_hub_id(prepared):
    sample = prepared("qwen38-nvfp4-1", "metal")
    directory = sample.root.parent / "drafter"
    for child in directory.iterdir():
        child.unlink()
    directory.rmdir()
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


def test_unpinned_drafter_is_not_authorized_by_supplied_revision(prepared):
    sample = prepared()
    sample.spec.pop("revision")
    with pytest.raises(launch.LocalPlanError, match="immutable upstream pin"):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("name", ["config.json", "tokenizer.json", "tokenizer_config.json"])
def test_gguf_tokenizer_requires_complete_closure(prepared, name):
    sample = prepared()
    tokenizer = artifact(sample, "tokenizer")
    tokenizer["files"] = [entry for entry in tokenizer["files"] if entry["file"] != name]
    (sample.root.parent / "tokenizer" / name).unlink()
    with pytest.raises(launch.LocalPlanError, match="tokenizer"):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("reference", ["../model.safetensors", "nested/model.safetensors", "missing.safetensors", "config.json"])
def test_index_requires_same_root_supplied_safetensors(prepared, reference):
    sample = prepared("qwen38-nvfp4-1", "metal")
    member = reseal(sample, "model.safetensors.index.json", json.dumps({"weight_map": {"x": reference}}).encode())
    entry = next(item for item in sample.quant["files"] if item["path"] == member["file"])
    entry.update(bytes=member["size_bytes"], sha256=member["sha256"])
    with pytest.raises(launch.LocalPlanError, match="weight index"):
        launch.resolve_local(sample.config())


def test_whole_drafter_needs_config_not_tokenizer(prepared):
    sample = prepared("qwen38-nvfp4-1", "metal")
    result = launch.resolve_local(sample.config())
    assert result.draft_path == str(sample.root.parent / "drafter")
    assert {e["file"] for e in artifact(sample, "drafter")["files"]} == {"config.json", "model.safetensors"}


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_unsealed_outer_profile_inputs_are_rejected(prepared, kind):
    sample = prepared()
    extra = sample.root.parent / "unsealed"
    if kind == "directory":
        extra.mkdir()
    else:
        extra.write_bytes(b"unsealed projector")
    with pytest.raises(launch.LocalPlanError, match="role root"):
        launch.resolve_local(sample.config())


def test_same_checkpoint_mtp_reuses_exact_model_root(prepared):
    sample = prepared("qwen38-nvfp4-1", "metal")
    sample.record["variants"]["metal"].pop("speculator")
    sample.artifacts[:] = [entry for entry in sample.artifacts if entry["role"] != "drafter"]
    directory = sample.root.parent / "drafter"
    for child in directory.iterdir():
        child.unlink()
    directory.rmdir()
    result = launch.resolve_local(sample.config())
    assert result.draft_path == str(sample.root)
    assert result.speculative_config == {"model": str(sample.root), "method": "qwen3_5_mtp", "num_speculative_tokens": 2}


def test_unused_artifact_role_is_rejected(prepared):
    sample = prepared()
    sample.record["speculative"] = False
    with pytest.raises(launch.LocalPlanError, match="unused artifact role"):
        launch.resolve_local(sample.config())


def test_cross_source_shared_artifact_cannot_impersonate_model_repository(prepared):
    sample = prepared()
    sample.source["shared"] = [{"path": "mmproj.gguf", "bytes": 1, "repo": "other/model"}]
    with pytest.raises(launch.LocalPlanError, match="cross-source"):
        launch.resolve_local(sample.config())


def test_assembled_output_without_assembly_or_source_parts(prepared):
    sample = prepared()
    old = artifact(sample)["files"][0]
    (sample.root / old["file"]).unlink()
    value = b"already assembled and sealed"
    name = "assembled.gguf"
    (sample.root / name).write_bytes(value)
    artifact(sample)["files"][:] = [{"file": name, "size_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}]
    sample.quant["assembly"] = {"output": name, "bytes": len(value), "sha256_published": hashlib.sha256(value).hexdigest()}
    assert launch.resolve_local(sample.config()).model_path == str(sample.root / name)
    (sample.root / name).unlink()
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("name", ["../weights.gguf", "hidden/.config.json", "a/b/c/d/e.gguf", "nested/model.safetensors.index.json"])
def test_invalid_members_and_nested_indexes_fail(prepared, name):
    sample = prepared()
    artifact(sample)["files"][0]["file"] = name
    with pytest.raises(ValueError):
        launch.resolve_local(sample.config())


def test_metadata_limits_apply_without_hashing(prepared):
    sample = prepared()
    artifact(sample, "tokenizer")["files"][0]["size_bytes"] = 64 * 1024 * 1024 + 1
    with pytest.raises(launch.LocalPlanError, match="metadata"):
        launch.resolve_local(sample.config(), verify_files=False)


@pytest.mark.parametrize("name", ["flashinfer", "flashinfer_cubin", "flashinfer_jit_cache"])
def test_unreviewed_online_kernel_module_blocks_resolve_and_availability(prepared, monkeypatch, name):
    sample = prepared()
    monkeypatch.setitem(sys.modules, name, ModuleType(name))
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())
    with pytest.raises(ProvenanceError):
        launch._availability()


def test_attention_network_guard_is_fixed_not_caller_authority(prepared):
    sample = prepared()
    result = launch.resolve_local(sample.config())
    expected = {**sample.record["variants"]["a100"]["engine"]["attention_config"], "use_trtllm_attention": False}
    assert result.engine["attention_config"] == expected
    assert json.loads(result.argv[result.argv.index("--attention-config") + 1]) == expected
    sample.record["variants"]["a100"]["engine"]["attention_config"]["use_trtllm_attention"] = True
    with pytest.raises(launch.LocalPlanError, match="unreviewed"):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("eager", [False, True])
def test_eager_policy_uses_pinned_compilation_config_not_stock_flag(prepared, eager):
    sample = prepared()
    sample.generation["enforce_eager"] = eager
    original = dict(sample.record["variants"]["a100"]["engine"]["compilation_config"])
    result = launch.resolve_local(sample.config())
    expected = {**original, "mode": 0, "cudagraph_mode": "NONE", "max_cudagraph_capture_size": 0} if eager else original
    assert result.engine["compilation_config"] == expected
    assert "enforce_eager" not in result.engine
    assert "--enforce-eager" not in result.argv
    assert json.loads(result.argv[result.argv.index("--compilation-config") + 1]) == expected


@pytest.mark.parametrize("location", ["source", "file", "engine"])
def test_unknown_speculative_settings_fail_closed(prepared, location):
    sample = prepared()
    target = sample.spec if location == "source" else sample.spec[location]
    target["unreviewed"] = True
    with pytest.raises(launch.LocalPlanError, match="speculative"):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("mutation", ["file-count", "file-name", "file-size", "total-size", "duplicate-role", "duplicate-member"])
def test_typed_closure_bounds_precede_filesystem_access(prepared, mutation):
    sample = prepared()
    model = artifact(sample)
    if mutation == "file-count":
        model["files"] = [{"file": f"member-{i}.gguf", "size_bytes": 1, "sha256": "a" * 64} for i in range(257)]
    elif mutation == "file-name":
        model["files"][0]["file"] = "a" * 193
    elif mutation == "file-size":
        model["files"][0]["size_bytes"] = 1024**4 + 1
    elif mutation == "total-size":
        for item in sample.artifacts:
            item["files"][0]["size_bytes"] = 1024**4
    elif mutation == "duplicate-role":
        sample.artifacts[1]["role"] = "model"
    else:
        model["files"].append(dict(model["files"][0]))
    with pytest.raises(ValueError):
        sample.config()


def test_metadata_aggregate_limit_counts_every_role(prepared):
    sample = prepared()
    for item in sample.artifacts:
        for entry in item["files"]:
            if entry["file"] in launch.LOCAL_SUPPORT_FILES:
                entry["size_bytes"] = 64 * 1024 * 1024
    artifact(sample)["files"].extend([
        {"file": name, "size_bytes": 64 * 1024 * 1024, "sha256": "a" * 64}
        for name in ("generation_config.json", "hf_quant_config.json")
    ])
    with pytest.raises(launch.LocalPlanError, match="aggregate"):
        launch.resolve_local(sample.config(), verify_files=False)


def test_unknown_host_ram_cannot_bypass_quant_requirement(prepared):
    sample = prepared()
    sample.quant["min_host_ram_bytes"] = {"a100": 1024**3}
    sample.record["variants"]["a100"]["engine"].pop("kv_transfer_config")
    sample.machine.host_ram_bytes = 0
    with pytest.raises(launch.LocalPlanError, match="host memory"):
        launch.resolve_local(sample.config())


def test_installed_module_digest_must_match(prepared):
    sample = prepared()
    with (sample.pinned.package / "engine.py").open("a") as output:
        output.write("\n# drift\n")
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


def test_rtx3090_pinned_ep_uses_same_tp_world_and_metadata_projection(prepared, monkeypatch, tmp_path):
    sample = prepared("qwen38fn-nvfp4-4", "rtx3090")
    tier = tmp_path.resolve() / "kv-tier"
    tier.mkdir()
    monkeypatch.setattr(launch, "KV_ROOT", tier)
    monkeypatch.setattr(launch.os, "fstatvfs", lambda descriptor: SimpleNamespace(f_bavail=3 * 1024**4, f_frsize=1))
    result = launch.resolve_local(sample.config())
    assert result.plan.gpus == 4 and result.engine["tensor_parallel_size"] == 4
    assert result.engine["enable_expert_parallel"] is True
    assert result.draft_path == str(sample.root)
    inventory = {e["path"] for e in result.plan.quant.files}
    closure = {e["file"] for e in artifact(sample)["files"]}
    assert ".gitattributes" in inventory and "README.md" in inventory
    assert not {".gitattributes", "README.md", "LICENSE"} & closure
    assert "hf_quant_config.json" in closure


@pytest.mark.parametrize("name", ["README.md", "recipe.yaml", "unreviewed.json"])
def test_projection_never_admits_unknown_actual_extras(prepared, name):
    sample = prepared()
    (sample.root / name).write_bytes(b"{}")
    artifact(sample)["files"].append({"file": name, "size_bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()})
    with pytest.raises(launch.LocalPlanError):
        launch.resolve_local(sample.config())


@pytest.mark.parametrize("field,value", [
    ("head_dim", None), ("head_dim", 128), ("indexer_head_dim", 64),
    ("indexer_kv_heads", 2), ("indexer_compress_ratio", 3),
    ("indexer_n_heads", None), ("indexer_budget", 2049),
    ("rope_parameters.partial_rotary_factor", 0.5),
    ("rope_parameters.rope_type", "yarn"),
    ("rope_parameters.mrope_section", [16, 16]),
    ("rope_parameters.mrope_section", [11, 11, 9]),
    ("rope_parameters.mrope_interleaved", False),
])
def test_qwen_config_rejects_flashinfer_dependent_or_unknown_shapes(prepared, field, value):
    sample = prepared("qwen38fn-nvfp4-4", "rtx3090")
    document = json.loads((sample.root / "config.json").read_bytes())
    target = document["text_config"]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    member = reseal(sample, "config.json", json.dumps(document).encode())
    entry = next(item for item in sample.quant["files"] if item["path"] == "config.json")
    entry.update(bytes=member["size_bytes"], sha256=member["sha256"])
    with pytest.raises(launch.LocalPlanError, match="Qwen"):
        launch.resolve_local(sample.config(), verify_files=False)


@pytest.mark.parametrize("ratio,sections", [(2, None), (4, [11, 11, 10]), (8, [16, 8, 8])])
def test_qwen_fused_predicate_accepts_only_complete_local_inputs(prepared, ratio, sections):
    sample = prepared("qwen38fn-nvfp4-4", "rtx3090")
    document = json.loads((sample.root / "config.json").read_bytes())
    text = document["text_config"]
    text.update(indexer_compress_ratio=ratio, indexer_budget=ratio * 512)
    if sections is None:
        text["rope_parameters"].pop("mrope_section")
    else:
        text["rope_parameters"]["mrope_section"] = sections
    member = reseal(sample, "config.json", json.dumps(document).encode())
    entry = next(item for item in sample.quant["files"] if item["path"] == "config.json")
    entry.update(bytes=member["size_bytes"], sha256=member["sha256"])
    result = launch.resolve_local(sample.config(), verify_files=False)
    assert result.draft_path == result.model_path == str(sample.root)


class ExecCaptured(BaseException):
    pass


def helper_input(monkeypatch, sample, args):
    monkeypatch.setattr(launch.sys, "executable", launch.PYTHON)
    monkeypatch.setattr(launch.sys, "argv", ["slimserve_launch", *args])
    value = json.dumps(sample.config().model_dump(exclude_none=True, exclude_unset=True)).encode()
    monkeypatch.setattr(launch.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(value)))


def test_helper_exec_is_fixed_offline_and_preserves_parent_token(prepared, monkeypatch):
    sample = prepared()
    helper_input(monkeypatch, sample, [])
    monkeypatch.setenv("SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN", "parent-private-token")
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("VLLM_DSV4_ALIGNED_Q8", "0")
    monkeypatch.setenv("VLLM_PLUGINS", "mps_native")
    captured = {}
    def execute(executable, argv, environment):
        captured.update(executable=executable, argv=argv, environment=environment)
        raise ExecCaptured()
    monkeypatch.setattr(launch.os, "execve", execute)
    with pytest.raises(ExecCaptured):
        launch.main()
    assert captured["executable"] == launch.PYTHON
    assert captured["argv"][:3] == [launch.PYTHON, "-m", "vllm.entrypoints.openai.api_server"]
    environment = captured["environment"]
    assert environment["SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN"] == "parent-private-token"
    assert environment["SLIMSERVE_KV_TIER_DIR"] == "/var/lib/sovereign-slimserve/kv"
    assert environment["VLLM_PLUGINS"] == ""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "VLLM_NO_USAGE_STATS"):
        assert environment[name] == "1"
    assert environment["VLLM_DSV4_ALIGNED_Q8"] == "1"
    identity = json.loads(environment["SOVEREIGN_SLIMSERVE_IDENTITY"])
    assert identity["config"] == sample.config().model_dump(exclude_none=True, exclude_unset=True)
    assert identity["source_commit"] == SOURCE_COMMIT and identity["profile_id"] == "dsv4-q4ktail-2"


def test_check_verifies_inputs_without_loading(prepared, monkeypatch, capsys):
    sample = prepared()
    helper_input(monkeypatch, sample, ["--check"])
    assert launch.main() == 0
    assert json.loads(capsys.readouterr().out) == {"validated": True, "engine_model": str(sample.root / sample.quant["files"][0]["path"])}


@pytest.mark.parametrize("failure", ["oversized", "invalid", "unknown-argument", "wrong-python"])
def test_helper_failure_is_bounded_and_sanitized(prepared, monkeypatch, capsys, failure):
    sample = prepared()
    helper_input(monkeypatch, sample, ["--check"])
    if failure == "oversized":
        monkeypatch.setattr(launch.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b" " * (1024 * 1024 + 1))))
    elif failure == "invalid":
        monkeypatch.setattr(launch.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"private-path":"/secret"}')))
    elif failure == "unknown-argument":
        monkeypatch.setattr(launch.sys, "argv", ["helper", "--model", "/secret"])
    else:
        monkeypatch.setattr(launch.sys, "executable", "/untrusted/python")
    assert launch.main() == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "SlimServe local launch validation failed\n"


@pytest.mark.parametrize("backend", ["cuda", "metal"])
def test_availability_reports_installation_not_execution(pinned, monkeypatch, capsys, backend):
    monkeypatch.setattr(launch.sys, "executable", launch.PYTHON)
    monkeypatch.setattr(launch.sys, "argv", ["helper", "--availability"])
    monkeypatch.setattr(launch.platform, "system", lambda: "Darwin" if backend == "metal" else "Linux")
    monkeypatch.setattr(launch.platform, "machine", lambda: "arm64" if backend == "metal" else "x86_64")
    extension = ModuleType("vllm._quixicore_C")
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    calls = []
    def verify(module, actual_backend):
        print("simulated extension import diagnostic")
        calls.append((module, actual_backend))
        return {"library": "quixicore-" + backend, "version": "base+slimserve." + SOURCE_COMMIT, "backend": backend}
    monkeypatch.setattr(pinned.provenance, "verify_extension", verify)
    assert launch.main() == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert calls == [(extension, backend)]
    assert "simulated extension import diagnostic" in output.err
    assert result == {"name": "slimserve", "version": SOURCE_COMMIT, "adapter": "slimserve-runtime",
                      "variants": ["metal"] if backend == "metal" else ["a100", "rtx3090"],
                      "kernel_library": {"name": "quixicore-" + backend, "version": "base+slimserve." + SOURCE_COMMIT}}


# Apache-2.0 upstream fixture, QuixiAI/SlimServe at 44a7d1a21851c7164c098c93fbc1baa12ab99847.
# Unmodified source bytes compressed solely to keep this two-file change offline.
# Regenerate from raw.githubusercontent.com/QuixiAI/SlimServe/<pin>/slimserve/<name>.
_PINNED = {
    'registry.py': ('7eca6ff164337e457e72f34d6c1f2da15482f8a29b3622e0fd172c96ff863dfe',
        'eNrVO2tvG0eS3/krOiMcRC5IWnF2F44cLqKzZceI7DiWkt1AEMZNTlOcaB709AwlRqf77VuPfsyLkpVsDjh/SKh+VFfXu6pr9sTp'
        '+5f/mpzEC5VpNXkTqayMl7EqDsXRWi5WavJ0ejDY41Wv4kS9yNfbIr5clWfqpjwU7k+xyLOyiOdVmRdalLkoV0psTk7einWR/6oW'
        '5SAIgjMYS9SlTCawehlfVoUs4zwThbqMdVlsx/BLRmJZ5CluW8J5evqrzrPpYND4W8SaDpBVucqLuNxOxbu8XMXZpVipQok428BF'
        'tJCicdBz2BTrQZpHVaJEniVbOFDnyUZpMTQHjMU6keUyL9Kx+FTJrBwBNLgPgVoUqlS4IBMyiwbqBn7GmR7DEsKn0qrYB9RUkcLg'
        '9WpL29J5nPFFAe0sL4VMkvxaRVOkyWBA1w3DZVVWhQpDEafrvIBFGSylbXowMGMLILf9jXSwv3PNUCJZykUitYYLmSk3NBbA1yTi'
        'hcsqW5R5nrhlSVGFC2Q4z69luUriuZ19D3/yRLldI5XN+FG2NfjrJE7h8hs1xcvb+VWVyiycb0sFd/hw/PrN6dmHX8L3R2ffiRkB'
        'HcK9gehhOJpex+UqzGSqhkGD18FoMBhEainCKos/VSrM5yhPw7WMC30oEpCc87JaJ+ocRGiMKF1cjMTkHyKKF6UfOxwI+AfsrhKQ'
        '2+YcIHN7R/PAd3GlQBA3MqlQjgQfQ5P4L17iPE4YUG6GwMtYK/Ez7j0uirwYLoMIUIsXEsTG7Gtc7lDcwvAXxR3c0gEhwOcwjogR'
        'IgZ3EBB7LtDkW8ezYSpvdPybmn05YkpZhRruJARSWzQ4Ms3XKhuqbJFHwOFZUJXLybNgJKQWy/VhDTvCgrQyyWU0XK7HglkSEq3C'
        'VZ5fzZrMsiw0dw/jSDNqxD1AzbGHgOPosHaJcycRwYUDZbQ0LOMyUUP756GALQQa/s9AwRIUW6BkE6BZDxCnl6p0+0d1PGjneUAn'
        'BBfIe4alEq0cAm18LoHX96EDOv/PlQSNi9O41GSRhF8eXK4rHYhhFGs2NgtZRHokQC6DVKV5sYVJoC3ocjSaov14/BURmJH3+kVp'
        'UYDoB2ODR4fY8yRfXKmo/37if8AKZ6p2zW3zenAZNGqCDIXYqnKMqOAesJUKzGOJK/74rSyj+EYarCjcBe6kqzWaJRWBWM9m9b87'
        '8o1I7aAQwwOkJFonAEtXsqDwWjvpFkaqlHHygHiQm8zByhLN8qX42IbzcUyWCl1OqrSWlwp+g0zpMl+j1yuq7M+QDXNzvkRAa3dK'
        'Bots0FJ8t8oZAiLBWHy2QKGwmN1NcQL3Kmu++z7BMlgAUXajbwXJTHQExIzTsMVnJiIFahvPVe2CDpLTAsTZO4IGM+6Tw0VeEMp2'
        '20YWMcQnsOvcQrigpcwmWMp7Wlpg9t+jGw5jC+hxqlI71S3tKE0PFg9oFIIZ9YuTU6vPkionTiegYxPiCAVyJkZDxZL2hIleqwUY'
        '2wUJGUY+aB+dIH0G4+/j20OqN6yRus1Jp4VuDWpjD1XvWdejjp5mjQ3mvkj/b11EOYTI7zeVzc6KSo0GNCR+xJCZKYyxHBGe/iIn'
        '6v+kmPAQQ2sW2SpNZbH18+TtTXTXjF9YxtM4C9FF1UM5AOYn2VmG5pwdi1a5LsNCpvcsw2A6nSfbTszItmlAi1AmC/WpiguVgr0c'
        'apUs+6QPoDZsWs3gg1yheFGyoox1b0QGr9//BPF/BRBgjvBF18DXdALZMjXNYISdnokiWoErCxziPW1Tryc+6tuC3GgtdbQxCU+Y'
        'Z16kiUburwat/DCzGMjmh5qMRYrOxIGfbrO0sYCYMIe8x98eUI/TKoU1dJM6F7t3BtLa9cCbJh9rNHklIUD0GMkNWog63p/JIw40'
        'kQZ1DAjeNxaRz0BgT3wHRJnkyyUG7C4BEUMwFFO0dXGWgZl9f3IsSjkHc7ZWBaQy2dVIZMq4OIajt7pUqfhw9JZiDxRJxN47YjUV'
        'P2BarTK4zAKAkvvFtQuQAVXUYAEamxjMJiAAJgqSBFjN0v9cHAC1ZKZFlV1l+XU2Jpf5mypyPHrqRRDY3OJfV6vvF18EPGzlcB6o'
        'ydcpkICEvyVcfUPf1Pc7uA/bzffgf5xj+gGOW1aJL0+AWaa6iQvs6t7Hurs+M9uxqw09GzRUjDfkFfAthOSzBoTG2gaQpqhCcshW'
        '30Sbl3HWv1Zlm/o4Zn0MHlxsBWjFG9iHytkeDfONKgqUlV64C4h8QxBLvBpgfi2Ly/6FwErnU/B0kBkqiQzBREnIqcOlXJSgdzNc'
        'wZKyB8k8eetJojYqEVEhlyVoB4m1i9vgMMDwEgR3CVKuxVwurkwNzEBhEhog9mp5waJsAoLQj+9wNoCw9znfAvtBUcuts7J+Pzmg'
        'nty/43zqRryLBnoammL0TVjhpoPRLkzSPFJJGMU1RLDa0zmXqhdhkeflcCSe1M86DyA0kQQjuNh1DKUmVEDadY5xrljOAruZRWBj'
        'TG2SBRWrG5MJodv2oZ2LI6tlaVNHuQSjlem80C1XuieORG0WxFMtrtY5+qGYjB24FxIzdN96hek9SBKE2VyqbMFCE1PmVyqLwfxx'
        'VPS8jn8prxTXQj3cuCTcdzt4y53OdUmdpy7keRiCZVpz43mQV+W6KoOLwSN20+XODy4gIQJ2IdetbWQ7xxW145uFWmOgPnLm8qhZ'
        '5eVU2JhuU5j1FWYqfEkqNJOv46SDjSrlFg2ZbAgUZaBYXqYLMDPAmmzA6Z2evHl7evzh5+PwxdGL747hFKwKl2gl/vcJLddOvqw1'
        'A23O9RSsYlwAAiRiLTCBy8Tsno4GURHVzo6mkMiAxGAZetioZOGy6SpPFelZwBjZG4dE/2G/mR/7OJ7IUQvwC3mNbtfoKwHBZBTX'
        'X9QPpy3eyeL8DP/jgzbyWjOA54ptfo78Ks/Rz/qccW88a/6ozxOLeNaUD8f1wI9iVp62f7VX1KM2XMl8as8EJnlr7m2GBs3dzbnu'
        'fqtKfpcdCUbjRi7ck3l6bvWVfo1NtCrBVpQLSuj4MAtBJQEVNuEHjvjSypSt8ZFLfkHTWCFuOByA/7tSx41RTXD3bHY1WCgwwlyU'
        'WuVJxKEtOjeTKWPw6fZXYLIL8dGmzB+nAlGPI8AHhRyfI0AHFxLEX6HCN2IcsOx05ouTNybE1FgJglFCBvBmAWX7iaaYQlCICNgi'
        'c0C18qVeos+YbLJFaTj62IBiLoHEpEqVI5qhfD1w052qnKtyu5qR42mralR7kuBHh4aRbBjuZWDCaMexWw/2i+LuuZAbGSdoDQ/F'
        '7f5Y7E9/BXdl5UmP7oJaIFsvKdoV5x5eQ/Fv//IXI1y1GhcHYENTW/fFkNGdEWhL2c+o5twr3r2MoMKgzIRCOqEpQuFGxsEizZ5D'
        '3QB+nfLpwwWezo16Mo94add38seH2Vhj2531b75KBo7vdsejyAh4HLSAgSJAMqFpW43nLQAjIuAaBY+vt+8YuX/RFYxGBdCwk/1C'
        'uLT3abN1sCP9vyfP35nj+zcl8jleIo7BQ26NeeK6ii8hc2oFyZarIzcqL48p9rEvbKs1j1Kd1tYAeSgwxSx0PPUKITuiOkXPHaE/'
        'uVTLvlfyrYBDw4YjZ8fNDMRfuKLtqxuVDRPD+VKN1xlEaNzgyLjFBIZ0YSOKFNOh0AKwdOrGFo9Sabt4H6T2GmPJEp2VNg7pewgL'
        'MZyX/LiNgeBcLSS4CVN70NxdENMqtAlg8deKoulCGX3CFgKClqhliakCXQRmro1tgODQu4OVRIfC2JgKbcPOP6Jmby2mz1w4wA+Y'
        'ZkMGdW5HL0bj+spNcxkHDDhs4or6asqCrRWur+cJ2nHegG/TYxIQ2GlxaY7XAqegN3nvw7F/YRdrTKoM7UQqt+Txt47292flXOeq'
        'gfp0rbKvnk2yzXL918mXh2T+37756uDgX+6MQlXa5FU+fwO5e3v2vgZoBfI05u2qlInbTYZEm+c5qSuMK16+glxm9dTiOpp2qQUZ'
        '9aHoow6m2kwL9JF74tS8RczVSm5ioJUi82YNFOzF0hk4Zop3HF0JIepCMAniHsI6O3r38s271+L9DydvXvwihk8Pnv59cvBs8vTZ'
        '6FCcJnF6SmbRvn8cnfzz6JdTEn1ZlTlkxPFigNU8tYxvKHmCVWM/J7DNhDSQxzGfhSVXCEtlGHREPPru+OfjDwCJCH9ZKBWBRZbp'
        'GvdNwS/W4QsZ0eP5ux9Evi4nEKoeWhrYV/k9q334DAIQ+ayQ8QwtnLKolMGpFhHDehQB7FzQA8zBt9xbhOXYqXhlXl5zNBxIT+3f'
        'i0g4XYJHHUb0ngRQKEY2FSBsj9E6nsdJXPqcFBWwHpKY1BT2vszJ2RcqwQdQm/sb/dN8REYFBdinIajY494rDShlJWxyS5t8Ej+8'
        'ekW7V9s5YCyGqUzn8snrl+9GjK8eA6jrVQxWT6/i9RoEmbVnmU2W62eTZxb3g+nBf3HqLFYYWmBVGOmKZcwJuFMsTBBPJoRBktA9'
        'iGVYvxNo/abiv1W2WEESd8W11SiWlxm4mXihUSxYoRCtfW34m6aKijpWUMQQC4EKjsdXlC+nB5Dtlvk6XAOCX//N/HElnh6MsViN'
        'O+H2BAsYGFULw5IROE9ETtShHbDCvJXgSL//GSm5iLXp6pKYtkxcC1meTeb5zWQjkziSGJwtVnkMgcEwJ2h0JOvZ15ODp9j2Bodp'
        'y3engV+Lorz56uDrAzFffvl3kK0MoYFaSivcIG4Aigxw/Jsig5ga/MgYeqEy6s9FLCfyBsOYxazJ2if2cIoTYQgM6V/Tr0buhHJV'
        '5NXlSlxtuA0pjDAdew6Azqpinv/IcQnHqcA+Dsb9ZvC6U/ESzSEX4HB0+PJ0DexvgBgRgZNriYrG0cmh0U6kvmM9kBDbYjSbWFh7'
        'ib15JcDCm5YSXEFJeSjN1/gnl0vKDcHcojFiznMFi9oaQYWQOsf2TWPOFiK0ZI9CvFQIZIh1KLMQX5PjRVyGjv8h858FKLD2L7Bt'
        'jC+VWp8qdfXk+ziNhS1lCw26tViN0cGT7fL7sLC8J16fvH3yI7AMg5nnbhvYfFCagrWFAj/4ARYgt3yYDkKsMYHND18evzr66eTs'
        'tK8hbmBiCzoazXmIltzcBBwVvl6wU3KBQF8lHlbe+hubbd0r8YQpgAS9xtqdaRNF8zjyuLSi9aJoHjB8s8kfTD4a7zhnGNJJehlG'
        'j2L6SHEJpaGmfZT6RGOnKZSNmJj2I2L7EWeuyUii/SNiKHTAXAi2nicCcUYnv4JYNUVvEBO3Nb7IgQNCcKZvbMK383Hsx/p1/XGX'
        'LE2gQxDn9FYw7s2FXGY662112dUW4/b9/0mN3UndjDhjL9Of3N3zCvxF7RUYjTWliN8YcL+bMrhdi1uGcofvt9pGZNpJlgYJgkV4'
        'YuM6RjHAjOJ9dqR47f4R/55Yp4FNfz8rafYgLlxfCazmtBc9CqHUSUcshWm5la1W/vt7imh87i2C/aK4IyG4NXD3STT2L3pkqr/E'
        'ZvcxPn0VFTC3Giz8TPTl9oPuIziv/7xncCxG/eceux8llAZNotcdNiBogm17FPCZDomliWm+6QBxaFOWZRobaiFcBsbUes6HNfRQ'
        '0RtijhlMG1Zjc6vA0WENyhbQy1ylt27CitasnIz6ek8MkD/YffIA3Xtp/zhj2STY6A/1HOmUyjIo27q6vHQRVW83WouGDUAy2sRk'
        'QPqui/ncrTmqhb59+UybrSn2HzXfBFluLc3SyI95/KPPS6wwcfctxIK7CPS7+WNMdl0wnUQjTsani6AHlPHzHQuPot+EWKdtx3oZ'
        'vJjMO0Xg0VrfVlxzrbuuVu4UR+/BGm6OSgjWy3VtMYmElbld0mZp3PGAe/ye6TtDORt/orINSOKvlS5T+vbIdldBVoN0FfMqAhPs'
        'CkIGln2fkEUZY/sJxOmKNbIsZKaXmINi0Jhg+kJLR1NxtIYEA3jOVS/fZeJCKa5a8pPbFh+jqSYh5jnk6nOp2b671b6Q6iLisPZI'
        'fW/xkgtVzT3YT3x7N6Ip76dshTnsvmV3v7oZNpfagibVPm11cBqDW9DDUdOqxEsLpc9E+hDGl1Kn63w9pONxfVNl0QY8DKH7rc6j'
        'brV54Er+uE37LMP4M1tR43YRX3ADuaIXjbkrARFazXqhrcS0pGhae+PVij5VUVpxwYsriFzgAvnM1yUmzNMd31B18s3uPTtMgRMN'
        'xswb/PRtGkGijD+GBHpkNNL2OEBe5U2OV+tZTcNbHQdOtDtdB7azwIes3e4CS6iZMxuNvtWZ8f4epgtkZ/5ne9pMjZsPPjM2nX6U'
        '6TRrk62+YDNrCE7tIF90n2G/3bDZNu5ng9God5fXdHdEfyG/dmhfbWBG7wGN03tLCN33AHqrcIfzy0XtsG4rWwfRvLAb3PeFu6KQ'
        'vufncbdEsOurFfqcB803OLnaC6R1xou8SsCUL8qKHiGxJ8J/vQLWDLJR6pOAO2fK1AXeYn3HeHf7TanTVfshKr1truPFFRW96POY'
        'ZgGg2Sxhti2rxAZUVNe0URq/DjaKABQDzeoPvJ8buZnQGQE8+DlJKm+GuHCMRmWWyHQeSVu2YaQY8BRdTYuVDzPR5C51/jnGvcPM'
        'sU5XiFWuiVsaBF0vt0QzwsF/j8WNBearLPd27HLcvnJJKx+mkADLo2PTfIAWtL9NpGtHgbLNsMt0JdgTMBzn380Quh4PuMf+msfv'
        'rDYruJoh/jETnZ7n3vZBulidvYGLbYxC2K40uiJLFdbUyLz79/3WNyGth37PrxTiMTHHKhn4LlA96ipNwJ9EkMtTO/xPH07gxouk'
        'AlvtvxDDIIk+D8tcQyqOhVWRGGJcA2GwEt2HDmw998x0XCRwtWZHTyEGNpXrtcqiJgNvO+GH7arpTASI3SFGzIjr3ZNb27Mhy9X+'
        'xV3Qs8N32R42b1trv+3ZVuQJ1oC5kbAF966VJ3dJ0GgnprYr++x8OGjIbWj4UPsG0XFhTEwa/Wkk9Aj8XxDSkOF+SqLvclLpu7Q5'
        '+fZD4IM5eXWmNObmcFJc7rwShzMaMq3NII3g7P88ecSjzvct74CEjyUpAfhcWnoXfz89jQniyw7+DY1eclo='
    ),
    'engine.py': ('189848aa68a9d89440ab6fab680507860fed8edcaaaa2dea715ad2aaca7b56e8',
        'eNqlVl1v2zYUfdevuFAeZre2OuzRRQYUXQYMy7qiLYYNgSEzEm2zpUmBpJx6Xf/7ziUli0nb7GF5aBPyfp577qEu6O3rn/5cXqtG'
        'Gi+Xv7TSBLVV0q3oRSeavVz+UH1fXCSrn5WWL213cmq3D+/kx7Ci85/UWBOcuu2DdZ6CpbCXdLy+/o06Z9/LJhRlWb7rnSFBr7Uw'
        'pAyMBLneGGV2JM1OGbkg68YbL91ROsQ9HIRpSeO6Kop3COvFQZKT3uqjbGEXAiJ4ap06Srq1YU8H20q/II84SGj8HQJtnT3Eqpq9'
        'CPTm6vV1wXG/YhAztyip7SyK4RJkuuW6I0SN0PrEPW/VrneyLVL9FXdZFNG2rrd9wF1dkzp01gV4GxtEUNb4ohjO3ntrxt+tT57h'
        '1DEkw+kLcxoieq0OsbbKyZ3ywZ1GG0a04DGNWHAbdrtl0IiH0GjhgZCVnlADBfFBkvD0QZ7urAMIbtcf0Jd/zp4n/C0RLU1g6dEy'
        't9rILnhizHYyAG3bdUDpbi8N3fZKt1xzHOwSI28k8o2g1G+v3vxxVf/+6vovumQk/2a2hVlB+PkU/+WfMgFf8/R0bTDlcjFdtnIr'
        'eh1qHl8d5KHTIsj6wx1q97mdNOJWy1qAiXWwVsPBgt6ZyQUwJdGpeqDYVovdIuKC8q9izS8QlLAHun2eYKiR5gjSidZncVCDqVKX'
        '1CoHmoMVYB27MBw+KK0ZLk8qVFONqS5wqO6Eg/FQ3OdiXhQFGkV1nT7V0hxnnGIVBzyn5Y/0yhq5isZg2tXHOH0eNiDfYj2/Y9CP'
        'ylnD46yYPGEfB6O59hNvC92eokuPxHSnjK+YtBxyi+0DJRZ0FLqXmOTY4LFSQNzP5qtzD9ZXQ6YKMYfhzCbvsZPad7LpMSssZ50W'
        '5mFPrWrCDei84HLX9E/WpNrGucQyskBTGU4GFhX2iGdsBI7dc7Auj8Wn3/b/j0VrWBNrZ20oorm2mCLyTeezOT2LOW7KeFmDF+U6'
        'GqfuVw/7vaRPn8cCeYa0uowBKvBmVvJJmeHeOrEN8IH/LGV/Fr1uyk6EfbmeR0upESxeV8pzCbPHQow+PgP2gq6taBOJnYgSfxbI'
        'fX+7QsvOJSJBrQyUwMmj8hA3XgAcZ5FiQtbyqIAs1NssGuiiWuwypNyIzu9ttinnUiOgTnZ2wHLA65zzHmbjaY7bNICb6Z7BH/8o'
        'MjpkmhTFqFylSiYNefIkBctPUpFJDsp1fvOQv7WF7jjFj9Sw+GlZku8gao+vyVkEfn2o4XGN7wvxBm/AZhEvgvSDioPyS7wR5Bun'
        'oOxVYvTbQbrObwVkAmNSfpWe1s1yGVdjE2Oc31NkOnSBdjx6Z/vdPgbbTNq5GWjhoCseuSGUKCB7Qon5W41dxf8TDrwgZywhMKtB'
        'nr6pVvHRGQSLSQKj2AksspdoAH7KczPMej2sxhAN21/zgs1zffmWqs3HRY52KiE4yVme7MsIKTOOcyom84EfE5wZORa0tx7fYpEa'
        'rFIr/oSKlNEQL6bMxJaX2edURHBz1PqQ2oxfO76yHZ5QVU0P5Ob8QMRH8JJuyuUyYbX4KlILggEXVaba4t9c2WiPX+frx5+cfIgT'
        'ePxUowCEK+kpO0Kh4dDIWVmXnKac5/KAxTY+CHy6zGL8BThs9QNVgF28vH86tgscgEc748xT7KivX0af8Y4uIuzzB1kidE+BXfra'
        '4O++qu0PnU++83UW28tHfRnC3On/c/KcAMBmUZYDL++Vy/dD4oGi7F38C7o56Gs='
    ),
    'term.py': ('833d89272490151216748843db6e66bd9ac9f9b19cb61d557c1a9b90fd3450b8',
        'eNqVVWFv2kgQ/e5fMXK/gAQ+IG3VkhIpTWguKglVgq4XXSW64DHsxeyudteAhfjvN+t1MHC9oLOUyLyZeTM7O2/8Bh6/Xf/ZHPAp'
        'CoPN2xiF5QlH3YVLxaZzbHaiVvDGe33hKV5JlWs+m9sRrm0Xdj9hKoXVfJJZqQ1YCXaOsBwM7kBp+TdObRCG4Qj1gguWgsysyiws'
        '5ZRNspTpPAqCoUAiSWWmQaGGZy5ikAks0Bg2wwYwpVKOhIk0h9UcRZEiRmOJ0nIpgBtgMBo9NcDIgIHiitx1VhhUyjhFUNER3KBA'
        'zSwZrXymc8NMuoqNjamuc8Al6tzOuZgBpgbJiibwdtS6Qbn5dO44V3NmIUVr4KdJ+cKgXiI8nzXXa9N8D00FURTBBTBhVqgju7Y/'
        'YSX1c+RaEQSJlgsYj5PMZhrHY+ALJbUlbyFtcR4TBCUmzcubyY0PtLlyBZawu4zbYRBcPV3eQw/CH+v25K+z94swuHno9/egDkFP'
        '/cFg+L3Czrzb0w752CJk+HB5f9OvvD6cvzvvtD6Q5aF/XcFtAj4PBxXigOvbu/H3329Hx+HvWkX4Y3+0MzgkCGJMqKFja/Naorrl'
        'cerQvICJlGk3AHp4Qn2IUCy5liKaoa2F98Px1XAwfAjr3sU9GqmfAr4wuroCtDr/lzVRETfMpasXJlxPUVmoXVo/w9jXWtJV/8HS'
        'zL//VwZfu6LhsjVbSMJYCvSDXP44OhJhnuylmHDj3bcbx7DdFB3ahu7Eu6bU/Sw6hzLnGBfc1g4SlVr5ddp7KdDnVdpV62veycsT'
        'uag6/SOl9xJFL2lm5r2RzrD+ck0ikbX9REfkvqwwbFTKpaGNvHheSIxFdZrETfOrNPL5NEkhgFdZSG94mser5lWiFdPiNJEXFrU2'
        '3C2NLmzKsG34K+KE8fQ0Menyf7HG/PDYbgZieqWxIHl2jnLs1+BFoxmniXzM6TIX/XUxi/FuTGjrzzR5H2WIHV8hakpRKOgoDa1G'
        '9yFIuaAWa1xpbi2tetrdtMGnSNvf73jak7H7JKCI3TeBPgV6ReVEbrWW26LUzt7BKw1TWLGCRKEyV5XXVxldqSQJf+iNG8Ttrpml'
        'Pov19ZU6S1w9+itVU6U7VE+RlVJ0jxLs5unVaN/UebZgYjzJLZqaKC7qcKEkUlPnaBYakAluXddqtTZ8+gRvWw0IR/xzSOL2yJlD'
        'bvaRjkPuHLLXKOqOgIuep61g95gpS9G1UcBv3n5grrabd+xGrWQLG1eY321lPJG3Wy3f/j3nduV8tCxdunar89b7fKWCg38AWbO6'
        '4A=='
    ),
    'hardware.py': ('82769054cf46ae2bd3836b3207f7447bf626fa7500c00f09478b84502ad127b3',
        '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'
    ),
    'profiles.json': ('d97536815d40e264453b5b0124d96a35f965695d9ad08d19b24dd298f1b58211',
        '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'
    ),
    '__init__.py': ('2d3c329b32b0f8dd0f266837fd62bdf95a6842b9ccca4d23a1dc58eaf241368e',
        'eNo9jEELgkAQRu/+ikEPnYzo6C2KIDAI7NDV1m91wnZkdhT79y0EHR+89wpqbqdHWbNDiCgvHYKxZ2hFh6l1A8r9dpcVP+vMI44y'
        'fZT7we5YraI/kpNgys/ZRCOZkA2gpa6vNKm84CzL87wZ+d1AF2wieU0BIXQVteR5RUcRRuLJEC1RGnruZ22NJcRtyrMvP+o7zA=='
    ),
}
