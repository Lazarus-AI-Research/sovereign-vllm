"""First-party workers for the pinned SlimServe V1 engine.

Only the load hook and a read-only RPC are added. In particular, Metal retains
upstream MetalWorker's MPS initialization and non-CUDA weight-loading path.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import plistlib
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path
from types import BuiltinFunctionType

import torch
from vllm.distributed import get_tp_group
from vllm.v1.worker.gpu_worker import Worker as UpstreamCudaWorker
from vllm.v1.worker.metal_worker import MetalWorker as UpstreamMetalWorker

from lazarus.appliance.backends.vllm_engine import _canonical_torch_uuid
from lazarus.appliance.slimserve_provenance import (
    KERNEL_ENTRYPOINTS,
    KERNEL_MODULE,
    ProvenanceError,
    installed_provenance,
)



def _on_device(tensor, device, backend: str) -> bool:
    if backend == "metal":
        return (
            device.type == "mps" and device.index in (None, 0)
            and tensor.device.type == "mps" and tensor.device.index in (None, 0)
        )
    return tensor.device == device


class KernelInvocation:
    """One-time native dispatch instrumentation, removed after synchronized success."""

    def __init__(self, backend: str, device):
        self.backend = backend
        self.device = device
        self.provenance = installed_provenance()
        self.module = importlib.import_module(KERNEL_MODULE)
        self.kernels = self.provenance.verify_extension(self.module, backend)
        self.executed = None
        self.originals = {}
        self.wrappers = {}

    def install(self) -> None:
        for name in sorted(KERNEL_ENTRYPOINTS):
            original = getattr(self.module, name, None)
            if original is None:
                continue
            if (
                not isinstance(original, BuiltinFunctionType)
                or getattr(original, "__module__", None) != KERNEL_MODULE
                or getattr(original, "__name__", None) != name
            ):
                raise ProvenanceError("kernel entrypoint is not the compiled module callable")
            self.originals[name] = original
        if not self.originals:
            raise ProvenanceError("compiled extension has no reviewed inference entrypoint")
        for name, original in self.originals.items():
            wrapper = self._wrap(name, original)
            self.wrappers[name] = wrapper
            setattr(self.module, name, wrapper)

    def _wrap(self, name, original):
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if self.executed is not None:
                return result
            # Empty geometry and CPU-only arguments cannot establish execution
            # on this worker's accelerator. No tensor data is copied or read.
            tensors = (
                value for value in itertools.chain(args, kwargs.values())
                if isinstance(value, torch.Tensor)
            )
            if not any(_on_device(value, self.device, self.backend) and value.numel() > 0 for value in tensors):
                return result
            if self.backend == "cuda":
                # Synchronization during capture is illegal. A captured dispatch
                # alone is not execution; wait for a noncaptured native call.
                if torch.cuda.is_current_stream_capturing():
                    return result
                torch.cuda.synchronize(self.device)
            else:
                torch.mps.synchronize()
            self.provenance.verify_extension(self.module, self.backend)
            self.executed = name
            self.restore()
            return result

        return observed

    def restore(self) -> None:
        for name, wrapper in self.wrappers.items():
            if getattr(self.module, name, None) is wrapper:
                setattr(self.module, name, self.originals[name])

    def evidence(self) -> dict:
        if self.executed is None:
            raise ProvenanceError("no successfully executed native inference kernel")
        if getattr(self.module, self.executed, None) is not self.originals[self.executed]:
            raise ProvenanceError("executed native entrypoint was replaced")
        kernels = self.provenance.verify_extension(self.module, self.backend)
        return {
            "kernels": kernels,
            "kernel_entrypoint": self.executed,
            "kernel_module": Path(self.module.__file__).relative_to(self.provenance.root).as_posix(),
            "kernel_artifacts": dict(self.provenance.artifacts),
        }


@lru_cache(maxsize=1)
def metal_platform_uuid() -> str:
    if sys.platform != "darwin":
        raise ProvenanceError("Metal identity requires a native Darwin host")
    result = subprocess.run(
        ["/usr/sbin/ioreg", "-rd1", "-c", "IOPlatformExpertDevice", "-a"],
        check=True,
        capture_output=True,
        timeout=5,
    )
    if len(result.stdout) > 1024 * 1024:
        raise ProvenanceError("oversized native host identity")
    records = plistlib.loads(result.stdout)
    if not isinstance(records, list) or len(records) != 1:
        raise ProvenanceError("ambiguous native host identity")
    identity = records[0].get("IOPlatformUUID")
    if not isinstance(identity, str):
        raise ProvenanceError("native host has no stable platform UUID")
    value = uuid.UUID(identity)
    if value.int == 0:
        raise ProvenanceError("native host UUID is empty")
    return "apple-platform-integrated-gpu-v1:" + hashlib.sha256(str(value).encode()).hexdigest()


class _ObservedWorker:
    _sovereign_backend: str

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        from lazarus.appliance.slimserve_observation import engine_config_facts

        if load_dummy_weights:
            raise ProvenanceError("dummy weights cannot produce serving evidence")
        self._sovereign_loaded_model = None
        previous = getattr(self, "_sovereign_kernel", None)
        if previous is not None:
            previous.restore()
        provenance = installed_provenance()
        base = UpstreamMetalWorker if self._sovereign_backend == "metal" else UpstreamCudaWorker
        provenance.verify_api(base.load_model)
        provenance.verify_api(base.init_device)
        provenance.verify_api(self.model_runner.load_model)
        provenance.verify_api(self.model_runner.get_model)
        loaded_config = engine_config_facts(self.vllm_config)
        observation = KernelInvocation(self._sovereign_backend, self.device)
        observation.install()
        self._sovereign_kernel = observation
        try:
            super().load_model(load_dummy_weights=False)
        except BaseException:
            observation.restore()
            raise
        model = self.model_runner.get_model()
        if model is None:
            observation.restore()
            raise ProvenanceError("worker did not load a model")
        provenance.verify_api(type(model))
        provenance.verify_api(type(self.model_runner.model))
        if engine_config_facts(self.vllm_config) != loaded_config:
            raise ProvenanceError("model configuration changed during loading")
        self._sovereign_loaded_config = loaded_config
        self._sovereign_loaded_model = model

    def sovereign_observation(self) -> dict:
        """Read actual worker state through AsyncLLM.collective_rpc; never RPC arguments."""
        try:
            return self._sovereign_observation()
        except Exception:
            # Keep paths, checkpoint details and exception text off the private
            # transport too. An incomplete rank makes the aggregate nonready.
            return {"ready": False}

    def _sovereign_observation(self) -> dict:
        from lazarus.appliance.slimserve_observation import engine_config_facts

        model = self.model_runner.get_model()
        if model is None or model is not getattr(self, "_sovereign_loaded_model", None):
            raise ProvenanceError("worker model is not the observed loaded model")
        provenance = installed_provenance()
        provenance.verify_api(type(model))
        provenance.verify_api(type(self.model_runner))
        provenance.verify_api(self.model_runner.get_model)
        provenance.verify_api(type(self.model_runner.model))
        if engine_config_facts(self.vllm_config) != self._sovereign_loaded_config:
            raise ProvenanceError("loaded model configuration has changed")
        provenance.verify_loaded_modules()
        if self.model_config is not self.vllm_config.model_config:
            raise ProvenanceError("worker and loader configuration differ")
        tensors = itertools.chain(model.parameters(), model.buffers())
        if not any(_on_device(value, self.device, self._sovereign_backend) and value.numel() > 0 for value in tensors):
            raise ProvenanceError("loaded model has no accelerator tensors")
        group = get_tp_group()
        if (
            torch.distributed.get_rank() != self.rank
            or torch.distributed.get_world_size() != self.parallel_config.world_size
            or group.world_size != self.parallel_config.tensor_parallel_size
            or group.rank_in_group != self.local_rank
        ):
            raise ProvenanceError("worker distributed ranks disagree")
        if self._sovereign_backend == "cuda":
            if self.device.type != "cuda" or torch.cuda.current_device() != self.device.index:
                raise ProvenanceError("worker is not bound to its CUDA device")
            identity = _canonical_torch_uuid(torch.cuda.get_device_properties(self.device).uuid)
            if identity is None:
                raise ProvenanceError("CUDA worker has no stable physical identity")
            device = {
                "identity_kind": "nvidia_gpu_uuid",
                "stable_identifier": identity,
                "gpu_uuid": identity,
                "local_rank": self.local_rank,
            }
            visible_count = torch.cuda.device_count()
        else:
            if self.device.type != "mps" or not torch.backends.mps.is_available():
                raise ProvenanceError("worker has no native Metal device")
            identity = metal_platform_uuid()
            device = {
                "identity_kind": "apple_platform",
                "stable_identifier": identity,
                "platform_id": identity,
            }
            visible_count = 1
        return {
            "ready": True,
            "engine": provenance.engine,
            **self._sovereign_kernel.evidence(),
            "config": engine_config_facts(self.vllm_config),
            "rank": self.rank,
            "local_rank": self.local_rank,
            "tensor_parallel_rank": group.rank_in_group,
            "world_size": torch.distributed.get_world_size(),
            "visible_device_count": visible_count,
            "device": device,
        }


class CudaWorker(_ObservedWorker, UpstreamCudaWorker):
    _sovereign_backend = "cuda"


class MetalWorker(_ObservedWorker, UpstreamMetalWorker):
    _sovereign_backend = "metal"
