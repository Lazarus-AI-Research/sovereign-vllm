"""MpsModelRunner — vLLM V1 GPU model runner adapted for PyTorch-MPS.

Overrides only the handful of methods the base class documents as
device-specific override points (``_init_device_properties``, ``_sync_device``)
plus any CUDA-stream paths that surface on MPS. The heavy lifting — input
batching, sampling, KV-cache management, native-model forward — is inherited.
"""

from __future__ import annotations

import subprocess

import torch

from vllm.logger import init_logger
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _detect_gpu_cores(default: int = 32) -> int:
    """Apple GPU core count (a stand-in for CUDA SM count in vLLM heuristics)."""
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPDisplaysDataType"], text=True, timeout=5
        )
        for line in out.splitlines():
            if "Total Number of Cores" in line:
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return default


class MpsModelRunner(GPUModelRunner):
    def _init_device_properties(self) -> None:
        self.num_sms = _detect_gpu_cores()

    def _sync_device(self) -> None:
        try:
            torch.mps.synchronize()
        except Exception:
            pass
