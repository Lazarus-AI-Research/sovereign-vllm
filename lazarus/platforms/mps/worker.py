"""MpsWorker — vLLM V1 GPU worker adapted for the Apple GPU (PyTorch-MPS).

Subclasses the stock ``Worker`` and overrides only the device-setup and
memory-profiling paths, which are hard-wired to CUDA. Everything else (weight
load, execute_model, KV-cache plumbing) is inherited unchanged and runs on
``mps`` tensors.
"""

from __future__ import annotations

import gc
from contextlib import AbstractContextManager, nullcontext

import torch

from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.workspace import init_workspace_manager

logger = init_logger(__name__)


class MpsWorker(Worker):
    def init_device(self):
        assert self.device_config.device_type == "mps"

        self.device = torch.device("mps")
        current_platform.check_if_supports_dtype(self.model_config.dtype)

        # Single-process gloo group (no NCCL on Apple). world_size is 1 for the
        # single-GPU appliance; TP/PP are meaningless on one unified device.
        init_distributed_environment(
            self.parallel_config.world_size,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
            self.parallel_config.prefill_context_parallel_size,
            self.parallel_config.decode_context_parallel_size,
        )

        set_random_seed(self.model_config.seed)

        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

        # Skip the CUDA MemorySnapshot; memory budgeting is handled by
        # determine_available_memory() from unified-memory totals.
        self.init_snapshot = None
        self.requested_memory = None

        num_ubatches = 2 if self.vllm_config.parallel_config.enable_dbo else 1
        init_workspace_manager(self.device, num_ubatches)

        from lazarus.platforms.mps.model_runner import MpsModelRunner

        self.model_runner = MpsModelRunner(self.vllm_config, self.device)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        # No CuMem sleep-mode allocator on MPS; weights load straight to the
        # unified-memory pool.
        return nullcontext()

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """KV-cache budget from unified memory.

        Apple's unified memory has no separate device pool to profile with
        ``torch.cuda.mem_get_info``. Run a profile pass for graph/shape warmup,
        then budget = gpu_memory_utilization * recommended_max - current_usage.
        """
        self.model_runner.profile_run()

        total = current_platform.get_device_total_memory()
        util = self.cache_config.gpu_memory_utilization
        try:
            in_use = int(torch.mps.current_allocated_memory())
        except Exception:
            in_use = 0
        available = int(total * util) - in_use
        if available <= 0:
            raise RuntimeError(
                f"No memory left for KV cache: total={total} util={util} "
                f"in_use={in_use}"
            )
        logger.info(
            "MPS KV-cache budget: %.2f GiB (total %.2f GiB * util %.2f - in-use %.2f GiB)",
            available / 2**30, total / 2**30, util, in_use / 2**30,
        )
        return available
