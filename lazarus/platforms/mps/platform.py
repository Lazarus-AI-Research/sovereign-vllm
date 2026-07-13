"""MpsPlatform — vLLM out-of-tree platform for the Apple GPU via PyTorch-MPS.

Design: keep vLLM's native model executor and V1 GPU worker/runner; run the
math on ``mps`` tensors. CustomOp.forward_native (pure torch) covers
norm/rope/activation; attention is dispatched to a QuixiCore Metal backend.
CUDA-graph capture is disabled (no MPS analog), which gates out most of the
CUDA-specific paths in the GPU runner. Remaining torch.cuda.* sites are handled
by MpsWorker/MpsModelRunner overrides as the boot probe surfaces them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class MpsPlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "mps"
    device_type: str = "mps"
    # Real torch dispatch key for the Apple GPU (unlike vllm-metal, which lies
    # "CPU"). Tensors live on and dispatch to MPS.
    dispatch_key: str = "MPS"
    # No NCCL analog; gloo is the only collective backend that works here.
    dist_backend: str = "gloo"
    # Metal attention backend (built in M-N1). Placeholder qualname resolved
    # lazily by get_attn_backend_cls.
    simple_compile_backend: str = "eager"

    @classmethod
    def is_available(cls) -> bool:
        return bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )

    @property
    def supported_dtypes(self) -> list[torch.dtype]:
        # MPS supports fp16 broadly and bf16 for most ops in recent torch.
        # Prefer fp16 first so "auto" dtype resolution lands on the best-tested
        # path (QuixiCore qgemm is fp16-activation).
        return [torch.float16, torch.bfloat16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "mps"

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability:
        # vLLM has CUDA-centric capability gates scattered through model and
        # quant selection. Present a modern-CUDA-equivalent capability so those
        # gates pass, matching vllm-metal's approach.
        return DeviceCapability(major=8, minor=0)

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Unified memory: report the Metal driver's recommended working-set max.
        try:
            return int(torch.mps.recommended_max_memory())
        except Exception:
            import psutil

            return int(psutil.virtual_memory().total)

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        total = cls.get_device_total_memory()
        try:
            used = int(torch.mps.driver_allocated_memory())
        except Exception:
            used = 0
        return max(total - used, 0), total

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        try:
            return float(torch.mps.current_allocated_memory())
        except Exception:
            return 0.0

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        # MPS is a single implicit device; nothing to select.
        pass

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            torch.manual_seed(seed)

    @classmethod
    def inference_mode(cls):
        # inference_mode() interacts badly with some non-CUDA backends; CPU/TPU
        # use no_grad() and MPS follows suit.
        return torch.no_grad()

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(f"MPS platform does not support dtype {dtype}.")

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return False

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        if getattr(attn_selector_config, "use_mla", False):
            # QuixiCore has mla_decode; wire in M-N4 when we tackle DeepSeek/omni.
            raise NotImplementedError("MLA attention not yet wired on MPS.")
        return "lazarus.platforms.mps.attention.MetalAttentionBackend"

    @classmethod
    def get_punica_wrapper(cls) -> str:
        # LoRA punica kernels are CUDA-only; not supported yet.
        raise NotImplementedError("LoRA (punica) is not supported on MPS.")

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        # Single Apple GPU: no real collectives. Reuse the base/CPU communicator.
        return "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        from vllm.config import CompilationMode

        # Safe point (vLLM fully imported) to install the Triton→torch compat
        # shims and disable dynamo/inductor (no MPS/CPU compile backend here).
        from lazarus.platforms.mps.compat import apply_compat_patches

        apply_compat_patches()

        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "lazarus.platforms.mps.worker.MpsWorker"
        parallel_config.disable_custom_all_reduce = True
        if getattr(parallel_config, "enable_dbo", False):
            parallel_config.enable_dbo = False

        scheduler_config = vllm_config.scheduler_config
        scheduler_config.async_scheduling = False

        # No CUDA graphs on MPS: force eager. This gates out the bulk of the
        # torch.cuda.graph/Stream capture paths in the GPU model runner.
        compilation_config = vllm_config.compilation_config
        compilation_config.cudagraph_capture_sizes = []
        try:
            from vllm.config import CUDAGraphMode

            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        except Exception:
            pass
        compilation_config.mode = CompilationMode.NONE

        model_config = vllm_config.model_config
        if model_config is not None:
            # QuixiCore cascade attention exists but is unwired; keep off for now.
            model_config.disable_cascade_attn = True

        cache_config = vllm_config.cache_config
        if not getattr(cache_config, "user_specified_block_size", False):
            # QuixiCore paged_attention block layout; 16 is the vLLM default and
            # a MultipleOf(16) that the Metal kernels accept.
            cache_config.block_size = 16
