"""Compatibility monkeypatches for running vLLM's V1 GPU path on PyTorch-MPS.

vLLM ships a handful of hot helpers as Triton kernels. Triton doesn't target
Metal and isn't installed on macOS, so those kernels degrade to plain Python
functions and fail when launched with the ``kernel[grid](...)`` syntax. We
replace them with equivalent pure-torch implementations that run on ``mps``.

Applied once, at platform registration.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_APPLIED = False


def _mps_compute_slot_mapping(self, num_reqs, query_start_loc, positions) -> None:
    """Torch replacement for the Triton slot-mapping kernel (BlockTable).

    slot = block_table[req, pos // block_size] * block_size + pos % block_size.
    Context/decode parallelism (pcp/dcp) is single-rank on one Apple GPU.
    """
    total_cp = self.pcp_world_size * self.dcp_world_size
    if total_cp != 1:
        raise NotImplementedError(
            "context/decode parallelism is not supported on MPS"
        )

    sm = self.slot_mapping.gpu
    device = sm.device
    bs = self.block_size
    num_tokens = positions.shape[0]

    # Per-token request index from the query_start_loc offsets.
    qsl = query_start_loc[: num_reqs + 1].to("cpu", dtype=torch.long)
    counts = (qsl[1:] - qsl[:-1]).clamp_min(0)
    req_idx = torch.repeat_interleave(
        torch.arange(num_reqs, dtype=torch.long), counts
    ).to(device)

    pos = positions[:num_tokens].to(device=device, dtype=torch.long)
    n = req_idx.shape[0]
    pos = pos[:n]

    block_ids = self.block_table.gpu[req_idx, pos // bs].to(torch.long)
    slots = block_ids * bs + (pos % bs)

    sm[:n] = slots.to(sm.dtype)
    if sm.shape[0] > n:
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        sm[n:] = PAD_SLOT_ID


def apply_compat_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    # There is no usable torch.compile backend on this host: inductor's CPU
    # path needs a C++ toolchain that isn't present, and there is no MPS
    # inductor backend. Force every @torch.compile in vLLM to run eager.
    try:
        import torch._dynamo

        torch._dynamo.config.disable = True
    except Exception:
        logger.warning("Could not disable torch._dynamo; compile paths may fail")

    from vllm.v1.worker.block_table import BlockTable

    BlockTable.compute_slot_mapping = _mps_compute_slot_mapping

    _patch_cpu_gpu_buffer_blocking()
    _preload_quixicore()

    _APPLIED = True
    logger.info(
        "Applied MPS compat patches (dynamo-disable, slot-mapping, blocking-copies)"
    )


def _preload_quixicore() -> None:
    """Import QuixiCore's tk_torch eagerly at startup.

    tk_torch's first import runs build_metallib() + cpp_extension.load() (a file
    lock + JIT link). Doing that lazily inside the attention forward — mid-decode,
    against in-flight MPS work — can deadlock. Warming it once here, before any
    generation, makes the fast attention path reliable. Absence is non-fatal:
    the backend falls back to the torch SDPA path.
    """
    import os

    if os.environ.get("SOVEREIGN_MPS_NO_QUIXI") == "1":
        return
    try:
        import tk_torch  # noqa: F401

        logger.info("Preloaded QuixiCore tk_torch (Metal attention fast path)")
    except Exception as e:
        logger.warning("tk_torch unavailable (%s); using SDPA attention", e)


def _patch_cpu_gpu_buffer_blocking() -> None:
    """Force CpuGpuBuffer H2D/D2H copies to be blocking on MPS.

    vLLM issues ``copy_(..., non_blocking=True)`` for its input-prep buffers.
    On CUDA the following same-stream kernels wait for the copy; on MPS a
    non-blocking copy from non-pinned host memory is NOT ordered against the
    dependent MPS graph, so a gather (e.g. positions[req_indices]) can read a
    stale buffer and index out of bounds. Making these copies synchronous
    removes the hazard. (Pin memory is unavailable on MPS anyway.)
    """
    from vllm.v1.utils import CpuGpuBuffer

    def copy_to_gpu(self, n=None):
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=False)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=False)

    def copy_to_cpu(self, n=None):
        if n is None:
            return self.cpu.copy_(self.gpu, non_blocking=False)
        return self.cpu[:n].copy_(self.gpu[:n], non_blocking=False)

    CpuGpuBuffer.copy_to_gpu = copy_to_gpu
    CpuGpuBuffer.copy_to_cpu = copy_to_cpu
