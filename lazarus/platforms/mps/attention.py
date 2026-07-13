"""MetalAttentionBackend — vLLM V1 attention on the Apple GPU.

KV-cache layout ``(2, num_blocks, block_size, num_kv_heads, head_size)`` so that
``kv_cache[0]`` / ``kv_cache[1]`` are exactly the ``(num_blocks, block_size,
H_kv, D)`` tensors QuixiCore's ``paged_attention`` consumes.

M-N1 uses a correctness-first path: an in-place torch scatter writes new K/V
into the persistent paged cache, and attention is computed per request with
``F.scaled_dot_product_attention`` (runs natively on MPS). M-N3 replaces the
decode read with ``tk_torch.paged_attention`` and prefill with
``tk_torch.attn_varlen_prefill`` for speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

_TK = None
# QuixiCore fused attention is on by default; set SOVEREIGN_MPS_NO_QUIXI=1 to
# force the pure-torch SDPA path (debugging / unsupported kernels).
_QUIXI_ENABLED = os.environ.get("SOVEREIGN_MPS_NO_QUIXI") != "1"


def _tk_torch():
    """Return QuixiCore's PyTorch-MPS kernels (preloaded at startup by compat)."""
    global _TK
    if _TK is None:
        import tk_torch

        _TK = tk_torch
    return _TK


def _quixi_paged_attention_supported(head_size: int) -> bool:
    # QuixiCore paged_attention supports head dims {64, 128}.
    return head_size in (64, 128)


class MetalAttentionBackend(AttentionBackend):
    # forward() writes K/V into the paged cache itself, so vLLM must not call
    # the separate unified_kv_cache_update op.
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]

    @staticmethod
    def get_name() -> str:
        # OOT backends use the CUSTOM enum slot (registered below).
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["MetalAttentionImpl"]:
        return MetalAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["MetalAttentionMetadataBuilder"]:
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Dim 0 selects key(0)/value(1); kv_cache[0] / kv_cache[1] are then
        # CONTIGUOUS (num_blocks, block_size, H_kv, D) tensors — exactly the
        # layout QuixiCore's paged_attention consumes with no per-step copy.
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class MetalAttentionMetadata:
    num_actual_tokens: int
    num_reqs: int
    max_query_len: int
    # CPU int tensors used to slice the batch per request (SDPA prefill path).
    query_start_loc_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    # On-device tensors for the cache write / QuixiCore paged decode.
    block_table: torch.Tensor  # int32 (num_reqs, max_blocks)
    seq_lens_gpu: torch.Tensor  # int32 (num_reqs,)
    slot_mapping: torch.Tensor
    causal: bool = True


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder[MetalAttentionMetadata]):
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = vllm_config.cache_config.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        m = common_attn_metadata
        seq_lens_cpu = m.seq_lens.to("cpu", dtype=torch.int32)
        qsl_cpu = m.query_start_loc_cpu.to(dtype=torch.int32)
        causal = m.causal if isinstance(m.causal, bool) else True
        return MetalAttentionMetadata(
            num_actual_tokens=m.num_actual_tokens,
            num_reqs=m.num_reqs,
            max_query_len=m.max_query_len,
            query_start_loc_cpu=qsl_cpu,
            seq_lens_cpu=seq_lens_cpu,
            block_table=m.block_table_tensor.to(torch.int32),
            seq_lens_gpu=m.seq_lens.to(torch.int32),
            slot_mapping=m.slot_mapping,
            causal=causal,
        )


class MetalAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.sliding_window = -1 if sliding_window is None else sliding_window
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi not yet supported on MPS attention.")
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                f"MPS attention supports decoder only, got {attn_type}."
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Warmup / profile run passes no metadata.
        if attn_metadata is None:
            return output
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Fused output quant unsupported on MPS.")

        num_tokens = attn_metadata.num_actual_tokens
        H, Hkv, D = self.num_heads, self.num_kv_heads, self.head_size

        # kv_cache: (2, num_blocks, block_size, Hkv, D) — kv_cache[0]/[1] contiguous
        _, num_blocks, block_size, _, _ = kv_cache.shape
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # --- write new K/V into the persistent paged cache (in-place, on MPS) ---
        # Per-token advanced-index scatter (O(num_tokens)). index_copy_ on the
        # flattened multi-million-row view is O(cache) on MPS and stalls.
        if self.kv_sharing_target_layer_name is None:
            slot = attn_metadata.slot_mapping[:num_tokens].to(torch.long)
            blk = slot // block_size
            off = slot % block_size
            key_cache[blk, off] = key[:num_tokens]
            value_cache[blk, off] = value[:num_tokens]

        out2d = output.view(num_tokens, H, D)

        # --- decode fast path: one fused QuixiCore kernel for the whole batch ---
        # Pure decode = every request contributes exactly one query token. ~34x
        # faster than the per-request SDPA loop (37 vs 1.1 tok/s on SmolLM2-135M).
        if (
            _QUIXI_ENABLED
            and attn_metadata.max_query_len == 1
            and num_tokens == attn_metadata.num_reqs
            and _quixi_paged_attention_supported(D)
        ):
            tk = _tk_torch()
            q = query[:num_tokens].contiguous()  # (num_reqs, H, D)
            o = tk.paged_attention(
                q,
                key_cache,
                value_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens_gpu,
                float(self.scale),
                0,
            )  # (num_reqs, H, D)
            out2d.copy_(o)
            return output

        # --- prefill / mixed path: per-request SDPA over the gathered cache ---
        qsl = attn_metadata.query_start_loc_cpu
        seq_lens = attn_metadata.seq_lens_cpu
        block_table = attn_metadata.block_table

        for r in range(attn_metadata.num_reqs):
            q0 = int(qsl[r])
            q1 = int(qsl[r + 1])
            q_len = q1 - q0
            if q_len == 0:
                continue
            seq_len = int(seq_lens[r])

            # gather this request's keys/values from the paged cache. Clamp block
            # ids to the cache size: vLLM's profile/dummy run uses a small dummy
            # cache with a block_table that can reference blocks past it; that
            # run's output is discarded, and real runs never clamp.
            n_blocks = (seq_len + block_size - 1) // block_size
            blocks = block_table[r, :n_blocks].to(torch.long).clamp_(0, num_blocks - 1)
            seq_len = min(seq_len, n_blocks * block_size)
            k_r = key_cache.index_select(0, blocks).reshape(-1, Hkv, D)[:seq_len]
            v_r = value_cache.index_select(0, blocks).reshape(-1, Hkv, D)[:seq_len]

            # GQA expansion: (seq_len, Hkv, D) -> (seq_len, H, D)
            if self.num_queries_per_kv > 1:
                k_r = k_r.repeat_interleave(self.num_queries_per_kv, dim=1)
                v_r = v_r.repeat_interleave(self.num_queries_per_kv, dim=1)

            q_r = query[q0:q1]  # (q_len, H, D)
            q_t = q_r.transpose(0, 1)
            k_t = k_r.transpose(0, 1)
            v_t = v_r.transpose(0, 1)

            # causal mask: query token j (abs pos seq_len-q_len+j) attends keys<=pos
            if attn_metadata.causal and q_len > 1:
                q_pos = torch.arange(seq_len - q_len, seq_len, device=q_t.device)
                k_pos = torch.arange(seq_len, device=q_t.device)
                attn_mask = (k_pos[None, :] <= q_pos[:, None])[None]
            else:
                attn_mask = None

            o = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=attn_mask, scale=self.scale
            )  # (H, q_len, D)
            out2d[q0:q1] = o.transpose(0, 1)

        return output


# Register the Metal backend in vLLM's OOT CUSTOM slot.
try:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    register_backend(
        AttentionBackendEnum.CUSTOM,
        "lazarus.platforms.mps.attention.MetalAttentionBackend",
    )
except Exception:  # pragma: no cover - registry shape drift is non-fatal here
    logger.warning("Could not register MetalAttentionBackend in CUSTOM slot")
