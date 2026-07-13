# Native GGUF-on-Metal backend (vLLM torch-MPS platform)

Status: **in progress** (M-N0). Owner: appliance/runtime track.

## Goal

Run vLLM's **native** torch models on the Apple GPU via PyTorch-MPS, with
QuixiCore-Metal supplying the kernels torch-MPS lacks (attention, quantized
matmul) and `vllm-gguf-plugin` supplying GGUF load + the quantized linear
method. This replaces the MLX detour (`vllm-metal`), whose model coverage is
gated by mlx-lm/mlx-vlm and therefore cannot load our two target checkpoints:

- **google/gemma-4-E2B-it** — gemma-3n MatFormer (KV-sharing, per-layer
  embeddings). Breaks mlx-lm convert/load and vllm-metal's GGUF adapter.
- **LCO-Embedding-Omni-3B** — Qwen2.5-Omni *thinker* (dense). mlx-vlm only has
  the MoE `qwen3_omni_moe`, a structural mismatch.

Both have **native vLLM implementations** (`vllm/model_executor/models/gemma3n.py`,
`qwen2_5_omni_thinker.py`), so running them on Metal needs **no model porting** —
only a platform + an attention backend.

## Why this is tractable (three seam maps)

### 1. The platform — vllm-metal is the registration template, not the exec model

vLLM selects a platform via the `vllm.platform_plugins` entry point; a
`register()` returning a qualname activates an out-of-tree platform that wins
over all builtins (`vllm/platforms/__init__.py`). vllm-metal uses this hook but
then swaps `worker_cls` for a **bespoke MLX runner** and never calls
`vllm.model_executor` — it masquerades as `device_type="cpu"`.

Ours does the opposite: `device_type="mps"`, `dispatch_key="MPS"`,
`dist_backend="gloo"`, and **keeps vLLM's native model executor**. The
`CustomOp` dispatch shim (`vllm/model_executor/custom_op.py`) routes every
out-of-tree platform to `forward_native` by default — so `RMSNorm`, `RotaryEmbedding`,
`SiluAndMul`, etc. run as **plain torch on MPS with zero kernels written**.
`@CustomOp.register_oot` later swaps in Metal versions for speed.

The one op family with **no** `forward_native` fallback is **attention**
(`paged_attention_v*`, `reshape_and_cache` are C-ops). That is the single
required Metal component.

The real cost is de-CUDA-fying the runner control plane
(`vllm/v1/worker/gpu_model_runner.py`, ~225 CUDA sites: `torch.cuda.Stream/Event`,
CUDA graphs, `mem_get_info`). Most are gated behind CUDA-graph capture, which
`check_and_update_config` disables (`CUDAGraphMode.NONE`, empty capture sizes) —
the CPU platform is the reference for exactly this.

### 2. Attention is already built — QuixiCore is a paged runtime, not a toy

`tk_torch` (QuixiCore's PyTorch-MPS binding) is `torch.Tensor`-in/out on `mps`
and already ships the vLLM serving surface:

- `paged_attention(q, key_cache, value_cache, block_table, context_lens, ...)`
  — GQA/MQA-aware, caches `(num_blocks, block_size, H_KV, D)`, int32
  block_table/context_lens. Maps 1:1 onto vLLM's `AttentionImpl.forward`.
- `kv_cache_scatter(..., slot_mapping)` == `reshape_and_cache`.
- `paged_attention_v2` (long-context), `_fp8`, `_window`, `cascade_*`,
  `attn_varlen_prefill` (ragged prefill), plus spec-decode/beam/samplers.
- `qgemm(wq_uint8, x_fp16, "q4_0"|"q8_0")` / `qgemv` — fused GGUF matmul that
  eats **raw GGUF blocks** directly (the plugin's weight format at the seam).

So the Metal attention backend is a **binding job**, not a kernel-writing job.

### 3. GGUF is one function

`vllm-gguf-plugin`'s load path is device-agnostic (`loader.py` honors
`device_config.device`; raw GGUF uint8 blocks land on `mps` unchanged). There
is exactly one compute seam — `vllm_gguf_plugin/ops.py`. Minimum viable Metal
backend = an MPS branch in `ggml_dequantize` (dequant→`x @ w.T`). Fast path =
route `ggml_mul_mat_vec_a8`/`ggml_mul_mat_a8` to `tk_torch.qgemv`/`qgemm`.

## Architecture

```
AnythingLLM → LiteLLM → sovereign-runtime (appliance) → vLLM AsyncLLM
                                                          └─ MpsPlatform (OOT)
                                                             ├─ native model_executor models (torch, on mps)
                                                             │   └─ CustomOp.forward_native  (free)  → register_oot Metal (fast)
                                                             ├─ MetalAttentionBackend → tk_torch.paged_attention / kv_cache_scatter
                                                             └─ GGUFConfig (vllm-gguf-plugin) → ops.py MPS → tk_torch.qgemm/qgemv
```

Lives in the fork: `lazarus/platforms/mps/` (platform + attention backend +
QuixiCore glue). The plugin's MPS ops branch is an overlay patch against
`vllm-gguf-plugin` (vendored under `third_party/` or carried as a patch).

## Known gaps to grow (from the QuixiCore surface map)

| Gap | Blocks | Plan |
|---|---|---|
| mrope / 3-D multimodal RoPE absent | Qwen2.5-Omni/VL | text-only embedding uses degenerate 1-D rope (all sections equal); add mrope kernel for full multimodal |
| attention head_dim only {64,128} | gemma-3n (256) | grow paged/flash attn head-dim coverage to 256 |
| dense GQA prefill limited (paged varlen exists) | simple runner | use paged path, or `repeat_interleave` KV in torch |
| bf16 (norms/attn/rope) vs fp16 (qgemm) dtype split | mixed | standardize on one; cast around qgemm |
| fused residual+norm width ∈ {256,512,768,1024} | wide models | dynamic `rms_norm` + separate `add_rt` fallback |
| no torch-side GGUF weight packer | — | plugin loads pre-packed blocks; no packing needed |
| tk_torch not co-installed with vllm | env | pip install -e into the vllm venv (JIT rebuilds for its torch) |

## Milestones

- **M-N0** — MpsPlatform skeleton; boot a tiny GQA model; record the first CUDA wall.
- **M-N1** — Metal attention backend (QuixiCore); tiny model end-to-end, coherent tokens.
- **M-N2** — GGUF on MPS (plugin `ggml_dequantize` + `qgemm` fast path).
- **M-N3** — perf pass (`register_oot` Metal norm/rope/act; fused matmul); benchmark vs 122 tok/s MLX reference.
- **M-N4** — target models: gemma-4 (head_dim 256) + LCO-Omni embeddings (mrope).
