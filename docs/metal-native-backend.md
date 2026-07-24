# Native GGUF-on-Metal backend (vLLM torch-MPS platform)

Status: **in progress** (M-N0). Owner: appliance/runtime track.

## Goal

Run vLLM's **native** torch models on the Apple GPU via PyTorch-MPS, with
QuixiCore-Metal supplying the kernels torch-MPS lacks (attention, quantized
matmul) and `vllm-gguf-plugin` supplying GGUF load + the quantized linear
method. This replaces the MLX detour (`vllm-metal`), whose model coverage is
gated by mlx-lm/mlx-vlm and therefore cannot load the certified generation checkpoint:

- **google/gemma-4-E2B-it** — gemma-3n MatFormer (KV-sharing, per-layer
  embeddings). Breaks mlx-lm convert/load and vllm-metal's GGUF adapter.
It has a **native vLLM implementation**
(`vllm/model_executor/models/gemma3n.py`), so running it on Metal needs **no
model porting** — only a platform + an attention backend.

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
| mrope / 3-D multimodal RoPE absent | future multimodal models | add an mrope kernel when a certified model requires it |
| attention head_dim only {64,128} | gemma-3n (256) | grow paged/flash attn head-dim coverage to 256 |
| dense GQA prefill limited (paged varlen exists) | simple runner | use paged path, or `repeat_interleave` KV in torch |
| bf16 (norms/attn/rope) vs fp16 (qgemm) dtype split | mixed | standardize on one; cast around qgemm |
| fused residual+norm width ∈ {256,512,768,1024} | wide models | dynamic `rms_norm` + separate `add_rt` fallback |
| no torch-side GGUF weight packer | — | plugin loads pre-packed blocks; no packing needed |
| tk_torch not co-installed with vllm | env | pip install -e into the vllm venv (JIT rebuilds for its torch) |

## Milestones

- **M-N0 — DONE.** MpsPlatform skeleton; native model construction boots on MPS.
  De-CUDA-fication turned out to be 4 small overrides (init_device, dtype check,
  compute-units, memory-pool context) because `check_and_update_config` disables
  CUDA graphs, gating out most of the runner's CUDA sites.
- **M-N1 — DONE.** vLLM's native LlamaForCausalLM (SmolLM2-135M) generates
  coherent tokens end-to-end on the Apple GPU. MetalAttentionBackend (flash-style
  KV layout, in-place cache write + per-request SDPA read on MPS), registered in
  vLLM's CUSTOM attention slot. Key fixes in `compat.py`: dynamo-disable (no
  compile backend on-host), Triton slot-mapping kernel → torch, and forcing
  `CpuGpuBuffer` H2D/D2H copies **blocking** on MPS (non-blocking copies from
  non-pinned host memory race the dependent gather → stale-index OOB). Perf is a
  correctness baseline (~0.8 tok/s); QuixiCore + eager-kill land in M-N3.
- **M-N2 — DONE (correctness path).** A Q8_0 GGUF (SmolLM2-135M-Instruct) loads
  via vllm-gguf-plugin + MpsPlatform and answers "Paris." correctly — GGUF
  running natively on the Apple GPU in vLLM. Plugin changes (local commits in
  `~/vllm-gguf-plugin`, not pushed to upstream): `setup.py` skips the
  CUDAExtension build when there's no CUDA/ROCm toolchain; `ops.py` sources GGML
  type ids from the `gguf` package + lazy-imports triton, and adds an MPS branch
  to `ggml_dequantize`/`ggml_mul_mat_vec_a8`/`ggml_mul_mat_a8` (dequant GGUF
  blocks via gguf's pure-Python unpacker → torch-MPS matmul; all quant types).
  A local `.gguf` needs an HF config/tokenizer source (`tokenizer=`/
  `hf_config_path=` a same-arch HF model) — standard vLLM GGUF requirement, to be
  wired into the appliance config. Fused `tk_torch.qgemm/qgemv` fast path → M-N3.
- **M-N3 — DONE (attention).** QuixiCore `paged_attention` replaces the
  per-request SDPA loop for the decode batch (default on; `SOVEREIGN_MPS_NO_QUIXI=1`
  forces SDPA). SmolLM2-135M: **60.9 tok/s** batch-of-4 / **37.8 tok/s** single
  (was 0.8–1.1). Two root-causes fixed along the way: (1) the KV layout is now
  `(2, num_blocks, ...)` so `kv_cache[0]`/`[1]` are contiguous (no per-step copy
  for QuixiCore); (2) the cache write uses per-token advanced indexing
  (`key_cache[blk, off] = key`) — the earlier `index_copy_` on the flattened
  multi-million-row view was O(cache) on MPS and stalled. tk_torch is preloaded
  at startup (compat) so its JIT/metallib build never happens mid-decode.
  Remaining headroom toward the ~71 tok/s raw-torch ceiling: `register_oot`
  Metal RMSNorm/RoPE/SiLU and `tk_torch.qgemm/qgemv` for the GGUF matmul.
- **M-N4** — target model: gemma-4 (head_dim 256); certify optional embedding models independently through Control.
