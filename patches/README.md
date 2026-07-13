# Overlay patches

Patches Lazarus carries against pinned upstream dependencies until upstreamed.
Applied at image/host build time (never edit installed packages in place on a
running appliance).

## 0001-gate-cpu-moe-bindings-on-apple.patch

Target: upstream **vLLM** (CPU build). Upstream excludes `cpu_fused_moe.cpp`
from the Apple-Silicon CPU build while `torch_bindings.cpp` binds it
unconditionally, so `vllm._C` fails to `dlopen` with an undefined symbol.
Gates the binding on `!defined(__APPLE__)`.

## 0002-vllm-metal-quantize-on-load.patch

Target: **vllm-metal** (`vllm_metal/v1/model_lifecycle.py`). Adds optional
in-place 4-bit quantization of an already-built mlx-lm model, enabled by
`VLLM_METAL_QUANTIZE_BITS` (group size via `VLLM_METAL_QUANTIZE_GROUP_SIZE`,
default 32).

**Why:** gemma-3n / gemma-4 MatFormer checkpoints use KV-sharing (later layers
share k/v projections), so the checkpoint carries params the fresh model
skeleton lacks. MLX's stock converter (`mlx_lm convert -q`) and `mlx_lm.load`
on a bare HF path both reject them (`N parameters not in model`), and
vllm-metal's GGUF loader rejects the `gemma4` architecture (dense-decoder
scope). vllm-metal's compatible-path loader is the *only* thing that builds
this model correctly; this patch quantizes its linears in place after that
load — sidestepping both the GGUF adapter and the broken converter.

**Result (Apple Silicon, gemma-4-E2B):** bf16 → 4-bit lifts warm decode to
~120 tok/s (from ~27 bf16), matching/beating the llama.cpp Q4_0 reference,
through the same `mx.quantized_matmul` path — pure vLLM, no llama.cpp.

Apply (into the vllm-metal venv's site-packages root):

```bash
python - <<'PY'
import vllm_metal, pathlib, subprocess
root = pathlib.Path(vllm_metal.__file__).parent.parent
subprocess.run(["patch","-p1","-d",str(root),"-i",
                str(pathlib.Path("patches/0002-vllm-metal-quantize-on-load.patch").resolve())], check=True)
PY
```

Enable at runtime: `VLLM_METAL_QUANTIZE_BITS=4` in the metal runtime's
environment.
