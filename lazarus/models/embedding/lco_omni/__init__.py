"""LCO-Embedding-Omni pooling plugin (M12).

LCO-Embedding-Omni-3B is a *thinker-only* Qwen2.5-Omni checkpoint:
`architectures=["Qwen2_5OmniThinkerForConditionalGeneration"]`, the thinker
config at the top level of config.json, and weight names unprefixed
(`model.*`, `lm_head.*`, `visual.*`, `audio_tower.*`).

The pinned vLLM only registers the full-omni architectures
(`Qwen2_5OmniModel`, `Qwen2_5OmniForConditionalGeneration`) and its native
thinker model reads `hf_config.thinker_config`.  A bare thinker checkpoint
therefore misses the registry and falls back to the Transformers modelling
backend, whose multimodal path calls
`Processor._get_num_multimodal_tokens` — a method `Qwen2_5OmniProcessor`
does not implement in the pinned transformers.  Net effect: the M12
AttributeError at engine start, regardless of --limit-mm-per-prompt.

The fix stays fully out of tree:

- `normalize_thinker_config` is an `hf_overrides` callable (applied by the
  appliance backend to embedding roles) that wraps a bare thinker config
  into a real `Qwen2_5OmniConfig`, so every native vLLM code path (mrope
  detection, max_model_len, processing info) sees the shape it expects.
- `register` is a `vllm.general_plugins` entry point (declared in
  pyproject.toml, loaded by vLLM in every process including engine
  workers).  It maps the thinker architecture onto a subclass of vLLM's
  native thinker model whose weights mapper also accepts the unprefixed
  checkpoint names.
"""

from __future__ import annotations

THINKER_ARCH = "Qwen2_5OmniThinkerForConditionalGeneration"


def normalize_thinker_config(config):
    """hf_overrides hook: wrap a bare thinker config in Qwen2_5OmniConfig.

    No-op for anything that is not a top-level qwen2_5_omni_thinker config,
    so it is safe to apply to every embedding role.
    """
    if getattr(config, "model_type", None) != "qwen2_5_omni_thinker":
        return config
    if hasattr(config, "thinker_config"):
        return config

    from transformers import Qwen2_5OmniConfig

    wrapped = Qwen2_5OmniConfig(
        thinker_config=config.to_dict(),
        enable_audio_output=False,
        architectures=[THINKER_ARCH],
    )
    # The engine resolves dtype=auto from the top-level config; the wrapper
    # must carry the checkpoint's dtype, not Qwen2_5OmniConfig's default.
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if dtype is not None:
        wrapped.dtype = dtype
    return wrapped


def register():
    """vllm.general_plugins entry point."""
    from vllm import ModelRegistry

    if THINKER_ARCH in ModelRegistry.get_supported_archs():
        # A future vLLM bump registers the thinker natively; defer to it.
        return
    ModelRegistry.register_model(
        THINKER_ARCH,
        "lazarus.models.embedding.lco_omni.model:LCOOmniThinkerForConditionalGeneration",
    )
