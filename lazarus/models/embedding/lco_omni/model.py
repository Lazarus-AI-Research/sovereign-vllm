"""vLLM model class for thinker-only Qwen2.5-Omni checkpoints (M12).

Imported lazily by the model registry (string reference in `register()`), so
this module may import vLLM internals; keep `lazarus.models.embedding.lco_omni`
itself import-light.
"""

from __future__ import annotations

from vllm.model_executor.models.qwen2_5_omni_thinker import (
    Qwen2_5OmniThinkerDummyInputsBuilder,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerMultiModalProcessor,
    Qwen2_5OmniThinkerProcessingInfo,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.multimodal import MULTIMODAL_REGISTRY


@MULTIMODAL_REGISTRY.register_processor(
    Qwen2_5OmniThinkerMultiModalProcessor,
    info=Qwen2_5OmniThinkerProcessingInfo,
    dummy_inputs=Qwen2_5OmniThinkerDummyInputsBuilder,
)
class LCOOmniThinkerForConditionalGeneration(Qwen2_5OmniThinkerForConditionalGeneration):
    """Native thinker model, tolerant of thinker-only checkpoint weight names.

    The parent expects full-omni names (`thinker.model.*`); thinker-only
    checkpoints ship the same tensors unprefixed (`model.*`, `lm_head.*`,
    `visual.*`, `audio_tower.*`).  Prefix rules apply in order, so full-omni
    names still map identically and never double-match the bare rules.
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "thinker.lm_head.": "language_model.lm_head.",
            "thinker.model.": "language_model.model.",
            "thinker.": "",
            "lm_head.": "language_model.lm_head.",
            "model.": "language_model.model.",
        }
    )
