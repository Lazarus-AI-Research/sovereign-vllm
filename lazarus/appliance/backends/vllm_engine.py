"""In-process vLLM backend (single generation role — M3 scope; embedding
engine arrives in M8).

Imports are lazy: this module is only loaded when SOVEREIGN_ENGINE_BACKEND is
"vllm" (the default in runtime images). Delegates OpenAI protocol handling to
vLLM's own serving layer so behavior matches `vllm serve` exactly.

STATUS: written against vLLM 0.25 (pinned in constraints.txt); the serving
constructor surface drifts between releases, so kwargs are filtered by
signature. Must be validated inside the runtime image before M3 exit — the
conformance harness is the gate.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.config import RoleConfig, RuntimeConfig

logger = logging.getLogger("sovereign.vllm")


def _filtered_kwargs(callable_, /, **kwargs):
    """Drop kwargs the callable doesn't accept — tolerates serving-layer
    signature drift across pinned-version bumps in overlay mode."""
    signature = inspect.signature(callable_)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class VllmBackend(EngineBackend):
    def __init__(self) -> None:
        import os

        self.backend_id = os.environ.get("VLLM_BACKEND", "cpu")
        self._roles: dict[str, RoleInfo] = {}
        self._engine = None
        self._serving_chat = None
        self._serving_completion = None

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise BackendStartError(
                "ACCELERATOR_UNAVAILABLE",
                f"vLLM is not installed in this image: {exc}",
                recoverable=False,
            ) from exc

        for name, role in config.roles.items():
            if not role.enabled:
                self._roles[name] = RoleInfo(status="disabled")
                continue
            if name == "generation":
                on_state("downloading")
                await self._load_generation(role)
            else:
                # M8: embedding engine as second in-process AsyncLLM.
                self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
                logger.warning("role %s enabled but not supported by this backend yet (M8)", name)

    async def _load_generation(self, role: RoleConfig) -> None:
        from vllm.engine.arg_utils import AsyncEngineArgs

        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:  # pre-V1 fallback
            from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM

        engine_kwargs = dict(
            model=role.model,
            served_model_name=[role.served_model_name],
        )
        if role.revision and role.revision not in ("<immutable-revision>", "main"):
            engine_kwargs["revision"] = role.revision
        if role.max_model_len:
            engine_kwargs["max_model_len"] = role.max_model_len

        try:
            args = AsyncEngineArgs(**_filtered_kwargs(AsyncEngineArgs, **engine_kwargs))
            self._engine = AsyncLLM.from_engine_args(args)
            model_config = await self._maybe_await(self._engine.get_model_config())
            await self._build_serving(model_config, role)
        except Exception as exc:  # load-time failure → role unhealthy, process alive (§3.2)
            logger.exception("generation role failed to load")
            code = "OUT_OF_MEMORY" if "memory" in str(exc).lower() else "MODEL_LOAD_FAILED"
            self._roles["generation"] = RoleInfo(status="unhealthy", error_code=code)
            return

        self._roles["generation"] = RoleInfo(
            status="healthy",
            engine_model=role.model,
            revision=role.revision or "main",
            context_length=getattr(model_config, "max_model_len", None) or role.max_model_len,
        )

    @staticmethod
    async def _maybe_await(value):
        if inspect.isawaitable(value):
            return await value
        return value

    async def _build_serving(self, model_config, role: RoleConfig) -> None:
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
        from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels

        base_paths = [BaseModelPath(name=role.served_model_name, model_path=role.model)]
        serving_models = OpenAIServingModels(
            **_filtered_kwargs(
                OpenAIServingModels,
                engine_client=self._engine,
                model_config=model_config,
                base_model_paths=base_paths,
                lora_modules=None,
            )
        )
        init = getattr(serving_models, "init_static_loras", None)
        if init is not None:
            await self._maybe_await(init())

        self._serving_chat = OpenAIServingChat(
            **_filtered_kwargs(
                OpenAIServingChat,
                engine_client=self._engine,
                model_config=model_config,
                models=serving_models,
                response_role="assistant",
                request_logger=None,
                chat_template=None,
                chat_template_content_format="auto",
                enable_auto_tools=False,
                tool_parser=None,
            )
        )
        self._serving_completion = OpenAIServingCompletion(
            **_filtered_kwargs(
                OpenAIServingCompletion,
                engine_client=self._engine,
                model_config=model_config,
                models=serving_models,
                request_logger=None,
            )
        )

    async def shutdown(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            shutdown = getattr(engine, "shutdown", None)
            if shutdown is not None:
                await self._maybe_await(shutdown())

    # ── introspection ────────────────────────────────────────────────────

    def role_info(self, role: str) -> RoleInfo:
        return self._roles.get(role, RoleInfo(status="disabled"))

    def engine_version(self) -> str:
        try:
            import vllm

            return vllm.__version__
        except ImportError:
            return "unavailable"

    def accelerator(self) -> dict:
        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "vendor": "nvidia" if torch.version.hip is None else "amd",
                    "device_count": torch.cuda.device_count(),
                    "unified_memory": False,
                }
        except Exception:  # torch missing or device probing failed
            pass
        return {"vendor": "cpu", "device_count": 0, "unified_memory": False}

    # ── OpenAI surface ───────────────────────────────────────────────────

    async def chat_completion(self, body: dict) -> dict:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

        request = ChatCompletionRequest(**body)
        result = await self._serving_chat.create_chat_completion(request, raw_request=None)
        if isinstance(result, ErrorResponse):
            raise RuntimeError(result.model_dump_json())
        return result.model_dump()

    async def chat_completion_stream(self, body: dict) -> AsyncIterator:
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse

        request = ChatCompletionRequest(**{**body, "stream": True})
        result = await self._serving_chat.create_chat_completion(request, raw_request=None)
        if isinstance(result, ErrorResponse):
            raise RuntimeError(result.model_dump_json())
        # vLLM returns an async generator of pre-framed SSE strings.
        async for chunk in result:
            yield chunk

    async def completion(self, body: dict) -> dict:
        from vllm.entrypoints.openai.protocol import CompletionRequest, ErrorResponse

        request = CompletionRequest(**body)
        result = await self._serving_completion.create_completion(request, raw_request=None)
        if isinstance(result, ErrorResponse):
            raise RuntimeError(result.model_dump_json())
        return result.model_dump()

    async def embeddings(self, body: dict) -> dict:
        raise NotImplementedError("embedding role arrives with the multi-role milestone (M8)")
