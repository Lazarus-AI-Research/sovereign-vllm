"""In-process vLLM backend: one supervised process, multiple engines
(design.md §9 — generation + embedding roles behind one port).

Imports are lazy: this module is only loaded when SOVEREIGN_ENGINE_BACKEND is
"vllm" (the default in runtime images). Delegates OpenAI protocol handling to
vLLM's own serving layer so behavior matches `vllm serve` exactly.

Multi-role notes:
- Roles load strictly serially, generation first (§24 steps 8–9), so GPU
  memory profiling never races between engines.
- memory_weight maps to per-engine gpu_memory_utilization with fixed headroom
  on accelerator backends; the CPU backend sizes its KV cache via
  VLLM_CPU_KVCACHE_SPACE instead (§3.5: best-effort, observed via metrics).
- Embedding dimensions are probed after load, never assumed (§10.1).

STATUS: written against vLLM 0.25 (pinned in constraints.txt); the serving
constructor surface drifts between releases, so kwargs are filtered by
signature. The conformance harness inside the runtime image is the gate.
"""

from __future__ import annotations

import inspect
import logging
import os
from collections.abc import AsyncIterator, Callable

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.config import RoleConfig, RuntimeConfig

logger = logging.getLogger("sovereign.vllm")

# Fraction of accelerator memory the two engines may divide between them;
# the remainder absorbs per-engine CUDA context and cudagraph overhead.
MEMORY_HEADROOM = 0.92


def _filtered_kwargs(callable_, /, **kwargs):
    """Drop kwargs the callable doesn't accept — tolerates serving-layer
    signature drift across pinned-version bumps in overlay mode."""
    signature = inspect.signature(callable_)
    if any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values()):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


class VllmBackend(EngineBackend):
    def __init__(self) -> None:
        self.backend_id = os.environ.get("VLLM_BACKEND", "cpu")
        self._roles: dict[str, RoleInfo] = {}
        self._engines: dict[str, object] = {}
        self._serving_chat = None
        self._serving_completion = None
        self._serving_embedding = None

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

        # §24 step 6: the appliance downloads/verifies weights itself, in the
        # downloading state, before any engine starts. The engine then loads
        # from the local cache — engine-core child processes never touch the
        # network (whose shared HTTP clients do not survive the fork).
        on_state("downloading")
        for name, role in config.roles.items():
            if role.enabled and role.source == "huggingface":
                try:
                    await self._download(role)
                except Exception as exc:
                    logger.exception("download failed for role %s", name)
                    self._roles[name] = RoleInfo(status="unhealthy", error_code=_download_error_code(exc))

        # RolesSection.items() is ordered: generation loads before embedding.
        for name, role in config.roles.items():
            if not role.enabled:
                self._roles[name] = RoleInfo(status="disabled")
                continue
            if self._roles.get(name) and self._roles[name].status == "unhealthy":
                continue  # download already failed
            on_state("loading")
            if name == "generation":
                await self._load_generation(role)
            elif name == "embedding":
                await self._load_embedding(role)
            else:
                # vision/audio/rerank roles are post-MVP (§27).
                self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
                logger.warning("role %s is not supported by this backend yet", name)

    async def _download(self, role: RoleConfig) -> None:
        import asyncio
        import functools

        from huggingface_hub import snapshot_download

        kwargs: dict = {"repo_id": role.model}
        if role.revision and role.revision not in ("<immutable-revision>",):
            kwargs["revision"] = role.revision
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, functools.partial(snapshot_download, **kwargs))
        logger.info("weights for %s at %s", role.model, path)

    def _engine_kwargs(self, role: RoleConfig) -> dict:
        kwargs = dict(
            model=role.model,
            served_model_name=[role.served_model_name],
        )
        if role.revision and role.revision not in ("<immutable-revision>", "main"):
            kwargs["revision"] = role.revision
        if role.max_model_len:
            kwargs["max_model_len"] = role.max_model_len
        if self.backend_id not in ("cpu", "mock"):
            # §3.5: best-effort weight → per-engine memory fraction.
            kwargs["gpu_memory_utilization"] = round(
                MEMORY_HEADROOM * role.memory_weight / 100.0, 3
            )
        return kwargs

    async def _build_engine(self, extra_kwargs: dict):
        from vllm.engine.arg_utils import AsyncEngineArgs

        try:
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError:  # pre-V1 fallback
            from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM

        args = AsyncEngineArgs(**_filtered_kwargs(AsyncEngineArgs, **extra_kwargs))
        engine = AsyncLLM.from_engine_args(args)
        model_config = await self._maybe_await(engine.get_model_config())
        return engine, model_config

    async def _load_generation(self, role: RoleConfig) -> None:
        try:
            engine, model_config = await self._build_engine(self._engine_kwargs(role))
            self._engines["generation"] = engine
            await self._build_generation_serving(engine, model_config, role)
        except Exception as exc:  # load failure → role unhealthy, process alive (§3.2)
            logger.exception("generation role failed to load")
            self._roles["generation"] = RoleInfo(status="unhealthy", error_code=_error_code(exc))
            return

        self._roles["generation"] = RoleInfo(
            status="healthy",
            engine_model=role.model,
            revision=role.revision or "main",
            context_length=getattr(model_config, "max_model_len", None) or role.max_model_len,
        )

    async def _load_embedding(self, role: RoleConfig) -> None:
        kwargs = self._engine_kwargs(role)
        # Across 0.2x releases the embed task moved from task= to runner=;
        # pass both, _filtered_kwargs keeps whichever exists.
        kwargs["task"] = "embed"
        kwargs["runner"] = "pooling"
        pooler = self._pooler_config(role)
        if pooler is not None:
            kwargs["override_pooler_config"] = pooler
        try:
            engine, model_config = await self._build_engine(kwargs)
            self._engines["embedding"] = engine
            await self._build_embedding_serving(engine, model_config, role)
            dimensions = await self._probe_dimensions(role)
        except Exception as exc:
            logger.exception("embedding role failed to load")
            self._roles["embedding"] = RoleInfo(status="unhealthy", error_code=_error_code(exc))
            return

        self._roles["embedding"] = RoleInfo(
            status="healthy",
            engine_model=role.model,
            revision=role.revision or "main",
            context_length=getattr(model_config, "max_model_len", None) or role.max_model_len,
            dimensions=dimensions,
            modalities=["text"],
        )

    @staticmethod
    def _pooler_config(role: RoleConfig):
        try:
            from vllm.config import PoolerConfig
        except ImportError:
            return None
        pooling = (role.pooling or "last").upper()
        return PoolerConfig(
            **_filtered_kwargs(
                PoolerConfig,
                pooling_type=pooling,
                normalize=role.normalization != "none",
            )
        )

    @staticmethod
    async def _maybe_await(value):
        if inspect.isawaitable(value):
            return await value
        return value

    async def _serving_models(self, engine, model_config, role: RoleConfig):
        from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels

        serving_models = OpenAIServingModels(
            **_filtered_kwargs(
                OpenAIServingModels,
                engine_client=engine,
                model_config=model_config,
                base_model_paths=[BaseModelPath(name=role.served_model_name, model_path=role.model)],
                lora_modules=None,
            )
        )
        init = getattr(serving_models, "init_static_loras", None)
        if init is not None:
            await self._maybe_await(init())
        return serving_models

    async def _build_generation_serving(self, engine, model_config, role: RoleConfig) -> None:
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

        serving_models = await self._serving_models(engine, model_config, role)
        self._serving_chat = OpenAIServingChat(
            **_filtered_kwargs(
                OpenAIServingChat,
                engine_client=engine,
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
                engine_client=engine,
                model_config=model_config,
                models=serving_models,
                request_logger=None,
            )
        )

    async def _build_embedding_serving(self, engine, model_config, role: RoleConfig) -> None:
        from vllm.entrypoints.openai.serving_embedding import OpenAIServingEmbedding

        serving_models = await self._serving_models(engine, model_config, role)
        self._serving_embedding = OpenAIServingEmbedding(
            **_filtered_kwargs(
                OpenAIServingEmbedding,
                engine_client=engine,
                model_config=model_config,
                models=serving_models,
                request_logger=None,
                chat_template=None,
                chat_template_content_format="auto",
            )
        )

    async def _probe_dimensions(self, role: RoleConfig) -> int:
        """§10.1: discover the output dimension from the loaded checkpoint."""
        result = await self.embeddings({"model": role.served_model_name, "input": "dimension probe"})
        vector = result["data"][0]["embedding"]
        return len(vector)

    async def shutdown(self) -> None:
        engines, self._engines = dict(self._engines), {}
        for engine in engines.values():
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
        from vllm.entrypoints.openai.protocol import EmbeddingRequest, ErrorResponse

        if self._serving_embedding is None:
            raise RuntimeError("embedding role is not loaded")
        request_type = EmbeddingRequest
        # EmbeddingRequest is a union alias in some releases; resolve to the
        # completion-style request when so.
        if hasattr(request_type, "__args__"):
            request_type = request_type.__args__[0]
        request = request_type(**body)
        result = await self._serving_embedding.create_embedding(request, raw_request=None)
        if isinstance(result, ErrorResponse):
            raise RuntimeError(result.model_dump_json())
        return result.model_dump()


def _error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "memory" in message or "oom" in message:
        return "OUT_OF_MEMORY"
    if "revision" in message:
        return "MODEL_REVISION_NOT_FOUND"
    if "not found" in message or "404" in message or "does not exist" in message:
        return "MODEL_NOT_FOUND"
    return "MODEL_LOAD_FAILED"


def _download_error_code(exc: Exception) -> str:
    message = str(exc).lower()
    if "revision" in message:
        return "MODEL_REVISION_NOT_FOUND"
    if "404" in message or "not found" in message or "repository" in message:
        return "MODEL_NOT_FOUND"
    return "MODEL_DOWNLOAD_FAILED"
