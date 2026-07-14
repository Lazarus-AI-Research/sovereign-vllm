"""In-process vLLM backend: one supervised process, multiple engines
(design.md §9 — generation + embedding roles behind one port).

Architecture: each role gets vLLM's OWN fully-assembled OpenAI FastAPI app
(`build_app` + `init_app_state` — the same assembly `vllm serve` uses),
running in-process. The appliance dispatches role-routed /v1 traffic to the
role's app over an in-process ASGI transport. This delegates protocol
behavior wholesale to the pinned vLLM instead of mirroring its internal
serving constructors, which reorganize between releases.

Multi-role invariants:
- Roles load strictly serially, generation first (§24 steps 8–9), so memory
  profiling never races between engines.
- The appliance downloads weights itself in the downloading state (§24 step
  6); engine child processes never touch the network.
- memory_weight maps to per-engine gpu_memory_utilization with fixed headroom
  on accelerator backends (§3.5 best-effort); CPU sizes KV cache via
  VLLM_CPU_KVCACHE_SPACE.
- Embedding dimensions are probed after load, never assumed (§10.1).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
from collections.abc import Callable

import httpx

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.backends.roleclient import RoleClientMixin
from lazarus.appliance.config import RoleConfig, RuntimeConfig

logger = logging.getLogger("sovereign.vllm")

# Fraction of accelerator memory the engines may divide between them; the
# remainder absorbs per-engine CUDA context and cudagraph overhead.
MEMORY_HEADROOM = 0.92


class _RoleApp:
    def __init__(self, engine, app, client: httpx.AsyncClient):
        self.engine = engine
        self.app = app
        self.client = client


class VllmBackend(RoleClientMixin, EngineBackend):
    def __init__(self) -> None:
        # `vllm serve` forces spawn before constructing an AsyncLLM, but this
        # in-process adapter bypasses that entrypoint. Forking the second role
        # after the first engine has started can deadlock during NCCL model-
        # parallel initialization. Preserve an explicit operator override,
        # otherwise use the CUDA-safe start method for every role engine.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        self.backend_id = os.environ.get("VLLM_BACKEND", "cpu")
        self._roles: dict[str, RoleInfo] = {}
        self._apps: dict[str, _RoleApp] = {}

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

        on_state("downloading")
        for name, role in config.roles.items():
            if role.enabled and role.source == "huggingface":
                try:
                    await self._download(role)
                except Exception as exc:
                    logger.exception("download failed for role %s", name)
                    self._roles[name] = RoleInfo(status="unhealthy", error_code=_download_error_code(exc))

        for name, role in config.roles.items():
            if not role.enabled:
                self._roles[name] = RoleInfo(status="disabled")
                continue
            if self._roles.get(name) and self._roles[name].status == "unhealthy":
                continue  # download already failed
            if name not in ("generation", "embedding"):
                # vision/audio/rerank roles are post-MVP (§27).
                self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
                logger.warning("role %s is not supported by this backend yet", name)
                continue
            on_state("loading")
            await self._start_role(name, role)

    async def _download(self, role: RoleConfig) -> None:
        from huggingface_hub import snapshot_download

        kwargs: dict = {"repo_id": role.model}
        if role.revision and role.revision not in ("<immutable-revision>",):
            kwargs["revision"] = role.revision
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(None, functools.partial(snapshot_download, **kwargs))
        logger.info("weights for %s at %s", role.model, path)

    # Appliance defaults: capabilities every deployment wants, applied to
    # every role. Unknown flags are dropped per engine version, so this list
    # can stay aspirational. Note --disable-log-requests is privacy (§2.4):
    # engine request logs can include prompt previews.
    APPLIANCE_DEFAULT_FLAGS = [
        "--enable-server-load-tracking",
        "--enable-request-id-headers",
        "--enable-force-include-usage",
        "--enable-prompt-tokens-details",
        "--disable-log-requests",
    ]

    @staticmethod
    def _infer_tool_parser(model: str) -> str | None:
        """Pick a tool-call parser from the model name. Only confident
        matches: a wrong parser corrupts outputs, so unknown models get no
        tool calling unless the admin sets tool_call_parser explicitly."""
        lowered = model.lower()
        for pattern, parser in (
            ("functiongemma", "functiongemma"),
            ("gemma-4", "gemma4"),
            ("gemma4", "gemma4"),
            ("qwen3-coder", "qwen3_coder"),
            ("qwen", "hermes"),
            ("llama-4", "llama4_pythonic"),
        ):
            if pattern in lowered:
                return parser
        return None

    @staticmethod
    def _infer_reasoning_parser(model: str) -> str | None:
        """Reasoning separation keeps thinking out of message content."""
        lowered = model.lower()
        for pattern, parser in (
            ("gemma-4", "gemma4"),
            ("gemma4", "gemma4"),
            ("qwen3", "qwen3"),
            ("deepseek-r1", "deepseek_r1"),
            ("kimi", "kimi_k2"),
        ):
            if pattern in lowered:
                return parser
        return None

    def _role_argv(self, name: str, role: RoleConfig) -> list[str]:
        argv = [
            "--model", role.model,
            "--served-model-name", role.served_model_name,
        ]
        if role.revision and role.revision not in ("<immutable-revision>", "main"):
            argv += ["--revision", role.revision]
        if role.max_model_len:
            argv += ["--max-model-len", str(role.max_model_len)]
        if self.backend_id not in ("cpu", "mock"):
            fraction = round(MEMORY_HEADROOM * role.memory_weight / 100.0, 3)
            argv += ["--gpu-memory-utilization", str(fraction)]
        argv += self.APPLIANCE_DEFAULT_FLAGS
        if name == "generation" and role.tool_call_parser != "off":
            parser = role.tool_call_parser or self._infer_tool_parser(role.model or "")
            if parser:
                argv += ["--enable-auto-tool-choice", "--tool-call-parser", parser]
            else:
                logger.warning(
                    "no tool-call parser known for %s; tool calling disabled "
                    "(set roles.generation.tool_call_parser to enable)",
                    role.model,
                )
        if name == "generation" and role.reasoning_parser != "off":
            parser = role.reasoning_parser or self._infer_reasoning_parser(role.model or "")
            if parser:
                argv += ["--reasoning-parser", parser]
        if role.engine_args:
            argv += role.engine_args
        if name == "embedding":
            argv += ["--runner", "pooling"]
            # vLLM 0.25 renamed --override-pooler-config to --pooler-config and
            # PoolerConfig's `normalize` to `use_activation` (the embed head's
            # activation IS the L2 normalization). The old spelling was being
            # silently dropped by the unknown-flag filter, so the config's
            # pooling/normalization fields never reached the engine.
            pooler: dict = {}
            if role.pooling:
                pooler["pooling_type"] = role.pooling.upper()
            if role.normalization:
                pooler["use_activation"] = role.normalization != "none"
            if pooler:
                argv += ["--pooler-config", json.dumps(pooler)]
        return argv

    async def _start_role(self, name: str, role: RoleConfig) -> None:
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.entrypoints.openai.api_server import build_app, init_app_state
            from vllm.entrypoints.openai.cli_args import make_arg_parser
            from vllm.utils.argparse_utils import FlexibleArgumentParser
            from vllm.v1.engine.async_llm import AsyncLLM

            parser = make_arg_parser(FlexibleArgumentParser())
            argv = self._role_argv(name, role)
            # Drop flags this vLLM version doesn't know (overlay-mode drift).
            known = parser._option_string_actions
            filtered: list[str] = []
            skip = False
            for i, token in enumerate(argv):
                if skip:
                    skip = False
                    continue
                if token.startswith("--") and token not in known:
                    logger.warning("dropping unsupported engine flag %s", token)
                    skip = i + 1 < len(argv) and not argv[i + 1].startswith("--")
                    continue
                filtered.append(token)
            args = parser.parse_args(filtered)

            engine_args = AsyncEngineArgs.from_cli_args(args)
            if name == "embedding" and not getattr(engine_args, "hf_overrides", None):
                # Thinker-only omni checkpoints (LCO) need their config wrapped
                # into the full-omni shape vLLM's native model expects (M12).
                # hf_overrides accepts a callable only through the Python API,
                # never through CLI argv, so it is injected here.
                from lazarus.models.embedding.lco_omni import normalize_thinker_config

                engine_args.hf_overrides = normalize_thinker_config
            engine = AsyncLLM.from_engine_args(engine_args)
            # supported_tasks gates which routers (generate vs pooling) mount.
            supported_tasks = None
            getter = getattr(engine, "get_supported_tasks", None)
            if getter is not None:
                result = getter()
                supported_tasks = await result if asyncio.iscoroutine(result) else result
            model_config = getattr(engine, "model_config", None)
            try:
                app = build_app(args, supported_tasks, model_config)
            except TypeError:  # older signature
                app = build_app(args)
            try:
                await init_app_state(engine, app.state, args, supported_tasks)
            except TypeError:
                await init_app_state(engine, app.state, args)
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://sovereign-role",
                timeout=600.0,
            )
            self._apps[name] = _RoleApp(engine, app, client)
        except Exception as exc:  # load failure → role unhealthy, process alive (§3.2)
            logger.exception("%s role failed to load", name)
            self._roles[name] = RoleInfo(status="unhealthy", error_code=_error_code(exc))
            return

        info = RoleInfo(
            status="healthy",
            engine_model=role.model,
            revision=role.revision or "main",
            context_length=getattr(getattr(engine, "model_config", None), "max_model_len", None)
            or role.max_model_len,
        )
        self._roles[name] = info

        if name == "embedding":
            try:
                result = await self.embeddings(
                    {"model": role.served_model_name, "input": "dimension probe"}
                )
                info.dimensions = len(result["data"][0]["embedding"])
                info.modalities = ["text"]
            except Exception as exc:
                logger.exception("embedding dimension probe failed")
                info.status = "unhealthy"
                info.error_code = "DIMENSION_MISMATCH"
                self._roles[name] = info
                _ = exc
                return
            # Modalities are probed, never assumed (§10.1): a modality is
            # advertised only after a real request round-trips with the
            # expected vector shape.
            info.modalities += await self._probe_embedding_modalities(role, info.dimensions)

    async def _probe_embedding_modalities(self, role: RoleConfig, dimensions: int) -> list[str]:
        found: list[str] = []
        for modality, make_part in (("image", _image_probe_part), ("audio", _audio_probe_part)):
            try:
                part = make_part()
                result = await self.embeddings(
                    {
                        "model": role.served_model_name,
                        "messages": [{"role": "user", "content": [part]}],
                    }
                )
                vector = result["data"][0]["embedding"]
                if len(vector) == dimensions:
                    found.append(modality)
                else:
                    logger.warning(
                        "%s modality probe returned %d dims (text probe said %d); not advertising",
                        modality, len(vector), dimensions,
                    )
            except Exception as exc:
                logger.info("embedding %s modality probe negative: %s", modality, exc)
        return found

    async def shutdown(self) -> None:
        apps, self._apps = dict(self._apps), {}
        for role_app in apps.values():
            await role_app.client.aclose()
            shutdown = getattr(role_app.engine, "shutdown", None)
            if shutdown is not None:
                result = shutdown()
                if asyncio.iscoroutine(result):
                    await result

    # ── introspection ────────────────────────────────────────────────────

    def role_info(self, role: str) -> RoleInfo:
        return self._roles.get(role, RoleInfo(status="disabled"))

    def role_client(self, role: str) -> httpx.AsyncClient | None:
        role_app = self._apps.get(role)
        return role_app.client if role_app else None

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

    # OpenAI-surface methods (smoke test + probes) come from RoleClientMixin;
    # live traffic is forwarded raw by the API layer via role_client.


def _image_probe_part() -> dict:
    import base64
    import io

    from PIL import Image  # ships with vLLM's multimodal stack

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}


def _audio_probe_part() -> dict:
    import base64
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 8000)  # 0.5s of silence
    encoded = base64.b64encode(buf.getvalue()).decode()
    return {"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}


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
