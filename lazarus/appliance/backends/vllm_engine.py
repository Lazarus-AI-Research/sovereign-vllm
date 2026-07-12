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
from collections.abc import AsyncIterator, Callable

import httpx

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
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


class VllmBackend(EngineBackend):
    def __init__(self) -> None:
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
        if name == "embedding":
            argv += ["--runner", "pooling"]
            pooler: dict = {}
            if role.pooling:
                pooler["pooling_type"] = role.pooling.upper()
            if role.normalization:
                pooler["normalize"] = role.normalization != "none"
            if pooler:
                argv += ["--override-pooler-config", json.dumps(pooler)]
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

            engine = AsyncLLM.from_engine_args(AsyncEngineArgs.from_cli_args(args))
            app = build_app(args)
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

    # ── OpenAI surface (used by the startup smoke test and probes; live
    #    traffic is forwarded raw by the API layer via role_client) ────────

    async def _post(self, role: str, path: str, body: dict) -> dict:
        client = self.role_client(role)
        if client is None:
            raise RuntimeError(f"{role} role is not loaded")
        resp = await client.post(path, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"{path}: {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def chat_completion(self, body: dict) -> dict:
        return await self._post("generation", "/v1/chat/completions", body)

    async def chat_completion_stream(self, body: dict) -> AsyncIterator:
        client = self.role_client("generation")
        if client is None:
            raise RuntimeError("generation role is not loaded")
        async with client.stream(
            "POST", "/v1/chat/completions", json={**body, "stream": True}
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line + "\n\n"

    async def completion(self, body: dict) -> dict:
        return await self._post("generation", "/v1/completions", body)

    async def embeddings(self, body: dict) -> dict:
        return await self._post("embedding", "/v1/embeddings", body)


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
