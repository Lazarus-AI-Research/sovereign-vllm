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
- Hugging Face sources download in the downloading state (§24 step 6).
  Managed local CUDA generation consumes Control's prepared directory without
  downloading, repairing metadata, or switching to a repository source.
- memory_weight maps to per-engine gpu_memory_utilization with fixed headroom
  on accelerator backends (§3.5 best-effort); CPU sizes KV cache via
  VLLM_CPU_KVCACHE_SPACE.
- Embedding dimensions are probed after load, never assumed (§10.1).
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import os
import stat
from collections.abc import Callable
from pathlib import Path

import httpx

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.backends.roleclient import RoleClientMixin
from lazarus.appliance.config import RoleConfig, RuntimeConfig

logger = logging.getLogger("sovereign.vllm")

# Fraction of accelerator memory the engines may divide between them; the
# remainder absorbs per-engine CUDA context and cudagraph overhead.
MEMORY_HEADROOM = 0.92

# This is a consumer shape check, not another artifact manifest. Control owns
# the immutable filenames, revisions, byte lengths, SHA-256s and staging lock.
LOCAL_SUPPORT_FILES = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "chat_template.jinja",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.model",
        "vocab.txt",
        "model.safetensors.index.json",
    }
)
LOCAL_SUPPORT_FILE_LIMIT = 64 * 1024 * 1024
LOCAL_SUPPORT_TOTAL_LIMIT = 256 * 1024 * 1024


def _local_cuda_error(detail: str) -> BackendStartError:
    return BackendStartError(
        "CONFIG_INVALID",
        f"local CUDA generation bundle {detail}",
        role="generation",
        recoverable=True,
    )


def _local_bundle_json(path: Path) -> dict:
    with path.open("rb") as source:
        contents = source.read(LOCAL_SUPPORT_FILE_LIMIT + 1)
    if len(contents) > LOCAL_SUPPORT_FILE_LIMIT:
        raise _local_cuda_error("metadata exceeds the supported size limit")
    try:
        value = json.loads(contents)
    except (ValueError, UnicodeError, RecursionError):
        raise _local_cuda_error("contains invalid configuration or weight-index JSON") from None
    if not isinstance(value, dict):
        raise _local_cuda_error("configuration and weight index must be JSON objects")
    return value


def _validate_local_cuda_bundle(model: str | None) -> None:
    """Reject incomplete/ambiguous local inputs before importing the engine.

    Do not hash weights or duplicate Control's sealed support-file manifest.
    Directory enumeration and metadata reads are bounded; errors never echo
    the private path, unapproved filenames, or file contents.
    """
    if not model:
        raise _local_cuda_error("requires a prepared model directory")
    root = Path(model)
    try:
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise _local_cuda_error("requires a prepared directory, not a file or symlink")
        support: set[str] = set()
        weights: list[str] = []
        total = 0
        with os.scandir(root) as entries:
            for count, entry in enumerate(entries, start=1):
                if count > 17:  # One primary artifact and at most 16 support files.
                    raise _local_cuda_error("contains too many files")
                metadata = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size == 0:
                    raise _local_cuda_error("must contain only nonempty regular files")
                if entry.name in LOCAL_SUPPORT_FILES:
                    support.add(entry.name)
                    total += metadata.st_size
                    if (
                        metadata.st_size > LOCAL_SUPPORT_FILE_LIMIT
                        or total > LOCAL_SUPPORT_TOTAL_LIMIT
                    ):
                        raise _local_cuda_error("metadata exceeds the supported size limit")
                elif entry.name.endswith(".safetensors") and not entry.name.startswith("."):
                    weights.append(entry.name)
                    if len(weights) > 1:
                        raise _local_cuda_error("requires exactly one primary safetensors file")
                else:
                    raise _local_cuda_error("contains an unsupported file")
        if len(weights) != 1:
            raise _local_cuda_error("requires exactly one primary safetensors file")
        if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= support:
            raise _local_cuda_error("is missing required configuration or tokenizer files")
        config = _local_bundle_json(root / "config.json")
        # These shipped multimodal architectures consume processor metadata
        # even when the appliance exposes only text generation.
        model_type = config.get("model_type")
        if model_type in ("gemma4", "qwen3_5") and "processor_config.json" not in support:
            raise _local_cuda_error("is missing required processor metadata")
        if (
            model_type == "qwen3_5"
            and not {"preprocessor_config.json", "video_preprocessor_config.json"} <= support
        ):
            raise _local_cuda_error("is missing required processor metadata")
        if "model.safetensors.index.json" in support:
            index = _local_bundle_json(root / "model.safetensors.index.json")
            weight_map = index.get("weight_map")
            if (
                not isinstance(weight_map, dict)
                or not weight_map
                or any(filename != weights[0] for filename in weight_map.values())
            ):
                raise _local_cuda_error("weight index does not match the primary safetensors file")
    except (OSError, ValueError):
        raise _local_cuda_error("is unavailable or unreadable") from None


class _RoleApp:
    def __init__(self, engine, app, client: httpx.AsyncClient):
        self.engine = engine
        self.app = app
        self.client = client


class VllmBackend(RoleClientMixin, EngineBackend):
    engine_name = "vllm"
    adapter_id = "sovereign-runtime"

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
        self._engines: dict[str, object] = {}
        self._starting = False
        self._startup_attempted = False
        self._start_task: asyncio.Task | None = None
        self._loading_roles: set[str] = set()
        self._construction_tasks: set[asyncio.Task] = set()
        # Pinned AsyncLLM shutdown does not wait for every worker descendant.
        # Once its constructor is entered, missing apps can never establish
        # full withdrawal; recovery needs actual idle or process supervision.
        self._native_lifetime_started = False
        self._cleanup_pending = False
        self._local_cuda_generation = self.backend_id == "cuda"
        self.generation_paused = False

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        if self._starting or self._native_lifetime_started or self._apps:
            raise BackendStartError("CONFIG_INVALID", "vLLM engine lifetime is already managed or unresolved")
        self._starting = True
        self._startup_attempted = True
        self._start_task = asyncio.current_task()
        try:
            await self._start_inner(config, on_state)
        finally:
            self._starting = False
            self._start_task = None

    async def _start_inner(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        self._local_cuda_generation = self.backend_id == "cuda" or config.runtime.profile in (
            "cuda-x86_64",
            "cuda-arm64-dgx-spark",
        )
        generation = config.roles.generation
        if self._local_cuda_generation and generation.enabled and generation.source == "local":
            _validate_local_cuda_bundle(generation.model)
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise BackendStartError(
                "ACCELERATOR_UNAVAILABLE",
                f"vLLM is not installed in this image: {exc}",
                recoverable=False,
            ) from exc
        configured_devices = config.roles.generation.accelerator_device_ids
        if configured_devices:
            observed = self.accelerator()
            observed_devices = [item["gpu_uuid"] for item in observed.get("devices", [])]
            if observed_devices != configured_devices:
                raise BackendStartError(
                    "ACCELERATOR_UNAVAILABLE",
                    "visible CUDA devices do not match the managed generation placement",
                    recoverable=True,
                )

        on_state("downloading")
        for name, role in config.roles.items():
            if role.enabled and role.source == "huggingface":
                try:
                    await self._download(role)
                except Exception as exc:
                    logger.exception("download failed for role %s", name)
                    self._roles[name] = RoleInfo(
                        status="unhealthy", error_code=_download_error_code(exc)
                    )

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

    async def quiesce(self) -> None:
        self.generation_paused = True
        if self._cleanup_pending:
            raise BackendStartError("ENGINE_QUIESCE_UNAVAILABLE", "engine cleanup is unconfirmed")
        if self._starting or self._loading_roles or self._construction_tasks:
            raise BackendStartError("ENGINE_QUIESCE_UNAVAILABLE", "engine construction has not settled")
        role_app = self._apps.get("generation")
        if role_app is None:
            if self._startup_attempted and not self._native_lifetime_started and not self._engines and not self._apps:
                return  # Positively no constructor was entered; ingress stays closed.
            raise BackendStartError("ENGINE_QUIESCE_UNAVAILABLE", "generation engine is unavailable")
        try:
            # Pinned vLLM 0.25 awaits CoreProc's idle callback Future here.
            # The API has already drained admitted responses: wait mode freezes
            # queued work, so it cannot replace that outer drain. Never abort.
            await role_app.engine.pause_generation(mode="wait", clear_cache=False)
            if await role_app.engine.is_paused() is not True:
                raise RuntimeError("generation pause was not acknowledged")
        except Exception:
            raise BackendStartError(
                "ENGINE_QUIESCE_FAILED", "generation pause was not acknowledged"
            ) from None

    async def resume(self) -> None:
        self.generation_paused = True
        if self._cleanup_pending:
            raise BackendStartError("ENGINE_RESUME_UNAVAILABLE", "engine cleanup is unconfirmed")
        if self._starting or self._loading_roles or self._construction_tasks:
            raise BackendStartError("ENGINE_RESUME_UNAVAILABLE", "engine construction has not settled")
        role_app = self._apps.get("generation")
        if role_app is None:
            raise BackendStartError("ENGINE_RESUME_UNAVAILABLE", "generation engine is unavailable")
        try:
            await role_app.engine.resume_generation()
            if await role_app.engine.is_paused() is not False:
                raise RuntimeError("generation resume was not acknowledged")
        except Exception:
            raise BackendStartError(
                "ENGINE_RESUME_FAILED", "generation resume was not acknowledged"
            ) from None
        self.generation_paused = False

    async def _download(self, role: RoleConfig) -> None:
        if role.source == "local":
            return  # Local means prepared by Control, never repaired with a Hub snapshot.
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
            "--model",
            role.model,
            "--served-model-name",
            role.served_model_name,
        ]
        if role.revision and role.revision not in ("<immutable-revision>", "main"):
            argv += ["--revision", role.revision]
        if role.max_model_len:
            argv += ["--max-model-len", str(role.max_model_len)]
        if self.backend_id not in ("cpu", "mock"):
            fraction = round(MEMORY_HEADROOM * role.memory_weight / 100.0, 3)
            argv += ["--gpu-memory-utilization", str(fraction)]
        if name == "generation":
            argv += ["--tensor-parallel-size", str(role.tensor_parallel_size)]
            if role.engine_profile_id is not None:
                argv += ["--max-num-seqs", str(role.max_concurrent_requests)]
        if role.enforce_eager:
            argv.append("--enforce-eager")
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

    @staticmethod
    def _applied_generation_info(engine, app, role: RoleConfig, supported_tasks) -> RoleInfo:
        """Publish a selected identity only after matching the loaded engine."""
        config = getattr(engine, "vllm_config", None)
        model = getattr(engine, "model_config", None)
        scheduler = getattr(config, "scheduler_config", None)
        parallel = getattr(config, "parallel_config", None)
        context = getattr(model, "max_model_len", None)
        concurrency = getattr(scheduler, "max_num_seqs", None)
        tensor_parallel = getattr(parallel, "tensor_parallel_size", None)
        alias = getattr(model, "served_model_name", None)
        if (
            not role.model
            or getattr(model, "model", None) != role.model
            or not role.served_model_name
            or alias not in (role.served_model_name, [role.served_model_name])
            or type(context) is not int
            or context != role.max_model_len
            or type(concurrency) is not int
            or concurrency != role.max_concurrent_requests
            or type(tensor_parallel) is not int
            or tensor_parallel != role.tensor_parallel_size
        ):
            raise RuntimeError("vLLM loaded execution does not match the selected profile")
        quant = getattr(model, "quantization", None)
        if quant is None:
            dtype = getattr(model, "dtype", None)
            quant = str(dtype).removeprefix("torch.") if dtype is not None else None
            if quant not in ("bfloat16", "float16", "float32"):
                raise RuntimeError("vLLM loaded model dtype is unavailable")
        if not isinstance(quant, str) or not quant.strip():
            raise RuntimeError("vLLM loaded model quantization is unavailable")
        paths = {
            route.path for route in app.routes
            if "POST" in (getattr(route, "methods", None) or ())
        }
        if (
            not isinstance(supported_tasks, (tuple, list))
            or "generate" not in supported_tasks
            or not {"/v1/chat/completions", "/v1/completions"}.issubset(paths)
        ):
            raise RuntimeError("vLLM selected profile generation APIs are unavailable")
        return RoleInfo(
            status="healthy",
            engine_model=model.model,
            revision=role.revision or "main",
            context_length=context,
            device_count=len(role.accelerator_device_ids) or None,
            tensor_parallel_size=tensor_parallel,
            engine_profile_id=role.engine_profile_id,
            quant=quant,
            max_concurrent_requests=concurrency,
            capabilities=["chat_completions", "completions", "streaming", "text"],
        )

    async def _start_role(self, name: str, role: RoleConfig) -> None:
        selected_generation = name == "generation" and role.engine_profile_id is not None
        self._startup_attempted = True
        if name in self._engines or name in self._loading_roles:
            raise BackendStartError("CONFIG_INVALID", "role engine lifetime is already managed", role=name)
        self._loading_roles.add(name)
        engine = None
        applied_info = None
        try:
            engine, args, build_app, init_app_state = await self._construct_role_engine(name, role)
            self._engines[name] = engine
            self._native_lifetime_started = True
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
            if (
                name == "generation"
                and getattr(args, "tensor_parallel_size", None) != role.tensor_parallel_size
            ):
                raise RuntimeError("vLLM did not apply the managed tensor-parallel size")
            if selected_generation:
                applied_info = self._applied_generation_info(engine, app, role, supported_tasks)
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://sovereign-role",
                timeout=600.0,
            )
            self._apps[name] = _RoleApp(engine, app, client)
        except BaseException as exc:
            self.generation_paused = True
            self._roles[name] = RoleInfo(status="unhealthy", error_code=_error_code(exc))
            # Keep every returned engine, including unselected generation and
            # embedding failures. The app may never have been registered.
            engine = self._engines.get(name)
            if engine is not None:
                self._cleanup_pending = True
                try:
                    await self._shutdown_engine(engine)
                except Exception:
                    logger.exception("failed %s engine cleanup is unconfirmed", name)
                    raise BackendStartError(
                        "ENGINE_DEAD", "failed engine cleanup is unconfirmed", role=name, recoverable=False,
                    ) from None
            if isinstance(exc, (BackendStartError, asyncio.CancelledError)) or not isinstance(exc, Exception):
                raise
            logger.exception("%s role failed to load", name)
            return
        finally:
            self._loading_roles.discard(name)

        info = applied_info or RoleInfo(
            status="healthy",
            engine_model=getattr(model_config, "model", None),
            revision=role.revision or "main",
            context_length=getattr(getattr(engine, "model_config", None), "max_model_len", None)
            or role.max_model_len,
            device_count=len(role.accelerator_device_ids) or None,
            tensor_parallel_size=role.tensor_parallel_size
            if name == "generation" and role.accelerator_device_ids
            else None,
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

    async def _construct_role_engine(self, name: str, role: RoleConfig):
        """Construct vLLM off the API event loop.

        Model configuration, tokenizer setup, multiprocessing startup, and
        weight loading are synchronous inside ``AsyncLLM.from_engine_args``.
        Keeping that work on the Uvicorn loop makes even ``/health/live``
        unresponsive for minutes, defeating the appliance state contract.
        AsyncLLM starts its output handler lazily on the first request when it
        is constructed outside a running loop, so the returned engine safely
        attaches to the main loop during the probes below.
        """
        task = asyncio.create_task(asyncio.to_thread(self._construct_role_engine_sync, name, role))
        self._construction_tasks.add(task)
        task.add_done_callback(self._construction_tasks.discard)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # to_thread cancellation cannot stop a running constructor. Collect
            # its result before cancellation unwinds startup and cleanup.
            try:
                await asyncio.shield(task)
            except Exception:
                pass
            raise

    def _construct_role_engine_sync(self, name: str, role: RoleConfig):
        if name == "generation" and role.source == "local" and self._local_cuda_generation:
            _validate_local_cuda_bundle(role.model)
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import build_app, init_app_state
        from vllm.entrypoints.openai.cli_args import make_arg_parser
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        from vllm.v1.engine.async_llm import AsyncLLM

        parser = make_arg_parser(FlexibleArgumentParser())
        argv = self._role_argv(name, role)
        # Optional overlay defaults may drift; never drop the model source,
        # public alias, immutable revision or managed execution placement.
        known = parser._option_string_actions
        filtered: list[str] = []
        skip = False
        for i, token in enumerate(argv):
            if skip:
                skip = False
                continue
            if token.startswith("--") and token not in known:
                if token in (
                    "--model",
                    "--served-model-name",
                    "--revision",
                    "--tensor-parallel-size",
                    "--max-num-seqs",
                    "--enforce-eager",
                ):
                    raise BackendStartError(
                        "CONFIG_INVALID",
                        "vLLM does not support the required structured launch arguments",
                        role=name,
                        recoverable=True,
                    )
                logger.warning("dropping unsupported engine flag %s", token)
                skip = i + 1 < len(argv) and not argv[i + 1].startswith("--")
                continue
            filtered.append(token)
        args = parser.parse_args(filtered)

        engine_args = AsyncEngineArgs.from_cli_args(args)
        self._native_lifetime_started = True
        engine = AsyncLLM.from_engine_args(engine_args)
        self._engines[name] = engine
        return engine, args, build_app, init_app_state

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
                        modality,
                        len(vector),
                        dimensions,
                    )
            except Exception as exc:
                logger.info("embedding %s modality probe negative: %s", modality, exc)
        return found

    @staticmethod
    async def _shutdown_engine(engine) -> None:
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is None:
            raise BackendStartError("ENGINE_DEAD", "engine cleanup is unavailable", recoverable=False)
        result = shutdown()
        if inspect.isawaitable(result):
            await result

    async def shutdown(self) -> None:
        self.generation_paused = True
        start_task = self._start_task
        if start_task is not None and start_task is not asyncio.current_task():
            if not start_task.cancelling():
                start_task.cancel()
            await asyncio.gather(start_task, return_exceptions=True)
        if self._starting or self._loading_roles or self._construction_tasks:
            raise BackendStartError("ENGINE_DEAD", "engine construction has not settled", recoverable=False)
        self._cleanup_pending = True
        failed = False
        for name, role_app in list(self._apps.items()):
            try:
                await role_app.client.aclose()
            except Exception:
                failed = True
                logger.exception("%s client cleanup failed", name)
            else:
                del self._apps[name]
        for name, engine in self._engines.items():
            try:
                await self._shutdown_engine(engine)
            except Exception:
                failed = True
                logger.exception("%s engine cleanup is unconfirmed", name)
        self._cleanup_pending = self._native_lifetime_started or failed
        if failed:
            raise BackendStartError("ENGINE_DEAD", "engine cleanup is unconfirmed", recoverable=False)

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
                devices = []
                for index in range(torch.cuda.device_count()):
                    raw_uuid = getattr(torch.cuda.get_device_properties(index), "uuid", None)
                    gpu_uuid = _canonical_torch_uuid(raw_uuid)
                    if gpu_uuid is not None:
                        devices.append(
                            {
                                "identity_kind": "nvidia_gpu_uuid",
                                "stable_identifier": gpu_uuid,
                                "local_rank": index,
                                "gpu_uuid": gpu_uuid,
                            }
                        )
                result = {
                    "vendor": "nvidia" if torch.version.hip is None else "amd",
                    "device_count": torch.cuda.device_count(),
                    "unified_memory": False,
                }
                if len(devices) == result["device_count"]:
                    result["devices"] = devices
                return result
        except Exception:
            pass
        return {"vendor": "cpu", "device_count": 0, "unified_memory": False}


def _canonical_torch_uuid(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes) and len(value) == 16:
        value = value.hex()
    value = str(value).lower().removeprefix("gpu-").replace("-", "")
    if len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        return None
    return f"GPU-{value[:8]}-{value[8:12]}-{value[12:16]}-{value[16:20]}-{value[20:]}"

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
