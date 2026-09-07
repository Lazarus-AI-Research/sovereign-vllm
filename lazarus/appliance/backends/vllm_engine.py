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
        self._local_cuda_generation = self.backend_id == "cuda"

    # ── lifecycle ────────────────────────────────────────────────────────

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
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

    async def _start_role(self, name: str, role: RoleConfig) -> None:
        try:
            engine, args, build_app, init_app_state = await self._construct_role_engine(name, role)
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
            client = httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://sovereign-role",
                timeout=600.0,
            )
            self._apps[name] = _RoleApp(engine, app, client)
        except BackendStartError:
            raise
        except Exception as exc:  # load failure → role unhealthy, process alive (§3.2)
            logger.exception("%s role failed to load", name)
            self._roles[name] = RoleInfo(status="unhealthy", error_code=_error_code(exc))
            return

        info = RoleInfo(
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
        return await asyncio.to_thread(self._construct_role_engine_sync, name, role)

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
        engine = AsyncLLM.from_engine_args(engine_args)
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
