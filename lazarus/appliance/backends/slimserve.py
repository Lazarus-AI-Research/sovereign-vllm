"""One managed SlimServe process group in an isolated interpreter."""
from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import signal
from collections.abc import Callable
from pathlib import Path

import httpx

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.backends.roleclient import RoleClientMixin
from lazarus.appliance.config import (
    QUIXICORE_CUDA_COMMIT, QUIXICORE_METAL_COMMIT, SLIMSERVE_COMMIT, RuntimeConfig,
)

INTERPRETER = "/opt/sovereign-slimserve/bin/python"
LAUNCH_MODULE = "lazarus.appliance.slimserve_launch"
PRIVATE_URL = "http://127.0.0.1:18001"
CAPABILITIES = ["chat_completions", "completions", "streaming", "text"]


def _failure(message: str, code: str = "MODEL_LOAD_FAILED") -> BackendStartError:
    return BackendStartError(code, message, role="generation")


def child_environment() -> dict[str, str]:
    """Never inherit engine/plugin/Hub overrides from the parent process."""
    allowed = ("HOME", "TMPDIR", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        PATH="/opt/sovereign-slimserve/bin:/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
        VLLM_PLUGINS="",
        VLLM_WORKER_MULTIPROC_METHOD="spawn",
    )
    return environment


async def _stop_group(process) -> None:
    # Reaping the API leader or sending SIGKILL does not prove its workers exited.
    # Retain ownership until the leader is reaped AND the owned group is absent.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=30)
    except asyncio.TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await asyncio.wait_for(process.wait(), timeout=30)
    deadline = asyncio.get_running_loop().time() + 30
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise _failure("SlimServe owned process group withdrawal is unconfirmed", "ENGINE_QUIESCE_UNAVAILABLE")
        await asyncio.sleep(0.05)


def validate_observation(config: RuntimeConfig, payload: dict) -> dict:
    """Corroborate child/worker facts; never fill a missing observed value."""
    role = config.roles.generation
    selected = role.slimserve
    if selected is None or not isinstance(payload, dict):
        raise _failure("SlimServe observation is missing")
    backend = "metal" if selected.variant == "metal" else "cuda"
    kernel_commit = QUIXICORE_METAL_COMMIT if backend == "metal" else QUIXICORE_CUDA_COMMIT
    expected = {
        "engine": {"name": "slimserve", "version": SLIMSERVE_COMMIT, "adapter": "slimserve-runtime"},
        "kernels": {"library": f"quixicore-{backend}", "version": f"{kernel_commit}+slimserve.{SLIMSERVE_COMMIT}", "backend": backend},
        "engine_profile_id": role.engine_profile_id,
        "upstream_profile_id": selected.profile_id,
        "quant": selected.quant,
        "revision": role.revision,
        "context_length": role.max_model_len,
        "max_concurrent_requests": role.max_concurrent_requests,
        "device_count": role.tensor_parallel_size,
        "tensor_parallel_size": role.tensor_parallel_size,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise _failure("SlimServe observed engine, profile, kernel or policy differs")
    for key in ("context_length", "max_concurrent_requests", "device_count", "tensor_parallel_size"):
        if type(payload.get(key)) is not int:
            raise _failure("SlimServe observed execution cardinality is invalid")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, list) or len(capabilities) != len(CAPABILITIES) or any(not isinstance(item, str) for item in capabilities) or set(capabilities) != set(CAPABILITIES):
        raise _failure("SlimServe observed API capabilities are missing or changed")
    model = next(artifact for artifact in selected.artifacts if artifact.role == "model")
    allowed_inputs = {role.model} if any(entry.file.endswith(".safetensors") for entry in model.files) else {
        str(Path(role.model) / entry.file) for entry in model.files if entry.file.endswith(".gguf")
    }
    if not isinstance(payload.get("engine_model"), str) or payload["engine_model"] not in allowed_inputs:
        raise _failure("SlimServe loaded model input differs from its verified closure")
    accelerator = payload.get("accelerator")
    if not isinstance(accelerator, dict) or type(accelerator.get("device_count")) is not int or accelerator["device_count"] != role.tensor_parallel_size:
        raise _failure("SlimServe accelerator count is missing or changed")
    if accelerator.get("vendor") != ("apple" if backend == "metal" else "nvidia") or accelerator.get("unified_memory") is not (backend == "metal"):
        raise _failure("SlimServe observed accelerator backend differs")
    devices = accelerator.get("devices")
    if not isinstance(devices, list) or len(devices) != role.tensor_parallel_size or any(not isinstance(device, dict) for device in devices):
        raise _failure("SlimServe worker device identities are missing")
    if backend == "cuda":
        if any(type(device.get("local_rank")) is not int for device in devices):
            raise _failure("SlimServe worker ranks are missing")
        devices = sorted(devices, key=lambda device: device["local_rank"])
        if [device["local_rank"] for device in devices] != list(range(role.tensor_parallel_size)):
            raise _failure("SlimServe worker rank order differs")
        for device, expected_uuid in zip(devices, role.accelerator_device_ids, strict=True):
            if device.get("identity_kind") != "nvidia_gpu_uuid" or device.get("gpu_uuid") != expected_uuid or device.get("stable_identifier") != expected_uuid:
                raise _failure("SlimServe worker UUID order differs from managed placement")
    else:
        import re
        identity = devices[0].get("stable_identifier")
        if devices[0].get("identity_kind") != "apple_platform" or not isinstance(identity, str) or not re.fullmatch(r"apple-platform-integrated-gpu-v1:[0-9a-f]{64}", identity) or devices[0].get("platform_id") != identity:
            raise _failure("SlimServe native Apple identity is unavailable")
    accepted = copy.deepcopy(payload)
    accepted["accelerator"]["devices"] = copy.deepcopy(devices)
    return accepted


class SlimServeBackend(RoleClientMixin, EngineBackend):
    engine_name = "slimserve"
    adapter_id = "slimserve-runtime"

    def __init__(self) -> None:
        self.backend_id = "unknown"
        self.generation_paused = False
        self._process = None
        self._check_process = None
        # No startup has established ownership yet. False means a launch or
        # cleanup is unresolved; only verified withdrawal establishes True.
        self._withdrawn: bool | None = None
        self._cleanup_pending = False
        self._client: httpx.AsyncClient | None = None
        self._monitor: asyncio.Task | None = None
        self._info = RoleInfo(status="disabled")
        self._observed: dict = {}
        self._token = secrets.token_urlsafe(32)
        self._config: RuntimeConfig | None = None
        self._expected_engine_model: str | None = None
        self._validated_config: str | None = None
        self._on_state = None
        self._operation = asyncio.Lock()

    @classmethod
    async def probe_available_engines(cls) -> list[dict]:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                INTERPRETER, "-m", LAUNCH_MODULE, "--availability",
                env=child_environment(), stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, start_new_session=True, cwd="/",
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=60)
            if process.returncode != 0 or len(stdout) > 4096:
                return []
            result = json.loads(stdout)
            if not isinstance(result, dict) or result.get("name") != "slimserve" or result.get("version") != SLIMSERVE_COMMIT or result.get("adapter") != "slimserve-runtime":
                return []
            variants = result.get("variants")
            if variants == ["metal"]:
                library, commit = "quixicore-metal", QUIXICORE_METAL_COMMIT
            elif variants == ["a100", "rtx3090"]:
                library, commit = "quixicore-cuda", QUIXICORE_CUDA_COMMIT
            else:
                return []
            if result.get("kernel_library") != {"name": library, "version": f"{commit}+slimserve.{SLIMSERVE_COMMIT}"}:
                return []
            if set(result) != {"name", "version", "adapter", "variants", "kernel_library"}:
                return []
            return [result]
        except (OSError, ValueError, asyncio.TimeoutError):
            return []
        finally:
            if process is not None:
                await _stop_group(process)

    async def validate(self, config: RuntimeConfig) -> None:
        async with self._operation:
            await self._validate_inner(config)

    async def _validate_inner(self, config: RuntimeConfig) -> None:
        if self._cleanup_pending or self._check_process is not None or (self._withdrawn is False and self._process is None):
            raise _failure("SlimServe lifetime ownership is unresolved", "ENGINE_QUIESCE_UNAVAILABLE")
        previously_withdrawn = self._withdrawn is not False
        self._withdrawn = False
        self._validated_config = None
        self._expected_engine_model = None
        try:
            process = self._check_process = await asyncio.create_subprocess_exec(
                INTERPRETER, "-m", LAUNCH_MODULE, "--check", env=child_environment(),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL, start_new_session=True, cwd="/",
            )
        except OSError as exc:
            self._withdrawn = previously_withdrawn
            raise _failure("pinned isolated SlimServe interpreter is unavailable", "ACCELERATOR_UNAVAILABLE") from exc
        try:
            raw = config.model_dump_json(exclude_none=True, exclude_unset=True).encode()
            stdout, _ = await asyncio.wait_for(process.communicate(raw), timeout=3600)
            if process.returncode != 0 or len(stdout) > 4096:
                raise _failure("SlimServe profile, platform or local artifacts failed validation", "CONFIG_INVALID")
            try:
                resolved = json.loads(stdout)
                model = resolved["engine_model"]
                if resolved.get("validated") is not True or not isinstance(model, str) or len(model) > 1024:
                    raise ValueError("invalid resolved model")
            except (ValueError, TypeError, KeyError) as exc:
                raise _failure("SlimServe resolved loader identity is unavailable", "CONFIG_INVALID") from exc
            self._expected_engine_model = model
            self._validated_config = raw.decode()
        except asyncio.TimeoutError as exc:
            raise _failure("SlimServe local-input verification exceeded its deadline", "CONFIG_INVALID") from exc
        finally:
            await _stop_group(process)
            self._check_process = None
            self._withdrawn = self._process is None

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        async with self._operation:
            if self._process is not None or self._withdrawn is False or self._cleanup_pending:
                raise _failure("a SlimServe generation lifetime is already managed or unresolved", "CONFIG_INVALID")
            if self._withdrawn is None:
                self._withdrawn = True
            if self._validated_config != config.model_dump_json(exclude_none=True, exclude_unset=True):
                await self._validate_inner(config)
            self._config = config
            self._on_state = on_state
            self.backend_id = "metal" if config.roles.generation.slimserve.variant == "metal" else "cuda"
            self._info = RoleInfo(status="loading")
            self._observed = {}
            on_state("loading")
            environment = child_environment()
            environment["SOVEREIGN_SLIMSERVE_OBSERVATION_TOKEN"] = self._token
            try:
                self._withdrawn = False
                # Third-party startup logs expose raw arguments and private artifact
                # paths. Keep diagnostics on the bounded first-party state/error and
                # observation channels, not inherited appliance log descriptors.
                self._process = await asyncio.create_subprocess_exec(
                    INTERPRETER, "-m", LAUNCH_MODULE, env=environment,
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True, cwd="/",
                )
                self._process.stdin.write(config.model_dump_json(exclude_none=True, exclude_unset=True).encode())
                await self._process.stdin.drain()
                self._process.stdin.close()
                self._client = httpx.AsyncClient(base_url=PRIVATE_URL, timeout=600, trust_env=False,
                    headers={"x-sovereign-observation-token": self._token})
                await self._wait_loaded()
                await self._refresh_observation()
                self._monitor = asyncio.create_task(self._watch())
            except BaseException as exc:
                # A failed/cancelled spawn without a returned handle cannot be
                # guessed absent. Ordinary spawn errors return no owned process.
                if isinstance(exc, OSError) and self._process is None:
                    self._withdrawn = True
                await self._shutdown_inner()
                self._info = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
                if isinstance(exc, (BackendStartError, asyncio.CancelledError)):
                    raise
                raise _failure("SlimServe failed to start its verified generation tree") from exc

    async def _wait_loaded(self) -> None:
        deadline = asyncio.get_running_loop().time() + 1800
        while asyncio.get_running_loop().time() < deadline:
            if self._process.returncode is not None:
                raise _failure("SlimServe generation process exited while loading")
            try:
                response = await self._client.get("/health", timeout=3)
                if response.status_code == 200:
                    role = self._config.roles.generation
                    response = await self._client.post("/v1/chat/completions", json={
                        "model": role.served_model_name,
                        "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 1,
                    })
                    response.raise_for_status()
                    body = response.json()
                    if body.get("model") != role.served_model_name or not body.get("choices"):
                        raise _failure("SlimServe startup response identity is invalid", "SMOKE_TEST_FAILED")
                    return
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            await asyncio.sleep(1)
        raise _failure("SlimServe generation did not become ready before its load deadline")

    async def _refresh_observation(self) -> None:
        response = await self._client.get("/sovereign/observation", timeout=30)
        response.raise_for_status()
        if len(response.content) > 64 * 1024:
            raise _failure("SlimServe child observation exceeds its size bound")
        observed = validate_observation(self._config, response.json())
        if self._expected_engine_model is None or observed["engine_model"] != self._expected_engine_model:
            raise _failure("SlimServe observed input differs from independently resolved model")
        self._observed = observed
        self._info = RoleInfo(status="healthy", **{key: self._observed[key] for key in (
            "engine_model", "revision", "context_length", "device_count", "tensor_parallel_size",
            "engine_profile_id", "upstream_profile_id", "quant", "max_concurrent_requests", "capabilities",
        )})

    async def _watch(self) -> None:
        try:
            while True:
                await asyncio.sleep(2)
                if self._process.returncode is not None:
                    raise _failure("SlimServe generation tree exited", "ENGINE_DEAD")
                await self._refresh_observation()
        except asyncio.CancelledError:
            raise
        except Exception:
            self.generation_paused = True
            self._observed = {}
            self._info = RoleInfo(status="unhealthy", error_code="ENGINE_DEAD")
            if self._on_state:
                self._on_state("runtime_error")
            async with self._operation:
                await self._shutdown_inner()
                self._info = RoleInfo(status="unhealthy", error_code="ENGINE_DEAD")

    async def quiesce(self) -> None:
        self.generation_paused = True
        async with self._operation:
            if self._cleanup_pending or self._check_process is not None:
                raise _failure("SlimServe lifetime cleanup is incomplete", "ENGINE_QUIESCE_UNAVAILABLE")
            if self._withdrawn and self._process is None and self._check_process is None and self._client is None:
                return
            if self._client is None or self._process is None or self._process.returncode is not None:
                raise _failure("SlimServe is unavailable for quiescence", "ENGINE_QUIESCE_UNAVAILABLE")
            response = await self._client.post("/sovereign/quiesce", timeout=None)
            if response.status_code != 200 or response.json() != {"quiesced": True}:
                raise _failure("SlimServe engine did not acknowledge idle", "ENGINE_QUIESCE_UNAVAILABLE")

    async def resume(self) -> None:
        self.generation_paused = True
        async with self._operation:
            if self._cleanup_pending or self._check_process is not None:
                raise _failure("SlimServe lifetime cleanup is incomplete", "ENGINE_RESUME_UNAVAILABLE")
            if self._client is None or self._process is None or self._process.returncode is not None:
                raise _failure("SlimServe is unavailable for resume", "ENGINE_RESUME_UNAVAILABLE")
            response = await self._client.post("/sovereign/resume", timeout=30)
            if response.status_code != 200 or response.json() != {"resumed": True}:
                raise _failure("SlimServe engine did not acknowledge resume", "ENGINE_RESUME_UNAVAILABLE")
            self.generation_paused = False

    async def _shutdown_inner(self) -> None:
        # The host preflights a fresh candidate then closes that check lifetime
        # before its first start. It has never exposed generation ingress.
        # Every attempted generation/unknown lifetime closes ingress; an existing
        # pause is never cleared by cleanup or a subsequent start.
        if self._config is not None or self._withdrawn is not True:
            self.generation_paused = True
        self._cleanup_pending = True
        self._observed = {}
        self._info = RoleInfo(status="unhealthy", error_code="ENGINE_DEAD")
        withdrew = self._withdrawn
        if self._monitor is not None:
            if self._monitor is not asyncio.current_task():
                self._monitor.cancel()
                await asyncio.gather(self._monitor, return_exceptions=True)
            self._monitor = None
        if self._check_process is not None:
            await _stop_group(self._check_process)
            withdrew = True
        if self._process is not None:
            await _stop_group(self._process)
            withdrew = True
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._process = None
        self._check_process = None
        self._cleanup_pending = False
        self._withdrawn = withdrew
        self._observed = {}
        self._info = RoleInfo(status="disabled")
        self._validated_config = None
        self._expected_engine_model = None

    async def shutdown(self) -> None:
        async with self._operation:
            await self._shutdown_inner()

    def role_info(self, role: str) -> RoleInfo:
        return self._info if role == "generation" else RoleInfo(status="disabled")

    def role_client(self, role: str) -> httpx.AsyncClient | None:
        return self._client if role == "generation" else None

    def engine_version(self) -> str | None:
        return self._observed.get("engine", {}).get("version")

    def observation(self) -> dict:
        return copy.deepcopy(self._observed)

    def accelerator(self) -> dict:
        return copy.deepcopy(self._observed.get("accelerator", {"vendor": "none", "device_count": 0, "unified_memory": False}))
