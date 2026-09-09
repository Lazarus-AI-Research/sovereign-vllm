"""sovereign-runtime-agent: supervise llama.cpp servers, serve one private
port. Fails closed: no token, no service. Binds loopback only — the agent is
never exposed beyond the host (§22)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import secrets
import subprocess
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
import yaml
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from lazarus.agent.config import AgentConfig, load_agent_config
from lazarus.appliance.backends.base import BackendStartError
from lazarus.appliance.backends.slimserve import SlimServeBackend
from lazarus.appliance.config import RuntimeConfig
from lazarus.appliance.manifest import RUNTIME_VERSION

logger = logging.getLogger("sovereign.agent.server")

AGENT_VERSION = RUNTIME_VERSION


def single_native_generation_task(path: str, body: bytes) -> bool:
    """b9960 schema aliases and tokenize_input_prompts cardinality, not validation."""
    try:
        data = json.loads(body)
        if not isinstance(data, dict):
            return False
        count = data.get("n_cmpl", data.get("n", 1))
        if type(count) is not int or count != 1:
            return False
        if path == "chat/completions":
            # The chat formatter supplies one prompt, independent of messages.
            return True
        if path != "completions":
            return False
        prompt = data.get("prompt")
        if isinstance(prompt, str):
            return True
        if isinstance(prompt, list):
            # A numeric/mixed token array is one prompt; otherwise each member
            # is a separate prompt. Invalid inputs are still sent to the engine.
            return len(prompt) == 1 or any(type(item) in (int, float) for item in prompt)
    except (ValueError, TypeError):
        pass
    return False

class RoleStreamingResponse(StreamingResponse):
    """Release admission even if downstream disconnects before iteration starts."""

    def __init__(self, *args, cleanup: Callable[[], Awaitable[None]], **kwargs):
        super().__init__(*args, **kwargs)
        self._cleanup = cleanup

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self._cleanup()



class RoleProcess:
    def __init__(self, name: str, command: list[str], port: int, model_path: str):
        self.name = name
        self.port = port
        self.model_path = model_path
        self.execution_uncertain = False
        # b9960 traces the final four key characters. Keep those public while
        # retaining 256 random bits that never enter native argv or logs.
        self.api_key = secrets.token_urlsafe(32) + "-agent" if name == "generation" else None
        child_env = None
        if self.api_key is not None:
            # b9960 accepts LLAMA_API_KEY without exposing a secret in argv/logs.
            # Extra keys would create ingress outside the agent's admission gate.
            if any(arg.split("=", 1)[0].replace("_", "-") in {"--api-key", "--api-key-file"} for arg in command):
                raise ValueError("native generation authentication is agent-owned")
            child_env = dict(os.environ)
            child_env.pop("LLAMA_ARG_API_KEY_FILE", None)
            child_env["LLAMA_API_KEY"] = self.api_key
            command = [*command, "--metrics"]
        log_dir = Path(os.environ.get("SOVEREIGN_AGENT_LOG_DIR", Path.home() / ".sovereign" / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{name}.llama.log"
        logger.info("starting %s: %s (log: %s)", name, " ".join(command), self.log_path)
        with open(self.log_path, "ab") as log_file:
            self.process = subprocess.Popen(command, stdout=log_file, stderr=log_file, env=child_env)

    def running(self) -> bool:
        return self.process.poll() is None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def healthy(self) -> bool:
        if not self.running():
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                resp = await client.get(f"http://127.0.0.1:{self.port}/health", headers=self.headers())
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def acknowledge_completed_rejection(self) -> None:
        """Fence ONE fully answered task, never repair ambiguous executions.

        Pinned a935fbffe server-context.cpp:3103-3110 releases the rejected
        slot on the engine thread. GET_LORA (4999-5025) is read-only and posts
        at normal priority; its ACK follows that callback and reader.stop's
        cancellation tasks (server-queue.cpp:393-465). Unlike metrics it cannot
        overtake the original task. Batches can retain deferred siblings and
        MUST NOT use this per-request proof.
        """
        if not self.api_key or not self.running() or self.execution_uncertain:
            raise RuntimeError("owned generation execution is uncertain")
        url = f"http://127.0.0.1:{self.port}/lora-adapters"
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 401:
                    raise RuntimeError("native FIFO authentication is unavailable")
            async with client.stream("GET", url, headers=self.headers()) as response:
                if response.status_code != 200 or response.headers.get("content-type", "").split(";", 1)[0] != "application/json":
                    raise RuntimeError("native FIFO acknowledgement is unavailable")
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(data) + len(chunk) > 65536:
                        raise RuntimeError("native FIFO acknowledgement exceeds its bound")
                    data.extend(chunk)
            adapters = json.loads(data)
            if not isinstance(adapters, list):
                raise RuntimeError("native FIFO acknowledgement is invalid")
            fields = {"id", "path", "scale", "task_name", "prompt_prefix"}
            alora_fields = {"alora_invocation_string", "alora_invocation_tokens"}
            for index, adapter in enumerate(adapters):
                if (
                    not isinstance(adapter, dict) or set(adapter) not in (fields, fields | alora_fields)
                    or type(adapter["id"]) is not int or adapter["id"] != index
                    or any(not isinstance(adapter[key], str) for key in ("path", "task_name", "prompt_prefix"))
                    or type(adapter["scale"]) not in (int, float) or not math.isfinite(adapter["scale"])
                ):
                    raise RuntimeError("native FIFO adapter observation is invalid")
                if "alora_invocation_tokens" in adapter and (
                    not isinstance(adapter["alora_invocation_string"], str)
                    or not isinstance(adapter["alora_invocation_tokens"], list)
                    or not adapter["alora_invocation_tokens"]
                    or any(type(token) is not int for token in adapter["alora_invocation_tokens"])
                ):
                    raise RuntimeError("native FIFO adapter token observation is invalid")
        if not self.running() or self.execution_uncertain:
            raise RuntimeError("owned generation execution is uncertain")

    async def wait_idle(self) -> None:
        """b9960 scheduler counts, never /health or HTTP handler count alone.

        Metrics snapshots are high-priority tasks, not FIFO barriers. Admission
        must already be closed and every submitted generation fully answered;
        an uncertain request cannot be repaired by a later zero snapshot.
        """
        if not self.api_key or not self.running() or self.execution_uncertain:
            raise RuntimeError("owned generation execution is uncertain")
        url = f"http://127.0.0.1:{self.port}/metrics"
        names = {"llamacpp:requests_processing", "llamacpp:requests_deferred"}
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            # Verify authentication is enforced, not merely accepted by a public
            # server occupying a configured port. The positive probe then binds
            # the snapshot to this exact child and its unshared ephemeral key.
            async with client.stream("GET", url) as response:
                if response.status_code != 401:
                    raise RuntimeError("native metrics authentication is unavailable")
            while True:
                if not self.running() or self.execution_uncertain:
                    raise RuntimeError("owned generation execution is uncertain")
                async with client.stream("GET", url, headers=self.headers()) as response:
                    response.raise_for_status()
                    if response.headers.get("content-type", "").split(";", 1)[0] != "text/plain":
                        raise RuntimeError("native scheduler metrics are unavailable")
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(data) + len(chunk) > 65536:
                            raise RuntimeError("native scheduler metrics exceed their bound")
                        data.extend(chunk)
                counts = {}
                for line in data.decode("utf-8").splitlines():
                    if not any(line.startswith(name) for name in names):
                        continue
                    fields = line.split()
                    if len(fields) != 2 or fields[0] not in names or fields[0] in counts:
                        raise RuntimeError("native scheduler metrics are ambiguous")
                    value = float(fields[1])
                    if not math.isfinite(value) or value < 0 or not value.is_integer():
                        raise RuntimeError("native scheduler count is invalid")
                    counts[fields[0]] = value
                if counts.keys() != names:
                    raise RuntimeError("native scheduler counts are missing")
                if not self.running() or self.execution_uncertain:
                    raise RuntimeError("owned generation execution is uncertain")
                if not any(counts.values()):
                    return
                await asyncio.sleep(0.05)

    def stop(self) -> None:
        if self.running():
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


class Agent:
    def __init__(self, config: AgentConfig, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        self.token = os.environ.get(config.token_env, "")
        self.roles: dict[str, RoleProcess] = {}
        self.role_lock = asyncio.Lock()
        default_root = self.config_path.parent / "models" if self.config_path else Path.home() / ".sovereign" / "models"
        self.model_root = Path(os.environ.get("SOVEREIGN_AGENT_MODEL_ROOT", default_root)).resolve()
        self.generation_backend: SlimServeBackend | None = None
        self.generation_state = "initializing" if config.slimserve_generation else "healthy"
        self.generation_error: dict | None = None
        self.generation_mapping: dict[str, str] | None = None
        self.available_engines: list[dict] = []
        self.generation_admission_paused = False
        self.generation_requests = 0
        self.generation_idle = asyncio.Event()
        self.generation_idle.set()

    async def discover_engines(self) -> None:
        self.available_engines = await SlimServeBackend().available_engines()
        binary = self.config.llama_server
        if binary == "llama-server":
            binary = shutil.which(binary)
        elif not Path(binary).is_absolute() or Path(binary).name != "llama-server":
            return
        if not binary or not Path(binary).is_file() or not os.access(binary, os.X_OK):
            return
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                binary, "--version", stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            output = await asyncio.wait_for(process.stdout.read(4097), timeout=5)
            if len(output) > 4096:
                return
            await asyncio.wait_for(process.wait(), timeout=5)
            if process.returncode != 0:
                return
            match = re.search(rb"(?m)^version: ([0-9]{1,8}) \(([0-9a-f]{7,40})\)\r?$", output)
            if match is not None:
                self.available_engines.append({
                    "name": "llama.cpp", "version": f"b{match[1].decode()}-{match[2].decode()}",
                    "adapter": "metal-host-agent", "variants": ["metal-arm64"],
                })
        except (OSError, asyncio.TimeoutError):
            return
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

    def role_command(self, name: str) -> list[str]:
        role = self.config.roles[name]
        command = [
            self.config.llama_server,
            "-m", role.model_path,
            *role.args,
            "--host", "127.0.0.1",
            "--port", str(role.port),
        ]
        if role.mmproj_path:
            command += ["--mmproj", role.mmproj_path]
        if role.context_length:
            command += ["-c", str(role.context_length)]
        return command

    def start_role(self, name: str) -> RoleProcess:
        role = self.config.roles[name]
        return RoleProcess(name, self.role_command(name), role.port, role.model_path)

    def start_roles(self) -> None:
        for name in self.config.roles:
            if name == "generation" and self.config.slimserve_generation is not None:
                continue
            self.roles[name] = self.start_role(name)

    async def wait_ready(self, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        pending = set(self.roles)
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                role = self.roles.get(name)
                if role is None:
                    pending.discard(name)
                    continue
                if await role.healthy():
                    logger.info("role %s healthy on :%d", name, role.port)
                    pending.discard(name)
            if pending:
                await asyncio.sleep(2)
        for name in pending:
            logger.error("role %s failed to become healthy", name)

    def stop(self) -> None:
        for role in self.roles.values():
            role.stop()

    def save_config(self) -> None:
        if self.config_path is None:
            raise RuntimeError("agent configuration path is unavailable")
        target = self.config_path
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(yaml.safe_dump(self.config.model_dump(exclude_none=True, exclude_unset=True), sort_keys=False))
        temporary.chmod(0o600)
        temporary.replace(target)

    def resolve_model(self, artifact: str, expected_sha256: str) -> Path:
        relative = Path(artifact)
        if relative.is_absolute() or ".." in relative.parts or relative.name == "":
            raise ValueError("artifact must be a relative path within the managed model directory")
        model = (self.model_root / relative).resolve(strict=True)
        if not model.is_relative_to(self.model_root) or not model.is_file():
            raise ValueError("artifact must resolve to a model file within the managed model directory")
        if model.suffix.lower() != ".gguf":
            raise ValueError("Metal embedding artifacts must be GGUF files")
        digest = hashlib.sha256()
        with model.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256.lower():
            raise ValueError("artifact checksum does not match sha256")
        return model

    async def wait_role_ready(self, role: RoleProcess, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await role.healthy():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"http://127.0.0.1:{role.port}/v1/embeddings",
                        json={"model": "embedding", "input": "sovereign embedding probe"},
                    )
                if response.status_code != 200:
                    raise RuntimeError(f"embedding probe failed: {response.status_code}: {response.text[:300]}")
                data = response.json().get("data") or []
                if not data or not data[0].get("embedding"):
                    raise RuntimeError("embedding probe returned no vector")
                return
            if not role.running():
                break
            await asyncio.sleep(1)
        raise RuntimeError("embedding role did not become healthy before timeout")

    def translate_generation(self, config: RuntimeConfig) -> RuntimeConfig:
        """Map only the fixed private staging layout into installer-owned storage."""
        wire = config.model_dump(exclude_unset=True)
        if (
            set(wire) - {"schema_version", "runtime", "roles"}
            or set(wire.get("runtime", {})) != {"profile"}
            or set(wire.get("roles", {})) != {"generation"}
        ):
            raise ValueError("host input permits only Runtime schema, Metal profile, and generation role")
        role = config.roles.generation
        if (
            config.runtime.profile != "metal-arm64"
            or not role.enabled
            or role.engine != "slimserve"
            or role.slimserve is None
            or role.slimserve.variant != "metal"
        ):
            raise ValueError("host generation requires a SlimServe Metal configuration")
        for name, other in config.roles.items():
            if name != "generation" and other.enabled:
                raise ValueError("host generation configuration cannot configure another role")
        model = role.model or ""
        match = re.fullmatch(
            r"/models/staged/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})/"
            r"([0-9a-f]{64})/([A-Za-z0-9][A-Za-z0-9_.-]{0,127})/model",
            model,
        )
        if match is None or match[3] != role.engine_profile_id:
            raise ValueError("generation model must use its exact managed staged profile directory")
        relative = Path(model).relative_to("/models")
        current = self.model_root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("managed generation paths cannot contain symlinks")
        resolved = current.resolve(strict=True)
        if resolved != current or not resolved.is_relative_to(self.model_root) or not resolved.is_dir():
            raise ValueError("generation model must be a managed local directory")
        translated = config.model_copy(deep=True)
        translated.roles.generation.model = str(resolved)
        return translated

    def generation_callback(self, backend: SlimServeBackend, state: str) -> None:
        if self.generation_backend is backend:
            self.generation_state = state
            if state in {"runtime_error", "configuration_error", "degraded"}:
                self.generation_error = {
                    "code": backend.role_info("generation").error_code or "ENGINE_DEAD",
                    "message": "native generation is not ready",
                    "rollback_verified": False,
                }

    async def verify_generation(self, backend: SlimServeBackend) -> None:
        if backend.role_info("generation").status != "healthy":
            raise BackendStartError("MODEL_LOAD_FAILED", "native generation did not become healthy")
        client = backend.role_client("generation")
        if client is None:
            raise BackendStartError("MODEL_LOAD_FAILED", "native generation client is unavailable")
        response = await client.get("/health")
        response.raise_for_status()

    async def wait_generation_ready(self, role: RoleProcess, timeout: float = 120) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await role.healthy():
                return
            if not role.running():
                break
            await asyncio.sleep(1)
        raise RuntimeError("previous generation did not become healthy before timeout")

    async def quiesce_generation_backend(self, backend: SlimServeBackend) -> None:
        # An empty HTTP handler count does not retire a disconnected scheduler
        # request. Keep this lifetime owned until its scheduler acknowledges idle.
        await backend.quiesce()
        if backend.generation_paused is not True:
            raise BackendStartError("ENGINE_QUIESCE_FAILED", "native scheduler idle is unverified")

    async def configure_generation(self, request: RuntimeConfig, *, persist: bool = True) -> dict:
        async with self.role_lock:
            translated = self.translate_generation(request)
            candidate = SlimServeBackend()
            try:
                await candidate.validate(translated)
            finally:
                await candidate.shutdown()
            if persist and self.config_path is None:
                raise ValueError("agent configuration path is unavailable")
            previous_config = self.config.slimserve_generation
            previous_backend = self.generation_backend
            previous_process = self.roles.get("generation")
            previous_mapping = self.generation_mapping
            self.generation_state = "loading"
            self.generation_error = None
            mutated = False
            try:
                self.generation_admission_paused = True
                await asyncio.wait_for(self.generation_idle.wait(), timeout=600)
                if previous_backend is not None:
                    await self.quiesce_generation_backend(previous_backend)
                if previous_process is not None:
                    await asyncio.wait_for(previous_process.wait_idle(), timeout=600)
                if previous_backend is not None:
                    await previous_backend.shutdown()
                if previous_process is not None:
                    previous_process.stop()
                    self.roles.pop("generation", None)
                mutated = True
                self.generation_backend = candidate
                self.generation_mapping = {
                    "runtime": request.roles.generation.model,
                    "host": translated.roles.generation.model,
                }
                await candidate.start(translated, lambda state: self.generation_callback(candidate, state))
                await self.verify_generation(candidate)
                self.config.slimserve_generation = request.model_copy(deep=True)
                if persist:
                    self.save_config()
                self.generation_state = "healthy"
                self.generation_admission_paused = False
                return {"status": "healthy", "role": "generation"}
            except (Exception, asyncio.CancelledError) as exc:
                self.config.slimserve_generation = previous_config
                rollback_verified = False
                rollback_error = None
                try:
                    if mutated:
                        # A shutdown failure must never launch a second generation tree.
                        await candidate.shutdown()
                        self.generation_backend = None
                        self.generation_mapping = previous_mapping
                        if previous_backend is not None and previous_config is not None:
                            self.generation_backend = previous_backend
                            restored = self.translate_generation(previous_config)
                            await previous_backend.start(
                                restored, lambda state: self.generation_callback(previous_backend, state)
                            )
                            await self.verify_generation(previous_backend)
                        elif previous_process is not None:
                            restored_process = self.start_role("generation")
                            self.roles["generation"] = restored_process
                            await self.wait_generation_ready(restored_process)
                        if persist:
                            self.save_config()
                        rollback_verified = previous_backend is not None or previous_process is not None
                except Exception as rollback_exc:
                    rollback_error = str(rollback_exc)
                self.generation_state = "configuration_error"
                self.generation_error = {
                    "code": getattr(exc, "code", "MODEL_LOAD_FAILED"),
                    "message": str(exc),
                    "rolled_back": rollback_verified,
                    "rollback_verified": rollback_verified,
                }
                if rollback_error is not None:
                    self.generation_error["rollback_error"] = rollback_error
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return {"status": "unhealthy", "role": "generation", **self.generation_error}

    async def resume_generation(self) -> dict:
        if self.config.slimserve_generation is None:
            return {"status": "disabled", "role": "generation"}
        try:
            return await self.configure_generation(self.config.slimserve_generation, persist=False)
        except Exception as exc:
            self.generation_state = "configuration_error"
            self.generation_error = {
                "code": getattr(exc, "code", "CONFIG_INVALID"),
                "message": str(exc),
                "rollback_verified": False,
            }
            return {"status": "unhealthy", "role": "generation", **self.generation_error}

    async def stop_generation(self, expected: RuntimeConfig | None = None) -> bool:
        async with self.role_lock:
            if expected is not None and expected != self.config.slimserve_generation:
                return False
            if self.generation_backend is not None:
                self.generation_admission_paused = True
                await asyncio.wait_for(self.generation_idle.wait(), timeout=600)
                await self.quiesce_generation_backend(self.generation_backend)
                await self.generation_backend.shutdown()
                self.generation_backend = None
            self.generation_state = "stopped"
            return True

    async def shutdown(self) -> None:
        try:
            await self.stop_generation()
        finally:
            self.stop()

    async def restore_llama_generation(self) -> dict:
        """Explicit cutback to installer-owned configuration, never caller argv."""
        async with self.role_lock:
            role = self.config.roles.get("generation")
            if role is None:
                raise ValueError("no installer-owned llama generation is configured")
            for path in (role.model_path, role.mmproj_path):
                if path is not None and (Path(path).is_symlink() or not Path(path).is_file()):
                    raise ValueError("previous llama generation artifact is unavailable")
            if self.config_path is None:
                raise ValueError("agent configuration path is unavailable")
            previous_backend = self.generation_backend
            previous_config = self.config.slimserve_generation
            previous_mapping = self.generation_mapping
            existing = self.roles.get("generation")
            if previous_backend is None and previous_config is None and existing is not None:
                self.generation_admission_paused = True
                try:
                    await asyncio.wait_for(self.generation_idle.wait(), timeout=600)
                    await asyncio.wait_for(existing.wait_idle(), timeout=600)
                except Exception as exc:
                    raise BackendStartError("ENGINE_QUIESCE_FAILED", "native generation idle is unverified") from exc
                await self.wait_generation_ready(existing)
                self.generation_state = "healthy"
                self.generation_error = None
                self.generation_admission_paused = False
                return {"status": "healthy", "role": "generation"}
            candidate = None
            self.generation_state = "loading"
            try:
                self.generation_admission_paused = True
                await asyncio.wait_for(self.generation_idle.wait(), timeout=600)
                if previous_backend is not None:
                    await self.quiesce_generation_backend(previous_backend)
                if existing is not None:
                    await asyncio.wait_for(existing.wait_idle(), timeout=600)
                if previous_backend is not None:
                    await previous_backend.shutdown()
                    self.generation_backend = None
                if existing is not None:
                    existing.stop()
                    self.roles.pop("generation", None)
                candidate = self.start_role("generation")
                self.roles["generation"] = candidate
                await self.wait_generation_ready(candidate)
                self.config.slimserve_generation = None
                self.save_config()
                self.generation_mapping = None
                self.generation_state = "healthy"
                self.generation_error = None
                self.generation_admission_paused = False
                return {"status": "healthy", "role": "generation"}
            except Exception as exc:
                self.config.slimserve_generation = previous_config
                verified = False
                rollback_error = None
                try:
                    if candidate is not None:
                        candidate.stop()
                        self.roles.pop("generation", None)
                    if previous_backend is not None and self.generation_backend is None and previous_config is not None:
                        self.generation_backend = previous_backend
                        self.generation_mapping = previous_mapping
                        await previous_backend.start(
                            self.translate_generation(previous_config),
                            lambda state: self.generation_callback(previous_backend, state),
                        )
                        await self.verify_generation(previous_backend)
                        self.save_config()
                        verified = True
                except Exception as rollback_exc:
                    rollback_error = str(rollback_exc)
                self.generation_state = "configuration_error"
                self.generation_error = {
                    "code": getattr(exc, "code", "MODEL_LOAD_FAILED"),
                    "message": str(exc), "rolled_back": verified, "rollback_verified": verified,
                }
                if rollback_error is not None:
                    self.generation_error["rollback_error"] = rollback_error
                return {"status": "unhealthy", "role": "generation", **self.generation_error}


class EmbeddingRoleRequest(BaseModel):
    """Constrained host-agent input; arbitrary llama.cpp flags are forbidden."""

    model_config = ConfigDict(extra="forbid")

    artifact: str
    revision: str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    pooling: str = Field(default="mean", pattern=r"^(mean|last|cls)$")
    normalization: str = Field(default="l2", pattern=r"^(l2|none)$")
    context_length: int = Field(default=2048, ge=128, le=131072)


def build_app(agent: Agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent.start_roles()
        await agent.discover_engines()
        task = asyncio.create_task(agent.wait_ready())
        generation_task = asyncio.create_task(agent.resume_generation())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            # Do not interrupt a generation swap between stop and verified rollback.
            await generation_task
            await agent.shutdown()

    app = FastAPI(title="Sovereign Runtime Agent", lifespan=lifespan)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if not agent.token or request.headers.get("Authorization") != f"Bearer {agent.token}":
            return JSONResponse(status_code=401, content={"error": "invalid agent token"})
        return await call_next(request)

    @app.get("/agent/manifest")
    async def manifest():
        roles = {}
        for name, role in list(agent.roles.items()):
            configured = agent.config.roles.get(name)
            if configured is None:
                continue
            healthy = await role.healthy()
            roles[name] = {
                "status": "healthy" if healthy else ("loading" if role.running() else "unhealthy"),
                "model": Path(role.model_path).name,
                "context_length": configured.context_length,
                "revision": configured.revision,
            }
        backend = agent.generation_backend
        native = backend is not None or agent.config.slimserve_generation is not None
        observation = backend.observation() if backend is not None else {}
        if backend is not None:
            roles["generation"] = asdict(backend.role_info("generation"))
            roles["generation"]["model"] = roles["generation"]["engine_model"]
        elif native:
            roles["generation"] = {"status": "unhealthy"}
        if agent.generation_state != "healthy" and "generation" in roles:
            roles["generation"]["status"] = (
                "loading" if agent.generation_state in {"initializing", "loading", "compiling"}
                else "unhealthy"
            )
        return {
            "agent_version": AGENT_VERSION,
            "engine": "slimserve" if observation.get("engine", {}).get("name") == "slimserve" else ("unavailable" if native else "llama.cpp"),
            "configured_engine": "slimserve" if agent.config.slimserve_generation is not None else "llama.cpp",
            "backend": "metal",
            "roles": roles,
            "state": agent.generation_state,
            "errors": [agent.generation_error] if agent.generation_error else [],
            "observation": observation,
            # This is current ingress state, not cached model health or idle proof.
            "generation_paused": agent.generation_admission_paused or (
                backend is not None and backend.generation_paused
            ),
            "model_mapping": agent.generation_mapping if backend is not None else None,
            "available_engines": agent.available_engines,
        }

    @app.put("/agent/admin/roles/generation")
    async def configure_generation(request: RuntimeConfig):
        try:
            result = await agent.configure_generation(request)
        except (OSError, ValueError, BackendStartError) as exc:
            return JSONResponse(status_code=422, content={"error": str(exc), "mutated": False})
        return JSONResponse(status_code=200 if result["status"] == "healthy" else 422, content=result)

    @app.delete("/agent/admin/roles/generation")
    async def stop_generation(request: RuntimeConfig):
        try:
            if not await agent.stop_generation(request):
                return JSONResponse(status_code=409, content={"error": "generation configuration changed"})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc), "status": "unhealthy"})
        return {"status": "stopped", "role": "generation"}

    async def native_generation_idle():
        role = agent.roles.get("generation")
        if (
            agent.generation_backend is not None or agent.config.slimserve_generation is not None
            or agent.generation_state != "healthy" or not isinstance(role, RoleProcess)
            or role.name != "generation" or not agent.generation_admission_paused
        ):
            raise RuntimeError("owned native generation is unavailable")
        await agent.generation_idle.wait()
        await role.wait_idle()
        if agent.roles.get("generation") is not role or agent.generation_requests:
            raise RuntimeError("native generation ownership changed")

    @app.post("/agent/admin/roles/generation/repair")
    async def repair_generation(request: Request):
        if await request.body():
            return JSONResponse(status_code=422, content={"error": "repair accepts no input"})
        result = await agent.resume_generation()
        return JSONResponse(status_code=200 if result["status"] == "healthy" else 422, content=result)

    @app.post("/agent/admin/roles/generation/llama")
    async def restore_llama(request: Request):
        if await request.body():
            return JSONResponse(status_code=422, content={"error": "engine cutback accepts no input"})
        try:
            result = await agent.restore_llama_generation()
        except (OSError, ValueError, BackendStartError) as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return JSONResponse(status_code=200 if result["status"] == "healthy" else 422, content=result)

    @app.post("/agent/admin/roles/generation/quiesce")
    async def quiesce_generation(request: Request):
        async for chunk in request.stream():
            if chunk:
                return JSONResponse(status_code=422, content={"error": "quiesce accepts no input"})
        async with agent.role_lock:
            agent.generation_admission_paused = True
            backend = agent.generation_backend
            expected = request.headers.get("X-Sovereign-Engine")
            if expected is not None and (
                expected != "llama.cpp" or backend is not None or agent.config.slimserve_generation is not None
            ):
                return JSONResponse(status_code=409, content={"error": "generation engine changed"})
            try:
                if backend is None:
                    await asyncio.wait_for(native_generation_idle(), timeout=600)
                else:
                    # Handler drain never replaces the SlimServe scheduler ACK.
                    await asyncio.wait_for(agent.generation_idle.wait(), timeout=600)
                    await agent.quiesce_generation_backend(backend)
            except Exception:
                return JSONResponse(status_code=503, content={"error": "engine idle acknowledgement failed"})
            return {"paused": True, "idle": True}

    @app.post("/agent/admin/roles/generation/resume")
    async def resume_requests(request: Request):
        async for chunk in request.stream():
            if chunk:
                return JSONResponse(status_code=422, content={"error": "resume accepts no input"})
        async with agent.role_lock:
            agent.generation_admission_paused = True
            backend = agent.generation_backend
            expected = request.headers.get("X-Sovereign-Engine")
            if expected is not None and (
                expected != "llama.cpp" or backend is not None or agent.config.slimserve_generation is not None
            ):
                return JSONResponse(status_code=409, content={"error": "generation engine changed"})
            try:
                if backend is None:
                    await asyncio.wait_for(native_generation_idle(), timeout=600)
                else:
                    await backend.resume()
                    if backend.generation_paused is not False:
                        raise RuntimeError("engine resume was not acknowledged")
                agent.generation_admission_paused = False
            except Exception:
                return JSONResponse(status_code=503, content={"error": "engine resume acknowledgement failed"})
            return {"paused": False}

    @app.put("/agent/admin/roles/embedding")
    async def configure_embedding(request: EmbeddingRoleRequest):
        if not re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", request.revision):
            return JSONResponse(status_code=422, content={"error": "revision must be an immutable git commit"})
        try:
            model = agent.resolve_model(request.artifact, request.sha256)
        except (OSError, ValueError) as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})

        from lazarus.agent.config import AgentRole

        async with agent.role_lock:
            previous_config = agent.config.roles.get("embedding")
            previous_process = agent.roles.get("embedding")
            port = previous_config.port if previous_config else 9102
            candidate_config = AgentRole(
                model_path=str(model),
                revision=request.revision.lower(),
                port=port,
                context_length=request.context_length,
                args=[
                    "--embedding", "--pooling", request.pooling,
                    "--embd-normalize", "2" if request.normalization == "l2" else "-1",
                ],
            )
            if previous_process:
                previous_process.stop()
            agent.config.roles["embedding"] = candidate_config
            candidate = None
            try:
                candidate = agent.start_role("embedding")
                await agent.wait_role_ready(candidate)
                agent.save_config()
            except Exception as exc:
                verified = False
                rollback_error = None
                try:
                    if candidate is not None:
                        candidate.stop()
                    if previous_config is None:
                        agent.config.roles.pop("embedding", None)
                        agent.roles.pop("embedding", None)
                    else:
                        agent.config.roles["embedding"] = previous_config
                        agent.roles["embedding"] = agent.start_role("embedding")
                        await agent.wait_role_ready(agent.roles["embedding"])
                    agent.save_config()
                    verified = True
                except Exception as rollback_exc:
                    rollback_error = str(rollback_exc)
                return JSONResponse(status_code=422, content={
                    "error": str(exc), "rolled_back": verified, "rollback_verified": verified,
                    "rollback_error": rollback_error,
                })
            agent.roles["embedding"] = candidate
            return {
                "status": "healthy",
                "role": "embedding",
                "model": model.name,
                "revision": candidate_config.revision,
            }

    @app.delete("/agent/admin/roles/embedding")
    async def remove_embedding():
        async with agent.role_lock:
            previous_config = agent.config.roles.get("embedding")
            if previous_config is None:
                return {"status": "disabled", "role": "embedding"}
            previous_process = agent.roles.pop("embedding", None)
            if previous_process:
                previous_process.stop()
            agent.config.roles.pop("embedding", None)
            try:
                agent.save_config()
            except Exception as exc:
                agent.config.roles["embedding"] = previous_config
                verified = False
                rollback_error = None
                try:
                    agent.roles["embedding"] = agent.start_role("embedding")
                    await agent.wait_role_ready(agent.roles["embedding"])
                    agent.save_config()
                    verified = True
                except Exception as rollback_exc:
                    rollback_error = str(rollback_exc)
                return JSONResponse(status_code=500, content={
                    "error": str(exc), "rolled_back": verified, "rollback_verified": verified,
                    "rollback_error": rollback_error,
                })
            return {"status": "disabled", "role": "embedding"}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        role_name = request.headers.get("X-Sovereign-Role", "")
        role = agent.roles.get(role_name)
        native = agent.generation_backend if role_name == "generation" else None
        if role_name == "generation" and agent.generation_state != "healthy":
            return JSONResponse(status_code=503, content={"error": "generation is not ready"})
        if role is None and native is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"unknown role {role_name!r} (set X-Sovereign-Role)"},
            )
        allowed = {"generation": {"chat/completions", "completions", "models"}, "embedding": {"embeddings", "models"}}
        if path not in allowed.get(role_name, set()):
            return JSONResponse(status_code=404, content={"error": "unsupported role endpoint"})
        if role_name == "generation" and (agent.generation_admission_paused or (native is not None and native.generation_paused)):
            return JSONResponse(status_code=503, content={"error": "generation admission is paused"})
        body = await request.body()
        if role_name == "generation" and (agent.generation_admission_paused or (native is not None and native.generation_paused)):
            return JSONResponse(status_code=503, content={"error": "generation admission is paused"})
        if role_name == "generation" and (
            role is not agent.roles.get("generation") or native is not agent.generation_backend
        ):
            return JSONResponse(status_code=503, content={"error": "generation ownership changed"})
        owned_client = native is None
        client = native.role_client("generation") if native is not None else httpx.AsyncClient(timeout=600.0, trust_env=False)
        if client is None or (native is not None and native.role_info("generation").status != "healthy"):
            return JSONResponse(status_code=503, content={"error": "generation is not ready"})
        target = f"/v1/{path}" if native is not None else f"http://127.0.0.1:{role.port}/v1/{path}"
        upstream = client.build_request(
            request.method,
            target,
            content=body,
            headers={
                "Content-Type": request.headers.get("Content-Type", "application/json"),
                **(role.headers() if isinstance(role, RoleProcess) else {}),
            },
        )
        counted = role_name == "generation"
        if counted:
            agent.generation_requests += 1
            agent.generation_idle.clear()

        def release_generation():
            nonlocal counted
            if counted:
                counted = False
                agent.generation_requests -= 1
                if agent.generation_requests == 0:
                    agent.generation_idle.set()

        def uncertain_generation():
            if role_name == "generation" and native is None:
                role.execution_uncertain = True
                agent.generation_admission_paused = True

        try:
            resp = await client.send(upstream, stream=True)
        except BaseException as exc:
            uncertain_generation()
            release_generation()
            if owned_client:
                await client.aclose()
            if isinstance(exc, httpx.HTTPError):
                return JSONResponse(status_code=503, content={"error": "role engine unavailable"})
            raise

        complete = False
        native_stream = role_name == "generation" and native is None and (
            resp.headers.get("content-type", "").split(";", 1)[0] == "text/event-stream"
        )
        rejection = (
            role_name == "generation" and native is None and resp.status_code == 400
            and resp.headers.get("content-type", "").split(";", 1)[0] == "application/json"
            and single_native_generation_task(path, body)
        )

        async def relay():
            nonlocal complete
            terminal = False
            received = False
            suffix = b""
            rejected_body = bytearray() if rejection else None
            async for chunk in resp.aiter_raw():
                received = received or bool(chunk)
                if rejected_body is not None:
                    if len(rejected_body) + len(chunk) <= 65536:
                        rejected_body.extend(chunk)
                    else:
                        rejected_body = None
                if native_stream:
                    terminal = terminal or b"data: [DONE]\n\n" in chunk or b"data: [DONE]\n\n" in (suffix + chunk[:32])
                    suffix = (suffix + chunk)[-32:] if len(chunk) < 32 else chunk[-32:]
                yield chunk
            complete = resp.is_success and received and (not native_stream or terminal)
            if rejected_body:
                try:
                    payload = json.loads(rejected_body)
                    error = payload.get("error") if isinstance(payload, dict) else None
                    if (
                        not isinstance(payload, dict) or set(payload) != {"error"}
                        or not isinstance(error, dict)
                        or type(error.get("code")) is not int or error["code"] != 400
                        or not isinstance(error.get("message"), str)
                    ):
                        return
                    if error.get("type") == "invalid_request_error":
                        if set(error) != {"code", "message", "type"}:
                            return
                        # All reachable b9960 generation origins either fail
                        # before enqueue, or return false before slot assignment
                        # (server-context.cpp:1715-1800,2371-2374,4112-4116).
                    elif error.get("type") == "exceed_context_size_error":
                        if (
                            set(error) != {"code", "message", "type", "n_prompt_tokens", "n_ctx"}
                            or type(error["n_prompt_tokens"]) is not int or type(error["n_ctx"]) is not int
                            or not 0 < error["n_ctx"] <= error["n_prompt_tokens"]
                        ):
                            return
                    else:
                        return
                    if agent.roles.get("generation") is not role or agent.generation_backend is not None:
                        return
                    await asyncio.wait_for(role.acknowledge_completed_rejection(), timeout=5.0)
                    complete = agent.roles.get("generation") is role and agent.generation_backend is None
                except (httpx.HTTPError, ValueError, TypeError, RuntimeError, OverflowError, asyncio.TimeoutError):
                    # A well-formed rejection alone is not a scheduler ACK.
                    pass

        async def cleanup():
            if not complete:
                uncertain_generation()
            try:
                await resp.aclose()
                if owned_client:
                    await client.aclose()
            finally:
                release_generation()

        return RoleStreamingResponse(
            relay(),
            cleanup=cleanup,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    return app


def main() -> int:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()

    config = load_agent_config(arguments.config)
    token = os.environ.get(config.token_env, "")
    if not token:
        print(f"error: {config.token_env} is required (the agent fails closed)", file=sys.stderr)
        return 1

    agent = Agent(config, arguments.config)
    uvicorn.run(build_app(agent), host=config.listen, port=config.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
