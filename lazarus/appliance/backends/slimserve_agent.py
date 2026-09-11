"""Private Runtime-to-host adapter for the isolated native SlimServe generation tree."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import fields
from pathlib import PurePosixPath

import httpx

from lazarus.appliance.backends.agent import HOST_GENERATION_TIMEOUT, AgentBackend
from lazarus.appliance.backends.base import BackendStartError, RoleInfo
from lazarus.appliance.backends.slimserve import validate_observation
from lazarus.appliance.config import RuntimeConfig


class SlimServeAgentBackend(AgentBackend):
    engine_name = "slimserve"
    adapter_id = "slimserve-runtime"

    def __init__(self) -> None:
        super().__init__()
        self.backend_id = "metal"
        self._config: RuntimeConfig | None = None
        self._wire: dict | None = None
        self._observation: dict = {}
        self._monitor_task: asyncio.Task | None = None
        self._owns_generation = False
        self.generation_paused = False

    def _accept_manifest(self, manifest: dict) -> None:
        if self._config is None:
            raise BackendStartError("CONFIG_INVALID", "native generation configuration is missing")
        self._require_managed_role_engines(manifest)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("roles"), dict):
            raise BackendStartError("CONFIG_INVALID", "native manifest has no typed role observations")
        role = (manifest.get("roles") or {}).get("generation") or {}
        if not isinstance(role, dict):
            raise BackendStartError("CONFIG_INVALID", "native generation role observation is invalid")
        if (
            manifest.get("engine") != "slimserve"
            or manifest.get("backend") != "metal"
            or manifest.get("state") != "healthy"
            or manifest.get("errors")
            or role.get("status") != "healthy"
        ):
            raise BackendStartError("ENGINE_DEAD", "host SlimServe generation is not ready")
        self._observe_generation_admission(manifest)
        payload = manifest.get("observation")
        mapping = manifest.get("model_mapping")
        if not isinstance(payload, dict) or not isinstance(mapping, dict):
            raise BackendStartError("CONFIG_INVALID", "native generation evidence is missing")
        runtime_path = self._config.roles.generation.model
        host_path = mapping.get("host")
        if (
            mapping.get("runtime") != runtime_path
            or not isinstance(host_path, str)
            or not host_path.startswith("/")
            or str(PurePosixPath(host_path)) != host_path
            or ".." in PurePosixPath(host_path).parts
            or not isinstance(runtime_path, str)
            or not runtime_path.startswith("/models/staged/")
            or not host_path.endswith(runtime_path.removeprefix("/models"))
        ):
            raise BackendStartError("CONFIG_INVALID", "native generation managed mapping does not match")
        # Translate only an exact directory or a sealed model member. Never replace
        # arbitrary observed prefixes or aliases with the requested model identity.
        allowed = {host_path: runtime_path}
        spec = self._config.roles.generation.slimserve
        for artifact in spec.artifacts:
            if artifact.role == "model":
                for item in artifact.files:
                    if item.file.lower().endswith(".gguf"):
                        allowed[f"{host_path}/{item.file}"] = f"{runtime_path}/{item.file}"
        observed_model = payload.get("engine_model")
        if not isinstance(observed_model, str) or observed_model not in allowed:
            raise BackendStartError("CONFIG_INVALID", "native generation observed model is outside its managed input")
        if role.get("engine_model") != observed_model:
            raise BackendStartError("CONFIG_INVALID", "native role and engine model observations disagree")
        normalized = {**payload, "engine_model": allowed[observed_model]}
        accepted = validate_observation(self._config, normalized)
        self._observation = accepted
        self._agent_manifest = manifest
        info = {field.name: accepted[field.name] for field in fields(RoleInfo) if field.name in accepted}
        info["status"] = "healthy"
        self._roles["generation"] = RoleInfo(**info)
        self._observe_native_roles(manifest)

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        if config.runtime.profile != "metal-arm64" or config.roles.generation.engine != "slimserve":
            raise BackendStartError("CONFIG_INVALID", "native SlimServe requires the Metal profile")
        if not self.token:
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "the host agent token is missing")
        self._config = config
        self._set_managed_binding(config)
        await self._verify_managed_binding_before_control()
        # Control's generation role is the only mutation authority carried over
        # this boundary. The paired managed identity remains in the private
        # document so an authenticated but wrong agent cannot accept it.
        runtime = {"profile": config.runtime.profile}
        if config.runtime.runtime_instance_id is not None:
            runtime["runtime_instance_id"] = config.runtime.runtime_instance_id
            runtime["deployment_id"] = config.runtime.deployment_id
        host_config = RuntimeConfig.model_validate({
            "schema_version": config.schema_version,
            "runtime": runtime,
            "roles": {"generation": config.roles.generation.model_dump(exclude_unset=True)},
        })
        self._wire = host_config.model_dump(exclude_unset=True)
        on_state("loading")
        try:
            async with httpx.AsyncClient(timeout=HOST_GENERATION_TIMEOUT, trust_env=False) as client:
                response = await client.put(
                    f"{self.url}/agent/admin/roles/generation",
                    headers=self._headers(), json=self._wire,
                )
                if response.status_code != 200:
                    raise BackendStartError(
                        "MODEL_LOAD_FAILED", f"host generation rejected configuration: {response.text[:500]}"
                    )
                self._owns_generation = True
            await super().start(config, on_state)
            self._accept_manifest(self._agent_manifest)
        except httpx.HTTPError as exc:
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "host generation management request failed") from exc
        except (ValueError, TypeError, AttributeError) as exc:
            raise BackendStartError("CONFIG_INVALID", "host generation returned invalid observations") from exc
        self._monitor_task = asyncio.create_task(self._monitor(on_state))

    async def _monitor(self, on_state: Callable[[str], None]) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                await asyncio.sleep(2.0)
                try:
                    response = await client.get(f"{self.url}/agent/manifest", headers=self._headers())
                    response.raise_for_status()
                    self._accept_manifest(response.json())
                    if any(info.status == "unhealthy" for info in self._roles.values()):
                        on_state("runtime_error")
                        return
                except (httpx.HTTPError, ValueError, TypeError, BackendStartError) as exc:
                    self.generation_paused = True
                    self._observation = {}
                    self._roles["generation"] = RoleInfo(
                        status="unhealthy", error_code=getattr(exc, "code", "HOST_AGENT_UNREACHABLE")
                    )
                    on_state("runtime_error")
                    return

    async def shutdown(self) -> None:
        task, self._monitor_task = self._monitor_task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        try:
            if self._owns_generation:
                await self._verify_managed_binding_before_control()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.request(
                        "DELETE", f"{self.url}/agent/admin/roles/generation",
                        headers=self._headers(), json=self._wire,
                    )
                    # A newer Control configuration owns the host tree now.
                    if response.status_code != 409:
                        response.raise_for_status()
                self._owns_generation = False
        finally:
            await super().shutdown()
            self._observation = {}
            self._roles["generation"] = RoleInfo(status="disabled")


    async def quiesce(self) -> None:
        await self._verify_managed_binding_before_control()
        self.generation_paused = True
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(
                f"{self.url}/agent/admin/roles/generation/quiesce", headers=self._headers()
            )
            response.raise_for_status()
            if response.json() != {"paused": True, "idle": True}:
                raise RuntimeError("host engine did not acknowledge scheduler idle")

    async def resume(self) -> None:
        await self._verify_managed_binding_before_control()
        self.generation_paused = True
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.url}/agent/admin/roles/generation/resume", headers=self._headers()
            )
            response.raise_for_status()
            if response.json() != {"paused": False}:
                raise RuntimeError("host engine did not acknowledge scheduler resume")
        self.generation_paused = False

    def engine_version(self) -> str | None:
        return (self._observation.get("engine") or {}).get("version")

    def accelerator(self) -> dict:
        return self._observation.get("accelerator") or {}

    def observation(self) -> dict:
        return dict(self._observation)
