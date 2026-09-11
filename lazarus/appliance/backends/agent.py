"""Host inference agent backend — Metal Phase 2 (design.md §2.6).

Docker Desktop exposes no GPU/Metal to containers, so the metal-arm64
runtime keeps the container contract while inference runs host-side in the
Sovereign agent (lazarus.agent: a supervised llama.cpp deployment). This
backend is the container half: it discovers roles from the agent's manifest
and forwards role traffic to the agent's single private port, routed by the
X-Sovereign-Role header.

Failure semantics (§3.2/§3.3): agent unreachable or incompatible →
HOST_AGENT_UNREACHABLE, state configuration_error, liveness stays green —
never a crash loop.

Env:
  SOVEREIGN_AGENT_URL         default http://host.docker.internal:9100
  SOVEREIGN_AGENT_TOKEN       bearer token written by the installer
  SOVEREIGN_AGENT_BACKEND_ID  manifest backend id (default metal)
  SOVEREIGN_AGENT_WAIT        seconds to wait for the agent (default 60)
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable

import httpx

from lazarus.agent.config import valid_native_model_identity
from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.backends.roleclient import RoleClientMixin
from lazarus.appliance.config import RuntimeConfig

logger = logging.getLogger("sovereign.agent")

# Fixed artifact-verification and engine-load budget; never a caller option.
HOST_GENERATION_TIMEOUT = 9000.0


class AgentBackend(RoleClientMixin, EngineBackend):
    engine_name = "llama.cpp"
    adapter_id = "metal-host-agent"

    def __init__(self) -> None:
        self.backend_id = os.environ.get("SOVEREIGN_AGENT_BACKEND_ID", "metal")
        self.url = os.environ.get("SOVEREIGN_AGENT_URL", "http://host.docker.internal:9100")
        self.token = os.environ.get("SOVEREIGN_AGENT_TOKEN", "")
        self._roles: dict[str, RoleInfo] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._agent_manifest: dict = {}
        self.generation_paused = False
        self._managed_binding: tuple[str, str, str] | None = None

    def _headers(self, role: str | None = None) -> dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if role:
            headers["X-Sovereign-Role"] = role
        return headers

    def _set_managed_binding(self, config: RuntimeConfig) -> None:
        runtime = config.runtime
        if runtime.runtime_instance_id is None:
            self._managed_binding = None
            return
        self._managed_binding = (
            runtime.runtime_instance_id, runtime.deployment_id, runtime.profile,
        )

    def _validate_managed_binding(self, manifest: object) -> None:
        if self._managed_binding is None:
            return
        instance_id, deployment_id, profile = self._managed_binding
        if not isinstance(manifest, dict) or (
            manifest.get("runtime_instance_id"), manifest.get("deployment_id"), manifest.get("profile")
        ) != (instance_id, deployment_id, profile):
            self.generation_paused = True
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "managed host agent identity does not match Runtime binding")

    def _require_managed_role_engines(self, manifest: object) -> None:
        if self._managed_binding is None:
            return
        self._validate_managed_binding(manifest)
        roles = manifest.get("roles") if isinstance(manifest, dict) else None
        generation = roles.get("generation") if isinstance(roles, dict) else None
        embedding = roles.get("embedding") if isinstance(roles, dict) else None
        if self.engine_name == "slimserve":
            valid_generation = isinstance(generation, dict) and (
                manifest.get("engine") == "slimserve" and generation.get("engine") == "slimserve"
            )
        else:
            valid_generation = isinstance(generation, dict) and generation.get("engine") == self.engine_name
        if not valid_generation or not isinstance(embedding, dict) or embedding.get("engine") != "embeddinggemma":
            self.generation_paused = True
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "managed host agent role engines do not match Runtime binding")

    async def _verify_managed_binding_before_control(self) -> None:
        if self._managed_binding is None:
            return
        if not self.token:
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "the host agent token is missing")
        async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
            response = await client.get(f"{self.url}/agent/manifest", headers=self._headers())
            response.raise_for_status()
            self._validate_managed_binding(response.json())

    def _observe_generation_admission(self, manifest: dict) -> None:
        # Passive observations may close Runtime ingress, never reopen it. Model
        # health is independent: embeddings remain usable while generation is fenced.
        paused = manifest.get("generation_paused")
        if (
            type(paused) is not bool or manifest.get("engine") != self.engine_name
            or manifest.get("backend") != self.backend_id
        ):
            self.generation_paused = True
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "invalid host admission observation")
        self.generation_paused = self.generation_paused or paused

    def _observe_native_roles(self, manifest: object) -> None:
        roles = manifest.get("roles") if isinstance(manifest, dict) else None
        valid_backend = isinstance(manifest, dict) and manifest.get("backend") == self.backend_id
        for name, current in self._roles.items():
            # SlimServe generation has its own stronger evidence validator and
            # monitor. Its independently placed native roles share this check.
            if current.status != "healthy" or (name == "generation" and self.engine_name != "llama.cpp"):
                continue
            observed = roles.get(name) if isinstance(roles, dict) else None
            if (
                valid_backend
                and (name != "generation" or manifest.get("engine") == self.engine_name)
                and isinstance(observed, dict)
                and observed.get("status") == "healthy"
                and valid_native_model_identity(observed.get("model"))
                and observed.get("model") == current.engine_model
                and observed.get("revision") == current.revision
                and observed.get("context_length") == current.context_length
            ):
                continue
            # Withdrawal is sticky until an explicit startup accepts/probes the
            # role again. A live child may still have weights in memory; that is
            # not proof of its current managed identity or embedding dimensions.
            self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
            if name == "generation":
                self.generation_paused = True

    async def refresh_role_observations(self) -> None:
        if not any(info.status != "disabled" for info in self._roles.values()):
            return
        generation_enabled = self.role_info("generation").status != "disabled"
        try:
            if not self.token:
                raise BackendStartError("HOST_AGENT_UNREACHABLE", "the host agent token is missing")
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                response = await client.get(f"{self.url}/agent/manifest", headers=self._headers())
                response.raise_for_status()
        except (httpx.HTTPError, BackendStartError):
            # No authenticated current observation: fence generation without
            # inventing withdrawal or restored identity for another role.
            if generation_enabled:
                self.generation_paused = True
            return
        try:
            manifest = response.json()
            self._validate_managed_binding(manifest)
            self._require_managed_role_engines(manifest)
        except (ValueError, TypeError, AttributeError, BackendStartError):
            manifest = None
            self.generation_paused = True
        self._observe_native_roles(manifest)
        if generation_enabled:
            try:
                self._observe_generation_admission(manifest)
            except (TypeError, AttributeError, BackendStartError):
                self.generation_paused = True

    async def available_engines(self) -> list[dict]:
        if not self.token:
            return []
        try:
            async with httpx.AsyncClient(timeout=5.0, trust_env=False) as client:
                response = await client.get(f"{self.url}/agent/manifest", headers=self._headers())
                response.raise_for_status()
                manifest = response.json()
            self._validate_managed_binding(manifest)
            self._require_managed_role_engines(manifest)
            available = manifest.get("available_engines")
            if not isinstance(available, list) or len(available) > 8:
                return []
            return [entry for entry in available if isinstance(entry, dict)
                    and isinstance(entry.get("name"), str)
                    and entry.get("name") in {"slimserve", "llama.cpp"}
                    and isinstance(entry.get("version"), str)
                    and 0 < len(entry["version"]) <= 128
                    and entry.get("variants") == (
                        ["metal"] if entry["name"] == "slimserve" else ["metal-arm64"]
                    )]
        except (httpx.HTTPError, ValueError, TypeError, AttributeError, BackendStartError):
            return []

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        on_state("loading")
        self._set_managed_binding(config)
        enabled = [name for name, role in config.roles.items() if role.enabled]
        manifest = await self._wait_for_agent(enabled)
        if self.engine_name == "llama.cpp" and (
            manifest.get("engine") == "slimserve" or manifest.get("configured_engine") == "slimserve"
        ):
            try:
                async with httpx.AsyncClient(timeout=HOST_GENERATION_TIMEOUT, trust_env=False) as client:
                    response = await client.post(
                        f"{self.url}/agent/admin/roles/generation/llama", headers=self._headers()
                    )
                    response.raise_for_status()
                manifest = await self._wait_for_agent(enabled)
                if manifest.get("engine") != "llama.cpp" or manifest.get("state") != "healthy":
                    raise BackendStartError("MODEL_LOAD_FAILED", "host engine cutback is not ready")
            except httpx.HTTPError as exc:
                raise BackendStartError("MODEL_LOAD_FAILED", "host llama generation cutback failed") from exc
        self._require_managed_role_engines(manifest)
        self._agent_manifest = manifest
        if "generation" in enabled:
            self._observe_generation_admission(manifest)
        agent_roles: dict = manifest.get("roles") or {}

        for name, role in config.roles.items():
            if not role.enabled:
                self._roles[name] = RoleInfo(status="disabled")
                continue
            agent_role = agent_roles.get(name)
            if not agent_role or agent_role.get("status") != "healthy":
                logger.warning(
                    "agent does not serve role %s (agent status: %s)",
                    name,
                    (agent_role or {}).get("status"),
                )
                self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_NOT_FOUND")
                continue
            model = agent_role.get("model")
            # SlimServe generation carries separately validated host evidence;
            # every llama role must already name its exact managed Runtime file.
            if self.engine_name == "llama.cpp" or name != "generation":
                if not valid_native_model_identity(model):
                    self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
                    continue
            self._clients[name] = httpx.AsyncClient(
                base_url=self.url, headers=self._headers(name), timeout=600.0
            )
            info = RoleInfo(
                status="healthy",
                engine_model=model,
                revision=agent_role.get("revision"),
                context_length=agent_role.get("context_length"),
            )
            self._roles[name] = info
            if name == "embedding":
                try:
                    result = await self.embeddings(
                        {"model": role.served_model_name, "input": "dimension probe"}
                    )
                    info.dimensions = len(result["data"][0]["embedding"])
                    info.modalities = ["text"]
                except Exception:
                    logger.exception("agent embedding dimension probe failed")
                    info.status = "unhealthy"
                    info.error_code = "DIMENSION_MISMATCH"
        if self.engine_name == "llama.cpp" and "generation" in enabled:
            # A stopped Runtime leaves the independently supervised host ingress
            # closed. Reopen only after a fresh, authenticated resume exchange.
            if self.generation_paused and all(self._roles[name].status == "healthy" for name in enabled):
                try:
                    await self.resume()
                except Exception as exc:
                    raise BackendStartError("ENGINE_RESUME_FAILED", "host generation could not resume") from exc

    async def quiesce(self) -> None:
        await self._verify_managed_binding_before_control()
        self.generation_paused = True
        if not self.token:
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "the host agent token is missing")
        async with httpx.AsyncClient(timeout=630.0, trust_env=False) as client:
            response = await client.post(
                f"{self.url}/agent/admin/roles/generation/quiesce",
                headers={**self._headers(), "X-Sovereign-Engine": "llama.cpp"},
            )
            response.raise_for_status()
            body = response.json()
            if (
                not isinstance(body, dict) or set(body) != {"paused", "idle"}
                or body["paused"] is not True or body["idle"] is not True
            ):
                raise RuntimeError("host engine did not acknowledge generation idle")

    async def resume(self) -> None:
        await self._verify_managed_binding_before_control()
        self.generation_paused = True
        if not self.token:
            raise BackendStartError("HOST_AGENT_UNREACHABLE", "the host agent token is missing")
        async with httpx.AsyncClient(timeout=630.0, trust_env=False) as client:
            response = await client.post(
                f"{self.url}/agent/admin/roles/generation/resume",
                headers={**self._headers(), "X-Sovereign-Engine": "llama.cpp"},
            )
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, dict) or set(body) != {"paused"} or body["paused"] is not False:
                raise RuntimeError("host engine did not acknowledge generation resume")
        self.generation_paused = False

    async def _wait_for_agent(self, enabled_roles: list[str]) -> dict:
        """Wait for the agent to be reachable AND its models to finish
        loading (llama.cpp loads take a while). Returns the last manifest;
        per-role health is judged by the caller."""
        wait = float(os.environ.get("SOVEREIGN_AGENT_WAIT", "300"))
        deadline = asyncio.get_running_loop().time() + wait
        last_error: Exception | None = None
        manifest: dict | None = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    resp = await client.get(f"{self.url}/agent/manifest", headers=self._headers())
                    if resp.status_code == 200:
                        manifest = resp.json()
                        self._validate_managed_binding(manifest)
                        roles = manifest.get("roles") or {}
                        pending = [
                            name
                            for name in enabled_roles
                            if (roles.get(name) or {}).get("status") == "loading"
                        ]
                        if not pending:
                            return manifest
                        logger.info("waiting for agent roles: %s", pending)
                    else:
                        last_error = RuntimeError(f"agent manifest: {resp.status_code}")
                except httpx.HTTPError as exc:
                    last_error = exc
                await asyncio.sleep(2.0)
        if manifest is not None:
            return manifest  # roles that never left loading are judged unhealthy
        raise BackendStartError(
            "HOST_AGENT_UNREACHABLE",
            f"host inference agent not reachable at {self.url}: {last_error}",
            recoverable=True,
        )

    async def shutdown(self) -> None:
        clients, self._clients = dict(self._clients), {}
        for client in clients.values():
            await client.aclose()

    def role_info(self, role: str) -> RoleInfo:
        return self._roles.get(role, RoleInfo(status="disabled"))

    def role_client(self, role: str) -> httpx.AsyncClient | None:
        # Keep the client alive for already-admitted streams until shutdown,
        # but never give a new caller a role whose current proof was withdrawn.
        if self.role_info(role).status != "healthy":
            return None
        return self._clients.get(role)

    def engine_version(self) -> str | None:
        # The host-agent manifest reports its package version, not the
        # llama.cpp build. Do not present it as engine-version evidence.
        return None

    def accelerator(self) -> dict:
        return {
            "vendor": "apple" if self.backend_id == "metal" else "cpu",
            "device_count": 1,
            "unified_memory": True,
        }
