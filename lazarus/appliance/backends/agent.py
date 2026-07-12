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

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.backends.roleclient import RoleClientMixin
from lazarus.appliance.config import RuntimeConfig

logger = logging.getLogger("sovereign.agent")


class AgentBackend(RoleClientMixin, EngineBackend):
    def __init__(self) -> None:
        self.backend_id = os.environ.get("SOVEREIGN_AGENT_BACKEND_ID", "metal")
        self.url = os.environ.get("SOVEREIGN_AGENT_URL", "http://host.docker.internal:9100")
        self.token = os.environ.get("SOVEREIGN_AGENT_TOKEN", "")
        self._roles: dict[str, RoleInfo] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._agent_manifest: dict = {}

    def _headers(self, role: str | None = None) -> dict[str, str]:
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if role:
            headers["X-Sovereign-Role"] = role
        return headers

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        on_state("loading")
        enabled = [name for name, role in config.roles.items() if role.enabled]
        manifest = await self._wait_for_agent(enabled)
        self._agent_manifest = manifest
        agent_roles: dict = manifest.get("roles") or {}

        for name, role in config.roles.items():
            if not role.enabled:
                self._roles[name] = RoleInfo(status="disabled")
                continue
            agent_role = agent_roles.get(name)
            if not agent_role or agent_role.get("status") != "healthy":
                logger.warning("agent does not serve role %s (agent status: %s)",
                               name, (agent_role or {}).get("status"))
                self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_NOT_FOUND")
                continue
            self._clients[name] = httpx.AsyncClient(
                base_url=self.url, headers=self._headers(name), timeout=600.0
            )
            info = RoleInfo(
                status="healthy",
                engine_model=agent_role.get("model", "host-agent"),
                revision=agent_role.get("revision", "host"),
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
                    resp = await client.get(
                        f"{self.url}/agent/manifest", headers=self._headers()
                    )
                    if resp.status_code == 200:
                        manifest = resp.json()
                        roles = manifest.get("roles") or {}
                        pending = [
                            name for name in enabled_roles
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
        return self._clients.get(role)

    def engine_version(self) -> str:
        return f"host-agent/{self._agent_manifest.get('agent_version', 'unknown')} ({self._agent_manifest.get('engine', 'unknown')})"

    def accelerator(self) -> dict:
        return {
            "vendor": "apple" if self.backend_id == "metal" else "cpu",
            "device_count": 1,
            "unified_memory": True,
        }
