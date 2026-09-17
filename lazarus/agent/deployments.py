"""Deployments: one supervised llama.cpp process per served model, created and
removed by Sovereign Control through the admin API and reached through
``/deployments/{id}/v1``. Roles are the installer's fixed pair; deployments are
what an operator adds and removes while the appliance runs, so each has its own
port, admission gate and lifecycle and none restarts another."""

from __future__ import annotations

import asyncio
import re
import socket
import time
from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from lazarus.agent.server import Agent, RoleProcess

DEPLOYMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# Roles keep 9101 and 9102; deployments take the next block so a role and a
# deployment can never collide on a port.
DEPLOYMENT_PORTS = range(9110, 9200)
# Bounded like the generation role's quiesce: a request that has not finished
# in this long is not worth keeping a replacement waiting for.
IDLE_TIMEOUT = 600.0
READY_TIMEOUT = 300.0

ALLOWED_PATHS = {
    "generation": {"chat/completions", "completions", "models"},
    "embedding": {"embeddings", "models"},
}


class AgentDeployment(BaseModel):
    """What agent.yaml records for a deployment; enough to start it again."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: Literal["generation", "embedding"]
    model_path: str
    mmproj_path: str | None = None
    revision: str
    sha256: str
    port: int = Field(ge=1, le=65535)
    served_model_name: str
    context_length: int = Field(ge=128, le=131072)
    pooling: Literal["mean", "last", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None

    @field_validator("model_path", "mmproj_path")
    @classmethod
    def canonical_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from pathlib import Path

        path = Path(value)
        if not path.is_absolute() or str(path) != value or value.startswith("//") or ".." in path.parts:
            raise ValueError("model paths must be canonical absolute paths")
        return value

    @model_validator(mode="after")
    def kind_options(self):
        if self.kind == "embedding":
            if self.mmproj_path is not None:
                raise ValueError("an embedding deployment has no projector")
            self.pooling = self.pooling or "mean"
            self.normalization = self.normalization or "l2"
        elif self.pooling is not None or self.normalization is not None:
            raise ValueError("pooling and normalization apply to embedding deployments only")
        return self


class DeploymentRequest(BaseModel):
    """Constrained input; arbitrary llama.cpp flags never cross this boundary."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["generation", "embedding"]
    artifact: str
    mmproj: str | None = None
    revision: str = Field(pattern=r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    mmproj_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    served_model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    context_length: int = Field(default=8192, ge=128, le=131072)
    pooling: Literal["mean", "last", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None

    @model_validator(mode="after")
    def projector_checksum(self):
        if (self.mmproj is None) != (self.mmproj_sha256 is None):
            raise ValueError("a projector is named together with its sha256")
        return self


class Admission:
    """Per-deployment ingress gate: closed while the process is being replaced
    or removed, and a count of requests in flight so a replacement waits for
    them rather than cutting an answer short."""

    def __init__(self) -> None:
        self.paused = False
        self.requests = 0
        self.idle = asyncio.Event()
        self.idle.set()

    def enter(self) -> None:
        self.requests += 1
        self.idle.clear()

    def leave(self) -> None:
        self.requests -= 1
        if self.requests == 0:
            self.idle.set()


def deployment_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    command = [agent.config.llama_server]
    if deployment.kind == "embedding":
        command += [
            "--embedding", "--pooling", deployment.pooling or "mean",
            "--embd-normalize", "2" if (deployment.normalization or "l2") == "l2" else "-1",
        ]
    else:
        command += ["--jinja"]
    command += [
        "--host", "127.0.0.1",
        "--port", str(deployment.port),
        # b9960 applies env before argv, then remote selection after argv; the
        # selectors are cleared so the final -m is the model that loads.
        "--model-url", "", "--hf-repo", "", "--docker-repo", "",
        "-m", deployment.model_path,
    ]
    if deployment.mmproj_path:
        command += ["--mmproj", deployment.mmproj_path]
    command += ["-c", str(deployment.context_length)]
    return command


def free_port(agent: Agent) -> int:
    taken = {role.port for role in agent.config.roles.values()}
    taken |= {deployment.port for deployment in agent.config.deployments.values()}
    taken.add(agent.config.port)
    for port in DEPLOYMENT_PORTS:
        if port in taken:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free deployment port")


def status_of(agent: Agent, deployment_id: str, healthy: bool) -> dict:
    deployment = agent.config.deployments[deployment_id]
    process = agent.deployments.get(deployment_id)
    running = process is not None and process.running()
    try:
        model = agent.observed_model(deployment.model_path)
    except (OSError, ValueError, TypeError, RuntimeError):
        return {"status": "unhealthy", "error_code": "MODEL_LOAD_FAILED", "kind": deployment.kind}
    return {
        "status": "healthy" if healthy and running else ("loading" if running else "unhealthy"),
        "kind": deployment.kind,
        "model": model,
        "port": deployment.port,
        "served_model_name": deployment.served_model_name,
        "context_length": deployment.context_length,
        "revision": deployment.revision,
        "engine": "llama.cpp",
    }


async def observe_deployments(agent: Agent) -> dict[str, dict]:
    result = {}
    for deployment_id in list(agent.config.deployments):
        process = agent.deployments.get(deployment_id)
        healthy = process is not None and await process.healthy()
        result[deployment_id] = status_of(agent, deployment_id, healthy)
    return result


async def wait_deployment_ready(agent: Agent, deployment: AgentDeployment, process: RoleProcess) -> None:
    timeout = getattr(agent, "deployment_ready_timeout", READY_TIMEOUT)
    if deployment.kind == "embedding":
        await agent.wait_role_ready(process, timeout=timeout)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await process.healthy():
            return
        if not process.running():
            break
        await asyncio.sleep(1)
    raise RuntimeError("deployment did not become healthy before timeout")


def start_deployment(agent: Agent, deployment_id: str) -> RoleProcess:
    from lazarus.agent.server import RoleProcess

    deployment = agent.config.deployments[deployment_id]
    agent.observed_model(deployment.model_path)
    return RoleProcess(
        deployment_id, deployment_command(agent, deployment), deployment.port, deployment.model_path,
        revision=deployment.revision, context_length=deployment.context_length,
        authenticated=deployment.kind == "generation",
    )


async def quiesce(agent: Agent, deployment_id: str) -> None:
    admission = agent.deployment_admission.setdefault(deployment_id, Admission())
    admission.paused = True
    try:
        await asyncio.wait_for(admission.idle.wait(), timeout=IDLE_TIMEOUT)
    except asyncio.TimeoutError:
        pass


def stop_deployment(agent: Agent, deployment_id: str) -> None:
    process = agent.deployments.pop(deployment_id, None)
    if process is not None:
        process.stop()


async def apply_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest) -> dict:
    model = agent.resolve_model(request.artifact, request.sha256)
    mmproj = agent.resolve_model(request.mmproj, request.mmproj_sha256) if request.mmproj else None
    async with agent.role_lock:
        previous = agent.config.deployments.get(deployment_id)
        candidate = AgentDeployment(
            kind=request.kind, model_path=str(model), mmproj_path=str(mmproj) if mmproj else None,
            revision=request.revision.lower(), sha256=request.sha256.lower(),
            port=previous.port if previous else free_port(agent),
            served_model_name=request.served_model_name, context_length=request.context_length,
            pooling=request.pooling, normalization=request.normalization,
        )
        if previous is not None:
            await quiesce(agent, deployment_id)
            stop_deployment(agent, deployment_id)
        agent.config.deployments = {**agent.config.deployments, deployment_id: candidate}
        agent.deployment_admission[deployment_id] = Admission()
        agent.deployment_admission[deployment_id].paused = True
        try:
            process = start_deployment(agent, deployment_id)
            agent.deployments[deployment_id] = process
            await wait_deployment_ready(agent, candidate, process)
            agent.save_config()
        except Exception as exc:
            stop_deployment(agent, deployment_id)
            rolled_back, rollback_error = False, None
            try:
                if previous is None:
                    agent.config.deployments = {k: v for k, v in agent.config.deployments.items() if k != deployment_id}
                    agent.deployment_admission.pop(deployment_id, None)
                else:
                    agent.config.deployments = {**agent.config.deployments, deployment_id: previous}
                    restored = start_deployment(agent, deployment_id)
                    agent.deployments[deployment_id] = restored
                    await wait_deployment_ready(agent, previous, restored)
                    agent.deployment_admission[deployment_id].paused = False
                agent.save_config()
                rolled_back = True
            except Exception as rollback_exc:
                rollback_error = str(rollback_exc)
            return {
                "status": "unhealthy", "id": deployment_id, "error": str(exc),
                "rolled_back": rolled_back, "rollback_verified": rolled_back, "rollback_error": rollback_error,
            }
        agent.deployment_admission[deployment_id].paused = False
        return {"status": "healthy", "id": deployment_id, **status_of(agent, deployment_id, True)}


async def remove_deployment(agent: Agent, deployment_id: str) -> dict:
    async with agent.role_lock:
        if deployment_id not in agent.config.deployments:
            return {"status": "absent", "id": deployment_id}
        await quiesce(agent, deployment_id)
        stop_deployment(agent, deployment_id)
        agent.config.deployments = {k: v for k, v in agent.config.deployments.items() if k != deployment_id}
        agent.deployment_admission.pop(deployment_id, None)
        agent.save_config()
        return {"status": "stopped", "id": deployment_id}


def register_deployment_routes(app: FastAPI, agent: Agent) -> None:
    @app.get("/agent/deployments")
    async def list_deployments():
        return {"deployments": await observe_deployments(agent)}

    @app.put("/agent/admin/deployments/{deployment_id}")
    async def put_deployment(deployment_id: str, request: DeploymentRequest):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        if deployment_id in agent.config.roles:
            return JSONResponse(status_code=409, content={"error": "a role owns that name"})
        try:
            result = await apply_deployment(agent, deployment_id, request)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return JSONResponse(status_code=200 if result["status"] == "healthy" else 422, content=result)

    @app.delete("/agent/admin/deployments/{deployment_id}")
    async def delete_deployment(deployment_id: str):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        return await remove_deployment(agent, deployment_id)

    @app.api_route("/deployments/{deployment_id}/v1/{path:path}", methods=["GET", "POST"])
    async def proxy_deployment(deployment_id: str, path: str, request: Request):
        deployment = agent.config.deployments.get(deployment_id)
        process = agent.deployments.get(deployment_id)
        if deployment is None or process is None:
            return JSONResponse(status_code=404, content={"error": f"unknown deployment {deployment_id!r}"})
        if path not in ALLOWED_PATHS[deployment.kind]:
            return JSONResponse(status_code=404, content={"error": "unsupported deployment endpoint"})
        admission = agent.deployment_admission.setdefault(deployment_id, Admission())
        if admission.paused:
            return JSONResponse(status_code=503, content={"error": "deployment admission is paused"})
        body = await request.body()
        if admission.paused or agent.deployments.get(deployment_id) is not process:
            return JSONResponse(status_code=503, content={"error": "deployment is being replaced"})
        admission.enter()
        client = httpx.AsyncClient(timeout=600.0, trust_env=False)
        upstream = client.build_request(
            request.method, f"http://127.0.0.1:{process.port}/v1/{path}", content=body,
            headers={"Content-Type": request.headers.get("Content-Type", "application/json"), **process.headers()},
        )
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError:
            admission.leave()
            await client.aclose()
            return JSONResponse(status_code=503, content={"error": "deployment engine unavailable"})
        except BaseException:
            admission.leave()
            await client.aclose()
            raise

        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()
                admission.leave()

        return StreamingResponse(relay(), status_code=response.status_code, media_type=response.headers.get("content-type"))
