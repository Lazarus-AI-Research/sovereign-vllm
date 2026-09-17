"""Deployments: one supervised llama.cpp process per served model, created and
removed by Sovereign Control through the admin API and reached through
``/deployments/{id}/v1``. Roles are the installer's fixed pair; deployments are
what an operator adds and removes while the appliance runs, so each has its own
port, admission gate and lifecycle and none restarts another."""

from __future__ import annotations

import asyncio
import hashlib
import re
import socket
import time
from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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
# The manifest is read by clients with a five-second budget; every deployment
# is probed at once and each probe is cut off well inside it, so a hung
# deployment can never make the manifest fail for the roles beside it.
OBSERVE_TIMEOUT = 1.5

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
    mmproj_sha256: str | None = None
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


# The fixed roles keep their names whether or not they are configured now, so
# enabling one later never collides with a deployment.
RESERVED_DEPLOYMENT_IDS = frozenset({"generation", "embedding"})


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
        "--alias", deployment.served_model_name,
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


def deployment_lock(agent: Agent, deployment_id: str) -> asyncio.Lock:
    return agent.deployment_locks.setdefault(deployment_id, asyncio.Lock())


def free_port(agent: Agent) -> int:
    """Called with role_lock held; a port handed out is reserved until the
    transition that took it commits or gives it back."""
    taken = {role.port for role in agent.config.roles.values()}
    taken |= {deployment.port for deployment in agent.config.deployments.values()}
    # A child kept registered without a record (a failed creation whose stop
    # failed) still owns its port until its termination is confirmed.
    taken |= {process.port for process in agent.deployments.values()}
    taken |= agent.port_reservations
    taken.add(agent.config.port)
    for port in DEPLOYMENT_PORTS:
        if port not in taken and port_available(port):
            return port
    raise RuntimeError("no free deployment port")


def port_available(port: int) -> bool:
    """A port nothing on this host listens on, whatever the records say."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def status_of(agent: Agent, deployment_id: str, deployment: AgentDeployment, healthy: bool) -> dict:
    process = agent.deployments.get(deployment_id)
    running = process is not None and process.running()
    try:
        model = agent.observed_model(deployment.model_path)
    except (OSError, ValueError, TypeError, RuntimeError):
        return {"status": "unhealthy", "error_code": "MODEL_LOAD_FAILED", "kind": deployment.kind}
    admission = agent.deployment_admission.get(deployment_id)
    paused = admission is not None and admission.paused
    if healthy and running:
        # A child that answers behind a closed gate is not serving: nothing
        # reaches it until the gate reopens.
        status = "paused" if paused else "healthy"
    else:
        status = "loading" if running else "unhealthy"
    return {
        "status": status,
        "admission": "paused" if paused else "open",
        "kind": deployment.kind,
        "model": model,
        "port": deployment.port,
        "served_model_name": deployment.served_model_name,
        "context_length": deployment.context_length,
        "revision": deployment.revision,
        "engine": "llama.cpp",
    }


async def observe_deployments(agent: Agent) -> dict[str, dict]:
    snapshot = list(agent.config.deployments.items())

    async def probe(deployment_id: str) -> bool:
        process = agent.deployments.get(deployment_id)
        if process is None:
            return False
        try:
            return await asyncio.wait_for(process.healthy(), timeout=OBSERVE_TIMEOUT)
        except asyncio.TimeoutError:
            return False

    healthy = await asyncio.gather(*(probe(deployment_id) for deployment_id, _ in snapshot))
    result = {}
    for (deployment_id, deployment), is_healthy in zip(snapshot, healthy):
        # A deployment removed while it was being probed is not reported.
        if agent.config.deployments.get(deployment_id) is not deployment:
            continue
        result[deployment_id] = status_of(agent, deployment_id, deployment, is_healthy)
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


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_deployment_files(deployment: AgentDeployment) -> None:
    """The bytes on disk are the bytes the record was made for. A weight
    overwritten since is refused, never loaded under the recorded revision."""
    if file_digest(deployment.model_path) != deployment.sha256.lower():
        raise ValueError(f"{deployment.model_path} no longer matches its recorded checksum")
    if deployment.mmproj_path and deployment.mmproj_sha256 and file_digest(deployment.mmproj_path) != deployment.mmproj_sha256.lower():
        raise ValueError(f"{deployment.mmproj_path} no longer matches its recorded checksum")


def start_deployment(agent: Agent, deployment_id: str, verify: bool = False, record: AgentDeployment | None = None) -> RoleProcess:
    """Starts the recorded deployment, or a candidate record not yet committed
    to the configuration."""
    from lazarus.agent.server import RoleProcess

    if agent.stopping:
        raise RuntimeError("the agent is shutting down")
    deployment = record or agent.config.deployments[deployment_id]
    agent.observed_model(deployment.model_path)
    # A request's files were checked as it arrived; a restart from agent.yaml
    # checks them again, since the disk may have changed meanwhile.
    if verify:
        verify_deployment_files(deployment)
    return RoleProcess(
        deployment_id, deployment_command(agent, deployment), deployment.port, deployment.model_path,
        revision=deployment.revision, context_length=deployment.context_length,
        authenticated=deployment.kind == "generation",
    )


async def quiesce(agent: Agent, deployment_id: str) -> bool:
    """Close the gate and wait for what is in flight; returns the gate's state
    before, so a transition that changes nothing can put it back."""
    admission = agent.deployment_admission.setdefault(deployment_id, Admission())
    was_paused = admission.paused
    admission.paused = True
    try:
        await asyncio.wait_for(admission.idle.wait(), timeout=IDLE_TIMEOUT)
    except asyncio.TimeoutError:
        pass
    return was_paused


async def stop_deployment(agent: Agent, deployment_id: str) -> None:
    # Terminating a child can take up to ten seconds of blocking waits; that
    # runs off the event loop so every other deployment keeps streaming. Only
    # a transition worker calls this, and no request's cancel scope reaches a
    # worker, so the wait always runs to the child's end.
    process = agent.deployments.get(deployment_id)
    if process is None:
        return
    # The child stays registered until it is confirmed gone, so a failed stop
    # can be retried and an agent shutdown still finds it.
    await asyncio.to_thread(process.stop)
    if agent.deployments.get(deployment_id) is process:
        del agent.deployments[deployment_id]


class Transition:
    """One replacement or removal. It runs in a task of its own so the request
    that asked for it can be cancelled without leaving it half done: the
    worker notices the request is gone at its next safe point and either
    changes nothing or rolls back, to completion."""

    def __init__(self) -> None:
        self.abandoned = False
        # Set once the candidate is confirmed and the worker waits to commit
        # it; a request cancelled after this point is still rolled back.
        self.committing = False


async def run_transition(agent: Agent, worker_coroutine, transition: Transition) -> dict:
    worker = asyncio.create_task(worker_coroutine)
    # A worker abandoned by its request still finishes and is joined at
    # shutdown; its outcome is read so a failure there is never an
    # unretrieved exception.
    agent.transitions.add(worker)
    worker.add_done_callback(agent.transitions.discard)
    worker.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        transition.abandoned = True
        raise


async def apply_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest) -> dict:
    # Checksumming multi-gigabyte weights must not stall every other
    # deployment's stream, so it runs off the event loop.
    model = await asyncio.to_thread(agent.resolve_model, request.artifact, request.sha256)
    mmproj = None
    if request.mmproj:
        mmproj = await asyncio.to_thread(agent.resolve_model, request.mmproj, request.mmproj_sha256)
    transition = Transition()
    return await run_transition(agent, replace_deployment(agent, deployment_id, request, model, mmproj, transition), transition)


async def replace_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest, model, mmproj, transition: Transition) -> dict:
    async with deployment_lock(agent, deployment_id):
        if transition.abandoned:
            # The request went away while this waited for the lock; nothing
            # has been touched, and nothing will be.
            return {"status": "unchanged", "id": deployment_id}
        previous = agent.config.deployments.get(deployment_id)
        async with agent.role_lock:
            if transition.abandoned:
                # The request went away while this waited for the shared
                # lock; no drain, no process, nothing.
                return {"status": "unchanged", "id": deployment_id}
            port = previous.port if previous else free_port(agent)
            agent.port_reservations.add(port)
        try:
            return await replace_on_port(agent, deployment_id, request, model, mmproj, transition, previous, port)
        finally:
            agent.port_reservations.discard(port)


async def replace_on_port(agent: Agent, deployment_id: str, request: DeploymentRequest, model, mmproj, transition: Transition, previous, port: int) -> dict:
    """The transition proper, with the deployment's own lock held and its
    port reserved by the caller."""
    candidate = AgentDeployment(
        kind=request.kind, model_path=str(model), mmproj_path=str(mmproj) if mmproj else None,
        mmproj_sha256=request.mmproj_sha256.lower() if request.mmproj_sha256 else None,
        revision=request.revision.lower(), sha256=request.sha256.lower(),
        port=port,
        served_model_name=request.served_model_name, context_length=request.context_length,
        pooling=request.pooling, normalization=request.normalization,
    )
    if previous is not None:
        was_paused = await quiesce(agent, deployment_id)
        if transition.abandoned:
            # Nothing has changed: the process that was serving keeps
            # serving, behind the gate it had before the drain.
            agent.deployment_admission[deployment_id].paused = was_paused
            return {"status": "unchanged", "id": deployment_id}
    agent.deployment_admission[deployment_id] = Admission()
    agent.deployment_admission[deployment_id].paused = True
    try:
        # The previous process goes, and so does a child a failed creation
        # left registered without a record: a handle is never overwritten
        # while its child may still run.
        if previous is not None or deployment_id in agent.deployments:
            await stop_deployment(agent, deployment_id)
        # The drain may have taken minutes; the files are checked again
        # right before they are loaded, so what starts is what was pinned.
        await asyncio.to_thread(verify_deployment_files, candidate)
        # The candidate stays out of the configuration until it is confirmed:
        # another deployment's save meanwhile persists only what was verified.
        process = start_deployment(agent, deployment_id, record=candidate)
        agent.deployments[deployment_id] = process
        await wait_deployment_ready(agent, candidate, process)
        transition.committing = True
        async with agent.role_lock:
            # Checked again under the lock: the request may have gone while
            # this waited for it.
            if transition.abandoned:
                raise RuntimeError("the request was cancelled before the deployment was confirmed")
            committed = agent.config.deployments
            agent.config.deployments = {**committed, deployment_id: candidate}
            try:
                agent.save_config()
            except Exception:
                # A record that could not be saved is not a record: what was
                # committed before stays, in memory as on disk.
                agent.config.deployments = committed
                raise
    except Exception as exc:
        # A cancelled request is a failed one. The configuration never held
        # the candidate, so whatever cleanup does next, the rejected
        # candidate is never what is persisted. The previous process is
        # already gone, so it is restored from files checked again against
        # their checksums.
        rolled_back, rollback_error = False, None
        try:
            await stop_deployment(agent, deployment_id)
            if previous is None:
                agent.deployment_admission.pop(deployment_id, None)
            else:
                await asyncio.to_thread(verify_deployment_files, previous)
                restored = start_deployment(agent, deployment_id, record=previous)
                agent.deployments[deployment_id] = restored
                await wait_deployment_ready(agent, previous, restored)
                agent.deployment_admission[deployment_id].paused = False
            async with agent.role_lock:
                agent.save_config()
            rolled_back = True
        except Exception as rollback_exc:
            rollback_error = str(rollback_exc)
        return {
            "status": "unhealthy", "id": deployment_id, "error": str(exc),
            "rolled_back": rolled_back, "rollback_verified": rolled_back, "rollback_error": rollback_error,
        }
    agent.deployment_admission[deployment_id].paused = False
    return {"status": "healthy", "id": deployment_id, **status_of(agent, deployment_id, candidate, True)}


class PersistenceError(RuntimeError):
    """The process is gone but agent.yaml still records it; a retry finishes
    the removal rather than reporting a deployment absent that a restarted
    agent would serve again."""


async def remove_deployment(agent: Agent, deployment_id: str) -> dict:
    transition = Transition()
    return await run_transition(agent, forget_deployment(agent, deployment_id, transition), transition)


async def forget_deployment(agent: Agent, deployment_id: str, transition: Transition) -> dict:
    async with deployment_lock(agent, deployment_id):
        if transition.abandoned:
            return {"status": "unchanged", "id": deployment_id}
        previous = agent.config.deployments.get(deployment_id)
        if previous is None:
            # No record, but a child a failed creation could not stop may
            # still be registered; it is stopped rather than called absent.
            if deployment_id in agent.deployments:
                await stop_deployment(agent, deployment_id)
                agent.deployment_admission.pop(deployment_id, None)
                return {"status": "stopped", "id": deployment_id}
            return {"status": "absent", "id": deployment_id}
        was_paused = await quiesce(agent, deployment_id)
        if transition.abandoned:
            agent.deployment_admission[deployment_id].paused = was_paused
            return {"status": "unchanged", "id": deployment_id}
        await stop_deployment(agent, deployment_id)
        # The record goes, is saved, or comes back, under one acquisition of
        # the shared lock: no creation can take the port in between.
        async with agent.role_lock:
            agent.config.deployments = {k: v for k, v in agent.config.deployments.items() if k != deployment_id}
            try:
                agent.save_config()
            except Exception as exc:
                agent.config.deployments = {**agent.config.deployments, deployment_id: previous}
                raise PersistenceError(f"deployment stopped but not forgotten: {exc}") from exc
        agent.deployment_admission.pop(deployment_id, None)
        return {"status": "stopped", "id": deployment_id}


def register_deployment_routes(app: FastAPI, agent: Agent) -> None:
    @app.get("/agent/deployments")
    async def list_deployments():
        return {"deployments": await observe_deployments(agent)}

    @app.put("/agent/admin/deployments/{deployment_id}")
    async def put_deployment(deployment_id: str, request: DeploymentRequest):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        if deployment_id in RESERVED_DEPLOYMENT_IDS or deployment_id in agent.config.roles:
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
        try:
            return await remove_deployment(agent, deployment_id)
        except (PersistenceError, OSError, RuntimeError) as exc:
            # The child may still be there; the record says so, and a retry
            # stops it again.
            return JSONResponse(status_code=500, content={"error": str(exc), "id": deployment_id})

    @app.api_route("/deployments/{deployment_id}/v1/{path:path}", methods=["GET", "POST"])
    async def proxy_deployment(deployment_id: str, path: str, request: Request):
        deployment = agent.config.deployments.get(deployment_id)
        if deployment is None:
            return JSONResponse(status_code=404, content={"error": f"unknown deployment {deployment_id!r}"})
        if path not in ALLOWED_PATHS[deployment.kind]:
            return JSONResponse(status_code=404, content={"error": "unsupported deployment endpoint"})
        admission = agent.deployment_admission.setdefault(deployment_id, Admission())
        process = agent.deployments.get(deployment_id)
        # A configured deployment with no process is one being replaced or
        # removed: a retryable pause, never an unknown name.
        if admission.paused or process is None:
            return JSONResponse(status_code=503, content={"error": "deployment admission is paused"})
        body = await request.body()
        if admission.paused or agent.deployments.get(deployment_id) is not process:
            return JSONResponse(status_code=503, content={"error": "deployment is being replaced"})
        # A child that exited still owns its record until it is stopped; its
        # port may meanwhile belong to something else, so nothing is forwarded.
        if not process.running():
            return JSONResponse(status_code=503, content={"error": "deployment process is not running"})
        client = httpx.AsyncClient(timeout=600.0, trust_env=False)
        # The request is built before admission is taken, so a request that
        # cannot be built never leaves a count behind.
        try:
            upstream = client.build_request(
                request.method, f"http://127.0.0.1:{process.port}/v1/{path}", content=body,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json"), **process.headers()},
            )
        except (UnicodeError, ValueError) as exc:
            await client.aclose()
            return JSONResponse(status_code=400, content={"error": f"request could not be forwarded: {exc}"})
        admission.enter()
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
            async for chunk in response.aiter_raw():
                yield chunk

        async def cleanup():
            # Runs shielded from the client's disconnect, and releases
            # admission even when closing the upstream fails: a leaked count
            # would make the next replacement wait the whole drain timeout.
            try:
                await response.aclose()
                await client.aclose()
            finally:
                admission.leave()

        from lazarus.agent.server import RoleStreamingResponse

        return RoleStreamingResponse(
            relay(), cleanup=cleanup, status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )
