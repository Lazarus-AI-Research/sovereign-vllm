"""Deployments: one supervised process per served model, created and removed
by Sovereign Control through the admin API and reached through
``/deployments/{id}/v1``. An LLM or embedding model is a llama-server child;
an image model is a stable-diffusion.cpp server child. Each has its own port,
admission gate and lifecycle, so none restarts another."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import socket
import sys
import time
from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from lazarus.agent.server import Agent, ServerProcess

DEPLOYMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# The agent listens on 9100 and the retired fixed roles held 9101 and 9102;
# deployments take a block above both, so an agent upgraded in place never
# collides with a role process still winding down.
DEPLOYMENT_PORTS = range(9110, 9200)
# A request that has not finished in this long is not worth keeping a
# replacement waiting for.
IDLE_TIMEOUT = 600.0
READY_TIMEOUT = 300.0
# The manifest is read by clients with a five-second budget; every deployment
# is probed at once and each probe is cut off well inside it, so a hung
# deployment can never make the manifest fail for the ones beside it.
OBSERVE_TIMEOUT = 1.5

ALLOWED_PATHS = {
    "generation": {"chat/completions", "completions", "models"},
    "embedding": {"embeddings", "models"},
    "image": {"images/generations", "models"},
    "transcription": {"audio/transcriptions"},
    "speech": {"audio/speech"},
}
KINDS = tuple(ALLOWED_PATHS)
# The kinds served without a context window: a diffusion, transcription or
# speech model has no prompt to size.
NO_CONTEXT = ("image", "transcription", "speech")

# The files an image model is served with beside its diffusion weights, by
# the flag stable-diffusion.cpp takes them under. A model that needs none of
# them (a single-file checkpoint) names none.
IMAGE_COMPONENTS = {"clip_l": "--clip_l", "t5xxl": "--t5xxl", "vae": "--vae"}
IMAGE_SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpm++2m", "lcm")
# A weight file the agent loads: GGUF for llama.cpp, GGUF or safetensors for
# stable-diffusion.cpp, whose text encoders and autoencoder ship as either.
WEIGHT_SUFFIXES = (".gguf", ".safetensors")

# The voice configuration piper reads beside a speech model's weights: the
# one component a speech deployment names, and it must be the file piper
# finds by name.
SPEECH_COMPONENTS = {"config"}
# The language a transcription deployment listens for; "auto" lets the
# model detect it.
LANGUAGE = r"^(auto|[a-z]{2,3})$"

# Where each kind's server says it is up. llama-server answers /health once
# its model is loaded; sd-server listens only once its model is loaded and
# answers the models listing; whisper-server answers /health with 503 while
# it loads; piper's server listens only once its voice is loaded.
HEALTH_PATHS = {
    "generation": "/health", "embedding": "/health", "image": "/v1/models",
    "transcription": "/health", "speech": "/voices",
}
ENGINES = {
    "generation": "llama.cpp", "embedding": "llama.cpp", "image": "stable-diffusion.cpp",
    "transcription": "whisper.cpp", "speech": "piper",
}


# What each kind's loader accepts: GGUF for llama-server, GGUF or safetensors
# for stable-diffusion.cpp, ggml for whisper-server, ONNX for piper.
def weight_suffixes(kind: str) -> tuple[str, ...]:
    return {"image": WEIGHT_SUFFIXES, "transcription": (".bin",), "speech": (".onnx",)}.get(kind, (".gguf",))


def component_suffixes(kind: str) -> tuple[str, ...]:
    return (".json",) if kind == "speech" else weight_suffixes(kind)


class Component(BaseModel):
    """One named file an image deployment loads beside its diffusion model."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        from pathlib import Path

        path = Path(value)
        if not path.is_absolute() or str(path) != value or value.startswith("//") or ".." in path.parts:
            raise ValueError("model paths must be canonical absolute paths")
        return value


class AgentDeployment(BaseModel):
    """What agent.yaml records for a deployment; enough to start it again."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: Literal[KINDS]
    model_path: str
    mmproj_path: str | None = None
    mmproj_sha256: str | None = None
    revision: str
    sha256: str
    port: int = Field(ge=1, le=65535)
    served_model_name: str
    context_length: int = Field(default=0, ge=0, le=131072)
    pooling: Literal["mean", "last", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None
    # An image deployment's companions and the sampling it was pinned with;
    # a speech deployment's one companion is its voice configuration.
    components: dict[str, Component] = {}
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg_scale: float | None = Field(default=None, ge=0, le=30)
    sampler: Literal[IMAGE_SAMPLERS] | None = None
    # A transcription deployment's spoken language.
    language: str | None = Field(default=None, pattern=LANGUAGE)

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
            self.pooling = self.pooling or "mean"
            self.normalization = self.normalization or "l2"
        elif self.pooling is not None or self.normalization is not None:
            raise ValueError("pooling and normalization apply to embedding deployments only")
        if self.kind != "generation" and self.mmproj_path is not None:
            raise ValueError(f"a {self.kind} deployment has no projector")
        if self.kind == "image":
            unknown = set(self.components) - set(IMAGE_COMPONENTS)
            if unknown:
                raise ValueError(f"unknown image components: {', '.join(sorted(unknown))}")
            self.steps = self.steps or 20
            self.cfg_scale = 7.0 if self.cfg_scale is None else self.cfg_scale
            self.sampler = self.sampler or "euler"
        else:
            if self.steps is not None or self.cfg_scale is not None or self.sampler is not None:
                raise ValueError("steps, cfg_scale and sampler apply to image deployments only")
            if self.kind == "speech":
                speech_components(self.components, self.model_path)
            elif self.components:
                raise ValueError("components apply to image and speech deployments only")
        if self.kind == "transcription":
            self.language = self.language or "auto"
        elif self.language is not None:
            raise ValueError("language applies to transcription deployments only")
        if self.kind not in NO_CONTEXT and self.context_length < 128:
            raise ValueError("a language model deployment needs a context length of at least 128")
        return self


# piper finds a voice's configuration by the weights' own name with .json
# appended; the one component a speech deployment names must be that file.
def speech_components(components: dict, model_path: str) -> None:
    if set(components) != SPEECH_COMPONENTS:
        raise ValueError("a speech deployment names its voice configuration as its one component, config")
    if components["config"].path != model_path + ".json":
        raise ValueError("a voice's configuration is the weights' name with .json appended")


class ComponentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class DeploymentRequest(BaseModel):
    """Constrained input; arbitrary llama.cpp flags never cross this boundary."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[KINDS]
    artifact: str
    mmproj: str | None = None
    revision: str = Field(pattern=r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    mmproj_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    served_model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    context_length: int = Field(default=8192, ge=128, le=131072)
    pooling: Literal["mean", "last", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None
    components: dict[str, ComponentRequest] = {}
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg_scale: float | None = Field(default=None, ge=0, le=30)
    sampler: Literal[IMAGE_SAMPLERS] | None = None
    language: str | None = Field(default=None, pattern=LANGUAGE)

    @model_validator(mode="after")
    def projector_checksum(self):
        if (self.mmproj is None) != (self.mmproj_sha256 is None):
            raise ValueError("a projector is named together with its sha256")
        if self.kind != "image" and (self.steps is not None or self.cfg_scale is not None or self.sampler is not None):
            raise ValueError("steps, cfg_scale and sampler apply to image deployments only")
        if self.kind == "image":
            unknown = set(self.components) - set(IMAGE_COMPONENTS)
            if unknown:
                raise ValueError(f"unknown image components: {', '.join(sorted(unknown))}")
        elif self.kind == "speech":
            if set(self.components) != SPEECH_COMPONENTS:
                raise ValueError("a speech deployment names its voice configuration as its one component, config")
        elif self.components:
            raise ValueError("components apply to image and speech deployments only")
        if self.kind != "transcription" and self.language is not None:
            raise ValueError("language applies to transcription deployments only")
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
    if deployment.kind == "image":
        return image_command(agent, deployment)
    if deployment.kind == "transcription":
        return transcription_command(agent, deployment)
    if deployment.kind == "speech":
        return speech_command(agent, deployment)
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


# The server has no API key of its own; it listens on loopback and is
# reached through the agent's proxy, which gates admission. A model with
# components is standalone diffusion weights; one without is a full
# checkpoint carrying its own text encoder and autoencoder, which the server
# loads under a different flag.
def image_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    command = [
        agent.config.sd_server,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(deployment.port),
        "--diffusion-model" if deployment.components else "--model", deployment.model_path,
    ]
    for name, flag in IMAGE_COMPONENTS.items():
        component = deployment.components.get(name)
        if component is not None:
            command += [flag, component.path]
    command += [
        "--steps", str(deployment.steps),
        "--cfg-scale", str(deployment.cfg_scale),
        "--sampling-method", deployment.sampler,
    ]
    return command


# whisper-server answers its inference route under the OpenAI transcription
# path, as the gateway and the workspace call it; timestamps are left out of
# the text it returns.
def transcription_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    return [
        agent.config.whisper_server,
        "--host", "127.0.0.1",
        "--port", str(deployment.port),
        "-m", deployment.model_path,
        "--inference-path", "/v1/audio/transcriptions",
        "--no-timestamps",
        "--language", deployment.language or "auto",
    ]


# piper's own HTTP server, run by the agent's interpreter unless the
# configuration names another; it loads the voice named by path and finds
# the configuration beside it.
def speech_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    server = shlex.split(agent.config.piper_server) or [sys.executable, "-m", "piper.http_server"]
    return [*server, "--host", "127.0.0.1", "--port", str(deployment.port), "-m", deployment.model_path]


def deployment_lock(agent: Agent, deployment_id: str) -> asyncio.Lock:
    return agent.deployment_locks.setdefault(deployment_id, asyncio.Lock())


def free_port(agent: Agent) -> int:
    """Called with records_lock held; a port handed out is reserved until the
    transition that took it commits or gives it back."""
    taken = {deployment.port for deployment in agent.config.deployments.values()}
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
        "engine": ENGINES[deployment.kind],
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


async def wait_deployment_ready(agent: Agent, deployment: AgentDeployment, process: ServerProcess) -> None:
    timeout = getattr(agent, "deployment_ready_timeout", READY_TIMEOUT)
    if deployment.kind == "embedding":
        await agent.wait_embedding_ready(process, timeout=timeout)
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
    for component in deployment.components.values():
        if file_digest(component.path) != component.sha256.lower():
            raise ValueError(f"{component.path} no longer matches its recorded checksum")


def start_deployment(agent: Agent, deployment_id: str, verify: bool = False, record: AgentDeployment | None = None) -> ServerProcess:
    """Starts the recorded deployment, or a candidate record not yet committed
    to the configuration."""
    from lazarus.agent.server import ServerProcess

    if agent.stopping:
        raise RuntimeError("the agent is shutting down")
    deployment = record or agent.config.deployments[deployment_id]
    agent.observed_model(deployment.model_path)
    # A request's files were checked as it arrived; a restart from agent.yaml
    # checks them again, since the disk may have changed meanwhile.
    if verify:
        verify_deployment_files(deployment)
    return ServerProcess(
        deployment_id, deployment_command(agent, deployment), deployment.port, deployment.model_path,
        revision=deployment.revision, context_length=deployment.context_length,
        authenticated=deployment.kind == "generation",
        health_path=HEALTH_PATHS[deployment.kind], engine=ENGINES[deployment.kind],
    )


async def quiesce(agent: Agent, deployment_id: str, transition: "Transition | None" = None) -> bool:
    """Close the gate and wait for what is in flight, or for the transition
    to be abandoned, whichever comes first; returns the gate's state before,
    so a transition that changes nothing can put it back."""
    admission = agent.deployment_admission.setdefault(deployment_id, Admission())
    was_paused = admission.paused
    admission.paused = True
    waits = [asyncio.ensure_future(admission.idle.wait())]
    if transition is not None:
        waits.append(asyncio.ensure_future(transition.gone.wait()))
    try:
        await asyncio.wait(waits, timeout=IDLE_TIMEOUT, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for wait in waits:
            wait.cancel()
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
        # Set with abandoned, so a drain waiting on in-flight requests wakes
        # at once and puts the gate back instead of holding it for the
        # whole idle timeout.
        self.gone = asyncio.Event()
        # Set once the candidate is confirmed and the worker waits to commit
        # it; a request cancelled after this point is still rolled back.
        self.committing = False


async def run_transition(agent: Agent, worker_coroutine, transition: Transition, http_request: Request | None = None) -> dict:
    worker = asyncio.create_task(worker_coroutine)
    # A worker abandoned by its request still finishes and is joined at
    # shutdown; its outcome is read so a failure there is never an
    # unretrieved exception.
    agent.transitions.add(worker)
    worker.add_done_callback(agent.transitions.discard)
    worker.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    # A client that went away is not cancelled by the server; it is watched
    # for, and its transition abandoned like a cancelled one.
    watcher = asyncio.create_task(abandon_on_disconnect(http_request, worker, transition)) if http_request is not None else None
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        transition.abandoned = True
        transition.gone.set()
        raise
    finally:
        if watcher is not None:
            watcher.cancel()


DISCONNECT_POLL = 0.5


async def abandon_on_disconnect(http_request: Request, worker: asyncio.Task, transition: Transition) -> None:
    while not worker.done():
        if await http_request.is_disconnected():
            transition.abandoned = True
            transition.gone.set()
            return
        await asyncio.sleep(DISCONNECT_POLL)


async def apply_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest, http_request: Request | None = None) -> dict:
    # Checksumming multi-gigabyte weights must not stall every other
    # deployment's stream, so it runs off the event loop.
    # llama-server loads GGUF alone; stable-diffusion.cpp takes its encoders
    # and autoencoder as safetensors too. A file the loader would refuse is
    # refused here, before a serving process is touched.
    suffixes = weight_suffixes(request.kind)
    model = await asyncio.to_thread(agent.resolve_model, request.artifact, request.sha256, suffixes)
    mmproj = None
    if request.mmproj:
        mmproj = await asyncio.to_thread(agent.resolve_model, request.mmproj, request.mmproj_sha256, suffixes)
    components = {}
    for name, component in request.components.items():
        path = await asyncio.to_thread(agent.resolve_model, component.artifact, component.sha256, component_suffixes(request.kind))
        components[name] = Component(path=str(path), sha256=component.sha256.lower())
    transition = Transition()
    return await run_transition(agent, replace_deployment(agent, deployment_id, request, model, mmproj, components, transition), transition, http_request)


async def replace_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest, model, mmproj, components: dict[str, Component], transition: Transition) -> dict:
    async with deployment_lock(agent, deployment_id):
        if transition.abandoned:
            # The request went away while this waited for the lock; nothing
            # has been touched, and nothing will be.
            return {"status": "unchanged", "id": deployment_id}
        previous = agent.config.deployments.get(deployment_id)
        async with agent.records_lock:
            if transition.abandoned:
                # The request went away while this waited for the shared
                # lock; no drain, no process, nothing.
                return {"status": "unchanged", "id": deployment_id}
            port = previous.port if previous else free_port(agent)
            agent.port_reservations.add(port)
        try:
            return await replace_on_port(agent, deployment_id, request, model, mmproj, components, transition, previous, port)
        finally:
            agent.port_reservations.discard(port)


async def replace_on_port(agent: Agent, deployment_id: str, request: DeploymentRequest, model, mmproj, components: dict[str, Component], transition: Transition, previous, port: int) -> dict:
    """The transition proper, with the deployment's own lock held and its
    port reserved by the caller."""
    candidate = AgentDeployment(
        kind=request.kind, model_path=str(model), mmproj_path=str(mmproj) if mmproj else None,
        mmproj_sha256=request.mmproj_sha256.lower() if request.mmproj_sha256 else None,
        revision=request.revision.lower(), sha256=request.sha256.lower(),
        port=port,
        served_model_name=request.served_model_name,
        context_length=0 if request.kind in NO_CONTEXT else request.context_length,
        pooling=request.pooling, normalization=request.normalization,
        components=components, steps=request.steps, cfg_scale=request.cfg_scale, sampler=request.sampler,
        language=request.language,
    )
    if previous is not None:
        was_paused = await quiesce(agent, deployment_id, transition)
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
        async with agent.records_lock:
            # Checked again under the lock: the request may have gone while
            # this waited for it, and so may the candidate. Either is the
            # failure the rollback below handles, never a record.
            if transition.abandoned:
                raise RuntimeError("the request was cancelled before the deployment was confirmed")
            if not process.running():
                raise RuntimeError("the deployment exited before it could be recorded")
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
            async with agent.records_lock:
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


async def remove_deployment(agent: Agent, deployment_id: str, http_request: Request | None = None) -> dict:
    transition = Transition()
    return await run_transition(agent, forget_deployment(agent, deployment_id, transition), transition, http_request)


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
        was_paused = await quiesce(agent, deployment_id, transition)
        if transition.abandoned:
            agent.deployment_admission[deployment_id].paused = was_paused
            return {"status": "unchanged", "id": deployment_id}
        await stop_deployment(agent, deployment_id)
        # The record goes, is saved, or comes back, under one acquisition of
        # the shared lock: no creation can take the port in between.
        async with agent.records_lock:
            agent.config.deployments = {k: v for k, v in agent.config.deployments.items() if k != deployment_id}
            try:
                agent.save_config()
            except Exception as exc:
                agent.config.deployments = {**agent.config.deployments, deployment_id: previous}
                raise PersistenceError(f"deployment stopped but not forgotten: {exc}") from exc
        agent.deployment_admission.pop(deployment_id, None)
        return {"status": "stopped", "id": deployment_id}


# The containers whisper-server cannot open: it decodes uploads in memory
# with miniaudio, which reads WAV, MP3, FLAC and Ogg Vorbis and nothing a
# browser records. The workspace converts its recordings to WAV before they
# arrive; an API client sending one of these is told what to send instead of
# a bare decode failure.
UNSUPPORTED_CONTAINERS = ((b"\x1a\x45\xdf\xa3", "WebM"), (b"ftyp", "MP4"))


def unsupported_container(body: bytes) -> str | None:
    marker = body.find(b'name="file"')
    if marker < 0:
        return None
    start = body.find(b"\r\n\r\n", marker)
    if start < 0:
        return None
    head = body[start + 4 : start + 16]
    for magic, name in UNSUPPORTED_CONTAINERS:
        if head.startswith(magic) or (name == "MP4" and head[4:8] == magic):
            return name
    return None


# What a voice reads aloud is the visible answer: a model's thinking, which
# Chat folds away, and the markdown that shapes text on a screen are not
# speech. Thinking blocks go; fences, headings, emphasis, links and list
# markers leave their words behind.
THINKING = re.compile(r"<(think|thought|reasoning)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE)
UNFINISHED_THINKING = re.compile(r"<(think|thought|reasoning)>.*$", re.DOTALL | re.IGNORECASE)
MARKDOWN = (
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`]*)`"), r"\1"),
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE), ""),
    (re.compile(r"(\*\*|__|~~)(.+?)\1", re.DOTALL), r"\2"),
    (re.compile(r"(?<!\w)[*_](.+?)[*_](?!\w)", re.DOTALL), r"\1"),
    (re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE), ""),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
)


def spoken_text(text: str) -> str:
    text = UNFINISHED_THINKING.sub("", THINKING.sub("", text))
    for pattern, replacement in MARKDOWN:
        text = pattern.sub(replacement, text)
    return text.strip()


# The OpenAI speech request as piper takes it: the visible answer as text,
# and the speed as a length scale. The voice named is the deployment's; the
# answer is WAV, whatever format was asked for, since that is what the
# voice produces.
def speech_request(body: bytes) -> dict:
    try:
        request = json.loads(body or b"{}")
    except ValueError:
        raise ValueError("the request is not JSON")
    text = request.get("input") if isinstance(request, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input is required")
    if len(text) > 4096:
        raise ValueError("input is at most 4096 characters")
    text = spoken_text(text)
    if not text:
        raise ValueError("input holds nothing to say once thinking and markup are set aside")
    speed = request.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4.0:
        raise ValueError("speed is between 0.25 and 4.0")
    return {"text": text, "length_scale": round(1.0 / float(speed), 4)}


async def synthesize(process: "ServerProcess", admission: Admission, body: bytes):
    try:
        payload = speech_request(body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    admission.enter()
    try:
        async with httpx.AsyncClient(timeout=600.0, trust_env=False) as client:
            answer = await client.post(f"http://127.0.0.1:{process.port}/synthesize", json=payload)
    except httpx.HTTPError:
        return JSONResponse(status_code=503, content={"error": "deployment engine unavailable"})
    finally:
        admission.leave()
    if answer.status_code != 200:
        return JSONResponse(status_code=502, content={"error": f"the voice answered {answer.status_code}"})
    return Response(content=answer.content, media_type="audio/wav")


def register_deployment_routes(app: FastAPI, agent: Agent) -> None:
    @app.get("/agent/deployments")
    async def list_deployments():
        return {"deployments": await observe_deployments(agent)}

    @app.put("/agent/admin/deployments/{deployment_id}")
    async def put_deployment(deployment_id: str, request: DeploymentRequest, http_request: Request):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        try:
            result = await apply_deployment(agent, deployment_id, request, http_request)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return JSONResponse(status_code=200 if result["status"] == "healthy" else 422, content=result)

    @app.delete("/agent/admin/deployments/{deployment_id}")
    async def delete_deployment(deployment_id: str, http_request: Request):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        try:
            return await remove_deployment(agent, deployment_id, http_request)
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
        if deployment.kind == "speech":
            return await synthesize(process, admission, body)
        if deployment.kind == "transcription" and (container := unsupported_container(body)):
            return JSONResponse(status_code=415, content={"error": f"{container} audio is not decoded here; send WAV, MP3, FLAC or Ogg Vorbis"})
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

        from lazarus.agent.server import RelayedResponse

        return RelayedResponse(
            relay(), cleanup=cleanup, status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )
