"""sovereign-runtime-agent: supervise one llama-server process per deployment
and serve them on one private port. Fails closed: no token, no service. Binds
loopback only — the agent is never exposed beyond the host (§22)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from anyio import CancelScope
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from lazarus.agent import log_level
from lazarus.agent.config import AgentConfig, load_agent_config, valid_native_model_identity
from lazarus.agent.deployments import Admission, observe_deployments, register_deployment_routes, start_deployment
from lazarus.agent.memory import memory_held
from lazarus.agent.queueing import Queueing
from lazarus.appliance.manifest import RUNTIME_VERSION

logger = logging.getLogger("sovereign.agent.server")

AGENT_VERSION = RUNTIME_VERSION


class RelayedResponse(StreamingResponse):
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


# SlimServe as the manifest names it: installed when its command is, at the
# version its package records (SlimServe's distribution is named vllm).
def slimserve_engine(command: str) -> dict | None:
    if not command or not os.access(command, os.X_OK):
        return None
    environment = Path(command).resolve().parent.parent
    records = sorted(environment.glob("lib/python3*/site-packages/vllm-*.dist-info"))
    version = records[-1].name.removeprefix("vllm-").removesuffix(".dist-info") if records else "unknown"
    return {"name": "slimserve", "version": version, "adapter": "metal-host-agent", "variants": ["metal-arm64"]}


def mlx_engine() -> dict | None:
    """mlx-lm where the agent's own Python carries it, as the distribution
    built for Apple Silicon does."""
    from importlib import metadata, util

    if util.find_spec("mlx_lm") is None:
        return None
    try:
        version = metadata.version("mlx-lm")
    except metadata.PackageNotFoundError:
        return None
    return {"name": "mlx-lm", "version": version, "adapter": "metal-host-agent", "variants": ["metal-arm64"]}


class ServerProcess:
    """One llama-server child: started here, probed here, stopped here."""

    def __init__(
        self, name: str, command: list[str], port: int, model_path: str,
        *, revision: str | None, context_length: int | None, authenticated: bool = False,
        environment: dict[str, str] | None = None,
        health_path: str = "/health", engine: str = "llama.cpp", weights_root: Path | None = None,
    ):
        self.name = name
        # Where the weights it maps are read from, so they count as its own.
        self.weights_root = weights_root
        self.port = port
        self.model_path = model_path
        self.health_path = health_path
        self.engine = engine
        # Keep loader-input metadata with this child, not a later desired config.
        self.revision = revision
        self.context_length = context_length
        # A generation child answers only with a key the agent alone holds,
        # so nothing on the host reaches it except through the agent. b11459
        # traces the final four key characters; those stay public while 256
        # random bits never enter argv or logs.
        self.api_key = secrets.token_urlsafe(32) + "-agent" if authenticated else None
        # What this child's video ids are signed with beside the agent's
        # secret: two engines numbering their jobs alike never sign the same
        # id, and an id outlives neither the child nor its jobs.
        self.instance = secrets.token_hex(16)
        child_env = None
        if self.api_key is not None or environment:
            child_env = dict(os.environ)
        if self.api_key is not None:
            # b11459 accepts LLAMA_API_KEY without exposing a secret in argv/logs.
            # Extra keys would create ingress outside the agent's admission gate.
            if any(arg.split("=", 1)[0].replace("_", "-") in {"--api-key", "--api-key-file"} for arg in command):
                raise ValueError("generation authentication is agent-owned")
            child_env.pop("LLAMA_ARG_API_KEY_FILE", None)
            child_env["LLAMA_API_KEY"] = self.api_key
            # SlimServe's server, vLLM's, reads its key the same way.
            child_env["VLLM_API_KEY"] = self.api_key
        if environment:
            child_env.update(environment)
        log_dir = Path(os.environ.get("SOVEREIGN_AGENT_LOG_DIR", Path.home() / ".sovereign" / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{name}.{engine}.log"
        logger.info("starting %s: %s (log: %s)", name, " ".join(command), self.log_path)
        with open(self.log_path, "ab") as log_file:
            self.process = subprocess.Popen(command, stdout=log_file, stderr=log_file, env=child_env)

    def running(self) -> bool:
        return self.process.poll() is None

    def memory_held(self) -> tuple[int, int] | None:
        """What the child holds now, and of that the weights it maps; None
        once it has exited or the host cannot say."""
        return memory_held(self.process.pid, self.weights_root) if self.running() else None

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def healthy(self) -> bool:
        if not self.running():
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0, trust_env=False) as client:
                resp = await client.get(f"http://127.0.0.1:{self.port}{self.health_path}", headers=self.headers())
                return resp.status_code == 200
        except httpx.HTTPError:
            return False

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
        self.deployments: dict[str, ServerProcess] = {}
        self.deployment_admission: dict[str, Admission] = {}
        # The requests that waited for a place in each language model,
        # kept across its relaunches.
        self.deployment_queueing: dict[str, Queueing] = {}
        # One lock per deployment for its transitions, so a long drain or
        # readiness wait on one never holds up another; records_lock guards
        # only what they share: ports and the saved configuration.
        self.deployment_locks: dict[str, asyncio.Lock] = {}
        self.port_reservations: set[int] = set()
        # Transition workers in flight, abandoned by their requests or not;
        # the agent joins them before it stops its children.
        self.transitions: set[asyncio.Task] = set()
        # Set once the agent begins to stop: no transition starts a child
        # after it, whatever the join does.
        self.stopping = False
        self.records_lock = asyncio.Lock()
        default_root = self.config_path.parent / "models" if self.config_path else Path.home() / ".sovereign" / "models"
        self.model_root = Path(os.environ.get("SOVEREIGN_AGENT_MODEL_ROOT", default_root)).resolve()
        self.available_engines: list[dict] = []

    async def discover_engines(self) -> None:
        """The installed llama-server's version, and SlimServe's where it is
        installed, for the manifest; not evidence of what any running child
        loaded."""
        self.available_engines = []
        slimserve = slimserve_engine(self.config.slimserve)
        if slimserve is not None:
            self.available_engines.append(slimserve)
        mlx = mlx_engine()
        if mlx is not None:
            self.available_engines.append(mlx)
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
            # "version: 9960 (a935fbffe)" before llama.cpp 0.6, and
            # "version: 0.6.0-dev (build 11459, commit f498f864f)" since.
            match = re.search(rb"(?m)^version: (?:([0-9]{1,8}) \(([0-9a-f]{7,40})\)|[0-9.]+(?:-[a-z]+)? \(build ([0-9]{1,8}), commit ([0-9a-f]{7,40})\))\r?$", output)
            if match is not None:
                self.available_engines.append({
                    "name": "llama.cpp", "version": f"b{(match[1] or match[3]).decode()}-{(match[2] or match[4]).decode()}",
                    "adapter": "metal-host-agent", "variants": ["metal-arm64"],
                })
        except (OSError, asyncio.TimeoutError):
            return
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()

    def start_deployments(self) -> None:
        for deployment_id in self.config.deployments:
            try:
                self.deployments[deployment_id] = start_deployment(self, deployment_id, verify=True)
            except (OSError, ValueError) as exc:
                logger.error("deployment %s did not start: %s", deployment_id, exc)

    async def wait_ready(self, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        pending = set(self.deployments)
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                process = self.deployments.get(name)
                if process is None:
                    pending.discard(name)
                    continue
                if await process.healthy():
                    logger.info("deployment %s healthy on :%d", name, process.port)
                    pending.discard(name)
            if pending:
                await asyncio.sleep(2)
        for name in pending:
            logger.error("deployment %s failed to become healthy", name)

    async def join_transitions(self) -> None:
        """A transition abandoned by its request still finishes; every one is
        awaited before the children are stopped, or a worker could start a
        child after the final sweep."""
        while self.transitions:
            await asyncio.gather(*list(self.transitions), return_exceptions=True)

    def stop(self) -> None:
        error = None
        for name, process in list(self.deployments.items()):
            try:
                process.stop()
            except Exception as exc:
                logger.exception("failed to stop deployment %s", name)
                if error is None:
                    error = exc
        if error is not None:
            raise error

    def save_config(self) -> None:
        if self.config_path is None:
            raise RuntimeError("agent configuration path is unavailable")
        target = self.config_path
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(yaml.safe_dump(self.config.model_dump(exclude_none=True, exclude_unset=True), sort_keys=False))
        temporary.chmod(0o600)
        temporary.replace(target)

    def resolve_model(self, artifact: str, expected_sha256: str, suffixes: tuple[str, ...] = (".gguf",)) -> Path:
        if not valid_native_model_identity(f"/models/{artifact}"):
            raise ValueError("artifact must use a bounded canonical relative native model path")
        model = self._resolve_managed_path(Path(artifact))
        if not model.is_file():
            raise ValueError("artifact must resolve to a model file within the managed model directory")
        if model.suffix.lower() not in suffixes:
            raise ValueError(f"Metal artifacts must be {' or '.join(suffixes)} files")
        digest = hashlib.sha256()
        with model.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256.lower():
            raise ValueError("artifact checksum does not match sha256")
        return model

    def resolve_snapshot(self, artifact: str, files: dict[str, str]) -> Path:
        """An MLX snapshot's directory, every file in it the one pinned."""
        if not valid_native_model_identity(f"/models/{artifact}"):
            raise ValueError("artifact must use a bounded canonical relative native model path")
        directory = self._resolve_managed_path(Path(artifact))
        if not directory.is_dir():
            raise ValueError("an MLX artifact must resolve to a snapshot directory within the managed model directory")
        from lazarus.agent.deployments import snapshot_holds_its_pins

        snapshot_holds_its_pins(directory, files)
        for name, expected in files.items():
            file = self._resolve_managed_path(Path(artifact) / name)
            if not file.is_file():
                raise ValueError(f"{name} is not in the snapshot")
            digest = hashlib.sha256()
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected.lower():
                raise ValueError(f"{name} does not match its sha256")
        return directory

    def _resolve_managed_path(self, relative: Path) -> Path:
        current = self.model_root
        for component in relative.parts:
            current = current / component
            if current.is_symlink():
                raise ValueError("managed model paths cannot contain symlinks")
        resolved = current.resolve(strict=True)
        if resolved != current or not resolved.is_relative_to(self.model_root):
            raise ValueError("model must use its exact managed local path")
        return resolved

    def observed_model(self, model_path: str, directory: bool = False) -> str:
        """Project an actual native loader input, never an intended model
        alias: a weight file, or an MLX snapshot's directory."""
        path = Path(model_path)
        if not path.is_absolute() or str(path) != model_path or ".." in path.parts:
            raise ValueError("observed model must use a canonical absolute path")
        relative = path.relative_to(self.model_root)
        identity = f"/models/{relative.as_posix()}"
        if not valid_native_model_identity(identity):
            raise ValueError("native model identity must use at most 512 UTF-8 bytes and canonical path components")
        resolved = self._resolve_managed_path(relative)
        if not (resolved.is_dir() if directory else resolved.is_file()):
            raise ValueError("observed model must be a managed local file, or an MLX snapshot's directory")
        return identity

    async def wait_embedding_ready(self, process: ServerProcess, timeout: float = 120) -> None:
        """An embedding child is ready once it answers a probe with a vector,
        not merely once its health route answers."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if await process.healthy():
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        f"http://127.0.0.1:{process.port}/v1/embeddings",
                        json={"model": "embedding", "input": "sovereign embedding probe"},
                    )
                if response.status_code != 200:
                    raise RuntimeError(f"embedding probe failed: {response.status_code}: {response.text[:300]}")
                data = response.json().get("data") or []
                if not data or not data[0].get("embedding"):
                    raise RuntimeError("embedding probe returned no vector")
                return
            if not process.running():
                break
            await asyncio.sleep(1)
        raise RuntimeError("embedding deployment did not become healthy before timeout")


class BearerAuth:
    """Every request carries the agent's bearer token or is refused."""

    def __init__(self, app, agent: Agent) -> None:
        self.app = app
        self.agent = agent

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        authorization = ""
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                authorization = value.decode("latin-1")
        if not self.agent.token or authorization != f"Bearer {self.agent.token}":
            response = JSONResponse(status_code=401, content={"error": "invalid agent token"})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def build_app(agent: Agent) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        lifespan_error = None
        # The web server set its own loggers' levels as it started; the
        # level last kept is put back over them.
        log_level.restore(agent.config_path)
        try:
            agent.start_deployments()
            await agent.discover_engines()
            task = asyncio.create_task(agent.wait_ready())
            yield
        except BaseException as exc:
            lifespan_error = exc
            raise
        finally:
            try:
                try:
                    # From here no transition starts a child; those in flight
                    # are joined inside the guarded region, so the final
                    # sweep below runs whatever cuts the join short.
                    agent.stopping = True
                    await agent.join_transitions()
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                finally:
                    agent.stop()
            except Exception:
                if lifespan_error is None:
                    raise
                logger.exception("agent cleanup failed after lifespan failure")

    app = FastAPI(title="Sovereign Runtime Agent", lifespan=lifespan)
    # A plain ASGI layer rather than Starlette's BaseHTTPMiddleware: the
    # latter wraps receive in a way that hides a client's disconnect from the
    # handlers, and a deployment transition must see its client go.
    app.add_middleware(BearerAuth, agent=agent)

    @app.get("/agent/manifest")
    async def manifest():
        return {
            "agent_version": AGENT_VERSION,
            "backend": "metal",
            "available_engines": agent.available_engines,
            "deployments": await observe_deployments(agent),
        }

    @app.put("/agent/log-level")
    async def set_log_level(request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = None
        name = body.get("level") if isinstance(body, dict) else None
        if not isinstance(name, str) or not log_level.apply(name):
            return JSONResponse(status_code=400, content={"error": 'the body is {"level": "error" | "warn" | "info" | "debug"}'})
        try:
            log_level.keep(agent.config_path, name)
        except OSError as error:
            # In force now, but forgotten at the next start: said so, so
            # Control lists the agent as not reached.
            return JSONResponse(status_code=500, content={"error": f"the level is set but could not be kept: {error}"})
        logger.info("log level applied: %s", name)
        return {"level": name}

    register_deployment_routes(app, agent)
    return app


def main() -> int:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    # The web server writes its request lines to standard output, which the
    # service manager points at a file: each line goes out as it is written.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()

    config = load_agent_config(arguments.config)
    token = os.environ.get(config.token_env, "")
    if not token:
        print(f"error: {config.token_env} is required (the agent fails closed)", file=sys.stderr)
        return 1

    agent = Agent(config, arguments.config)
    log_level.restore(agent.config_path)
    uvicorn.run(build_app(agent), host=config.listen, port=config.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
