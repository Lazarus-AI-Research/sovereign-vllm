"""sovereign-runtime-agent: supervise llama.cpp servers, serve one private
port. Fails closed: no token, no service. Binds loopback only — the agent is
never exposed beyond the host (§22)."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from lazarus.agent.config import AgentConfig, load_agent_config
from lazarus.appliance.manifest import RUNTIME_VERSION

logger = logging.getLogger("sovereign.agent.server")

AGENT_VERSION = RUNTIME_VERSION


class RoleProcess:
    def __init__(self, name: str, command: list[str], port: int, model_path: str):
        self.name = name
        self.port = port
        self.model_path = model_path
        log_dir = Path(os.environ.get("SOVEREIGN_AGENT_LOG_DIR", Path.home() / ".sovereign" / "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{name}.llama.log"
        logger.info("starting %s: %s (log: %s)", name, " ".join(command), self.log_path)
        log_file = open(self.log_path, "ab")
        self.process = subprocess.Popen(command, stdout=log_file, stderr=log_file)

    def running(self) -> bool:
        return self.process.poll() is None

    async def healthy(self) -> bool:
        if not self.running():
            return False
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"http://127.0.0.1:{self.port}/health")
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


class Agent:
    def __init__(self, config: AgentConfig, config_path: str | Path | None = None):
        self.config = config
        self.config_path = Path(config_path).resolve() if config_path else None
        self.token = os.environ.get(config.token_env, "")
        self.roles: dict[str, RoleProcess] = {}
        self.role_lock = asyncio.Lock()
        default_root = self.config_path.parent / "models" if self.config_path else Path.home() / ".sovereign" / "models"
        self.model_root = Path(os.environ.get("SOVEREIGN_AGENT_MODEL_ROOT", default_root)).resolve()

    def role_command(self, name: str) -> list[str]:
        role = self.config.roles[name]
        command = [
            self.config.llama_server,
            "-m", role.model_path,
            "--host", "127.0.0.1",
            "--port", str(role.port),
            *role.args,
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
            self.roles[name] = self.start_role(name)

    async def wait_ready(self, timeout: float = 300) -> None:
        deadline = time.monotonic() + timeout
        pending = set(self.roles)
        while pending and time.monotonic() < deadline:
            for name in list(pending):
                if await self.roles[name].healthy():
                    logger.info("role %s healthy on :%d", name, self.roles[name].port)
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
        temporary.write_text(yaml.safe_dump(self.config.model_dump(exclude_none=True), sort_keys=False))
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
        task = asyncio.create_task(agent.wait_ready())
        yield
        task.cancel()
        agent.stop()

    app = FastAPI(title="Sovereign Runtime Agent", lifespan=lifespan)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.headers.get("Authorization") != f"Bearer {agent.token}":
            return JSONResponse(status_code=401, content={"error": "invalid agent token"})
        return await call_next(request)

    @app.get("/agent/manifest")
    async def manifest():
        roles = {}
        for name, role in agent.roles.items():
            healthy = await role.healthy()
            roles[name] = {
                "status": "healthy" if healthy else ("loading" if role.running() else "unhealthy"),
                "model": Path(role.model_path).name,
                "context_length": agent.config.roles[name].context_length,
                "revision": agent.config.roles[name].revision,
            }
        return {
            "agent_version": AGENT_VERSION,
            "engine": "llama.cpp",
            "backend": "metal",
            "roles": roles,
        }

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
            candidate = agent.start_role("embedding")
            try:
                await agent.wait_role_ready(candidate)
                agent.save_config()
            except Exception as exc:
                candidate.stop()
                if previous_config is None:
                    agent.config.roles.pop("embedding", None)
                    agent.roles.pop("embedding", None)
                else:
                    agent.config.roles["embedding"] = previous_config
                    agent.roles["embedding"] = agent.start_role("embedding")
                return JSONResponse(status_code=422, content={"error": str(exc), "rolled_back": True})
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
                agent.roles["embedding"] = agent.start_role("embedding")
                return JSONResponse(status_code=500, content={"error": str(exc), "rolled_back": True})
            return {"status": "disabled", "role": "embedding"}

    @app.api_route("/v1/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        role_name = request.headers.get("X-Sovereign-Role", "")
        role = agent.roles.get(role_name)
        if role is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"unknown role {role_name!r} (set X-Sovereign-Role)"},
            )
        body = await request.body()
        client = httpx.AsyncClient(timeout=600.0)
        upstream = client.build_request(
            request.method,
            f"http://127.0.0.1:{role.port}/v1/{path}",
            content=body,
            headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
        )
        resp = await client.send(upstream, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
                await client.aclose()

        return StreamingResponse(
            relay(),
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
