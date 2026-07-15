"""sovereign-runtime-agent: supervise llama.cpp servers, serve one private
port. Fails closed: no token, no service. Binds loopback only — the agent is
never exposed beyond the host (§22)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

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
    def __init__(self, config: AgentConfig):
        self.config = config
        self.token = os.environ.get(config.token_env, "")
        self.roles: dict[str, RoleProcess] = {}

    def start_roles(self) -> None:
        for name, role in self.config.roles.items():
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
            self.roles[name] = RoleProcess(name, command, role.port, role.model_path)

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

    agent = Agent(config)
    uvicorn.run(build_app(agent), host=config.listen, port=config.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
