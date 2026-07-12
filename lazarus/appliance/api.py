"""The appliance's HTTP surface: contract endpoints + OpenAI routes.

The API layer owns role routing (§9.3), auth, admission control (per-role
concurrency + embedding throttling, §9.4), and SSE framing. Backends own
inference. This app must be constructible and serving even when the config
failed to load (§3.2).
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, generate_latest

from lazarus.appliance.backends.base import EngineBackend
from lazarus.appliance.config import RuntimeConfig
from lazarus.appliance.manifest import ManifestBuilder
from lazarus.appliance.state import STATES, StateMachine

logger = logging.getLogger("sovereign.api")

REQUESTS = Counter(
    "sovereign_requests_total",
    "Requests served, by role and outcome",
    ["role", "served_model", "outcome"],
)
IN_FLIGHT = Gauge("sovereign_requests_in_flight", "In-flight requests by role", ["role"])
STATE_GAUGE = Gauge("sovereign_runtime_state", "Runtime state machine position", ["state"])


def wire_state_metric(state: StateMachine) -> None:
    def update(current: str) -> None:
        for known in STATES:
            STATE_GAUGE.labels(state=known).set(1.0 if known == current else 0.0)

    state.on_change(update)


class Throttled(Exception):
    pass


class Admission:
    """Best-effort per-role admission control (§9.4)."""

    def __init__(self, config: RuntimeConfig | None) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._generation_waiting = 0
        self._embed_threshold: int | None = None
        if config is not None:
            for name, role in config.enabled_roles().items():
                self._semaphores[name] = asyncio.Semaphore(role.max_concurrent_requests)
            embedding = config.roles.embedding
            self._embed_threshold = embedding.throttle_when_generation_queue_above

    @property
    def generation_waiting(self) -> int:
        return self._generation_waiting

    @asynccontextmanager
    async def slot(self, role: str):
        if (
            role == "embedding"
            and self._embed_threshold is not None
            and self._generation_waiting > self._embed_threshold
        ):
            raise Throttled()
        semaphore = self._semaphores.get(role)
        if semaphore is None:
            yield
            return
        if role == "generation":
            self._generation_waiting += 1
        acquired = False
        try:
            await semaphore.acquire()
            acquired = True
            if role == "generation":
                self._generation_waiting -= 1
            IN_FLIGHT.labels(role=role).inc()
            try:
                yield
            finally:
                IN_FLIGHT.labels(role=role).dec()
        finally:
            if role == "generation" and not acquired:
                self._generation_waiting -= 1
            if acquired:
                semaphore.release()


def _error(status: int, message: str, error_type: str, code: str | None = None) -> JSONResponse:
    body: dict = {"error": {"message": message, "type": error_type}}
    if code:
        body["error"]["code"] = code
    return JSONResponse(status_code=status, content=body)


def _model_not_found(model: object) -> JSONResponse:
    return _error(404, f"model {model!r} is not served by this role", "invalid_request_error", "model_not_found")


def _not_ready() -> JSONResponse:
    return _error(503, "runtime is not ready", "server_error")


def build_app(
    *,
    state: StateMachine,
    backend: EngineBackend,
    config: RuntimeConfig | None,
    manifest: ManifestBuilder,
    lifecycle,
) -> FastAPI:
    from lazarus.appliance.manifest import RUNTIME_VERSION

    admission = Admission(config)
    api_key = config.api_key if config else None
    alias_map = config.alias_to_role() if config else {}
    runtime_id = f"sovereign-runtime-{manifest.profile}-{RUNTIME_VERSION}"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(lifecycle())
        yield
        task.cancel()
        await backend.shutdown()

    app = FastAPI(title="Sovereign Runtime", lifespan=lifespan)

    def role_status(name: str) -> str:
        return backend.role_info(name).status

    def is_ready() -> bool:
        return state.state == "healthy"

    @app.middleware("http")
    async def enforce_api_key(request: Request, call_next):
        if api_key and request.url.path.startswith("/v1/"):
            if request.headers.get("Authorization") != f"Bearer {api_key}":
                return _error(401, "invalid API key", "authentication_error")
        return await call_next(request)

    # ── contract endpoints ───────────────────────────────────────────────

    @app.get("/health/live")
    def health_live() -> dict:
        return {"status": "alive", "state": state.state}

    @app.get("/health/ready")
    def health_ready() -> JSONResponse:
        ready = is_ready()
        body = {
            "ready": ready,
            "state": state.state,
            "required_roles": {
                "generation": role_status("generation") == "healthy",
                "embedding": role_status("embedding") == "healthy",
            },
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.get("/health")
    def health() -> dict:
        roles = {}
        for name in ("generation", "embedding"):
            info = backend.role_info(name)
            role_config = config.role(name) if config else None
            entry: dict = {
                "status": info.status,
                "model_loaded": info.status == "healthy",
            }
            if role_config and role_config.served_model_name:
                entry["served_model_name"] = role_config.served_model_name
            if info.error_code:
                entry["error_code"] = info.error_code
            if name == "embedding" and info.modalities:
                entry["modalities"] = info.modalities
            roles[name] = entry
        return {
            "status": "healthy" if state.state == "healthy" else state.state,
            "state": state.state,
            "runtime_id": runtime_id,
            "roles": roles,
        }

    @app.get("/runtime/manifest")
    def runtime_manifest() -> dict:
        return manifest.build()

    @app.get("/runtime/errors")
    def runtime_errors() -> dict:
        return state.errors_payload()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    # ── OpenAI surface ───────────────────────────────────────────────────

    @app.get("/v1/models")
    def list_models() -> dict:
        data = [
            {"id": alias, "object": "model", "owned_by": "sovereign"}
            for alias, role in alias_map.items()
            if role_status(role) == "healthy"
        ]
        return {"object": "list", "data": data}

    def route(body: dict, expected_role: str, required_fields: tuple[str, ...]):
        """Shared request gate: field presence, readiness, role routing."""
        for field_name in required_fields:
            if field_name not in body:
                return _error(400, f"{field_name} is required", "invalid_request_error")
        model = body.get("model")
        role = alias_map.get(model)
        if role != expected_role or role_status(expected_role) != "healthy":
            return _model_not_found(model)
        if not is_ready():
            return _not_ready()
        return None

    async def forward(role: str, path: str, raw_body: bytes):
        """Raw in-process forward to the role's vLLM app (streaming intact).
        Returns None when the backend has no per-role app (fake backend)."""
        role_client = getattr(backend, "role_client", None)
        client = role_client(role) if role_client else None
        if client is None:
            return None
        upstream = client.build_request(
            "POST", path, content=raw_body, headers={"Content-Type": "application/json"}
        )
        resp = await client.send(upstream, stream=True)

        async def relay():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()

        return StreamingResponse(
            relay(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
        )

    async def openai_endpoint(
        request: Request, role: str, required_fields: tuple[str, ...]
    ):
        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return _error(400, "request body must be JSON", "invalid_request_error")
        if denied := route(body, role, required_fields):
            return denied
        alias = body["model"]
        try:
            async with admission.slot(role):
                if (response := await forward(role, request.url.path, raw_body)) is not None:
                    REQUESTS.labels(role=role, served_model=alias, outcome="ok").inc()
                    return response
                # Fake-backend path (tests/CI): dict-based handlers.
                if request.url.path.endswith("chat/completions") and body.get("stream"):

                    async def sse():
                        async for chunk in backend.chat_completion_stream(body):
                            if isinstance(chunk, str):
                                yield chunk if chunk.endswith("\n\n") else chunk + "\n\n"
                            else:
                                yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"

                    REQUESTS.labels(role=role, served_model=alias, outcome="ok").inc()
                    return StreamingResponse(sse(), media_type="text/event-stream")
                if request.url.path.endswith("chat/completions"):
                    result = await backend.chat_completion(body)
                elif request.url.path.endswith("/completions"):
                    result = await backend.completion(body)
                else:
                    result = await backend.embeddings(body)
                REQUESTS.labels(role=role, served_model=alias, outcome="ok").inc()
                return result
        except Throttled:
            REQUESTS.labels(role=role, served_model=alias, outcome="throttled").inc()
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "embedding requests throttled under generation pressure", "type": "server_error"}},
                headers={"Retry-After": "1"},
            )
        except Exception as exc:
            logger.exception("%s request failed", role)
            REQUESTS.labels(role=role, served_model=alias, outcome="error").inc()
            return _error(500, f"engine error: {exc}", "server_error")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await openai_endpoint(request, "generation", ("model", "messages"))

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await openai_endpoint(request, "generation", ("model", "prompt"))

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        return await openai_endpoint(request, "embedding", ("model", "input"))

    return app
