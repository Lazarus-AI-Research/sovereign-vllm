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
from contextlib import AsyncExitStack, asynccontextmanager

from anyio import CancelScope
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


class AdmittedStreamingResponse(StreamingResponse):
    """Keep admission until the response finishes, including disconnects."""

    def __init__(self, *args, admission: AsyncExitStack, **kwargs):
        super().__init__(*args, **kwargs)
        self._admission = admission

    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            with CancelScope(shield=True):
                await self._admission.aclose()


class Admission:
    """Best-effort per-role admission control (§9.4)."""

    def __init__(self, config: RuntimeConfig | None) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._generation_waiting = 0
        self._generation_active = 0
        self._generation_idle = asyncio.Event()
        self._generation_idle.set()
        self._embed_threshold: int | None = None
        if config is not None:
            for name, role in config.enabled_roles().items():
                self._semaphores[name] = asyncio.Semaphore(role.max_concurrent_requests)
            embedding = config.roles.embedding
            if embedding is not None and embedding.enabled:
                self._embed_threshold = embedding.throttle_when_generation_queue_above

    @property
    def generation_waiting(self) -> int:
        return self._generation_waiting

    async def wait_generation_idle(self) -> None:
        await self._generation_idle.wait()

    @asynccontextmanager
    async def slot(self, role: str):
        if (
            role == "embedding"
            and self._embed_threshold is not None
            and self._generation_waiting > self._embed_threshold
        ):
            raise Throttled()
        semaphore = self._semaphores.get(role)
        if role == "generation" and semaphore is not None:
            self._generation_waiting += 1
        acquired = False
        try:
            if semaphore is not None:
                await semaphore.acquire()
                acquired = True
                if role == "generation":
                    self._generation_waiting -= 1
            if role == "generation":
                self._generation_active += 1
                self._generation_idle.clear()
            IN_FLIGHT.labels(role=role).inc()
            try:
                yield
            finally:
                IN_FLIGHT.labels(role=role).dec()
                if role == "generation":
                    self._generation_active -= 1
                    if self._generation_active == 0:
                        self._generation_idle.set()
        finally:
            if role == "generation" and semaphore is not None and not acquired:
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


def _remote_media_error(messages: object) -> JSONResponse | None:
    """Multimodal embedding sovereignty rule (runtime-contract §embeddings):
    media arrives as base64 data URIs only; the runtime never fetches remote
    URLs. Any url-bearing content part that is not a data: URI is rejected."""
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            for value in part.values():
                if isinstance(value, dict) and "url" in value:
                    url = value["url"]
                    if not (isinstance(url, str) and url.startswith("data:")):
                        return _error(
                            400,
                            "remote media URLs are not allowed; send base64 data URIs",
                            "invalid_request_error",
                        )
    return None


def _unsupported_generation_content_error(messages: object) -> JSONResponse | None:
    """Text generation rejects multimodal message parts before engine dispatch."""
    for message in messages if isinstance(messages, list) else []:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            return _error(
                400,
                "the configured generation role accepts text content only",
                "invalid_request_error",
                "unsupported_modality",
            )
    return None


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
    generation_control = asyncio.Lock()
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

    def is_ready(role: str | None = None) -> bool:
        if config is None or state.state != "healthy":
            return False
        if backend.generation_paused and (
            role == "generation" or (role is None and config.roles.generation.enabled)
        ):
            return False
        if role is not None:
            return role_status(role) == "healthy"
        return all(role_status(name) == "healthy" for name in config.enabled_roles())

    async def refresh_role_observations() -> None:
        refresh = getattr(backend, "refresh_role_observations", None)
        if refresh is not None:
            await refresh()
            manifest.write()

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
    async def health_ready() -> JSONResponse:
        await refresh_role_observations()
        ready = is_ready()
        required_roles = {
            name: role_status(name) == "healthy"
            for name in (config.enabled_roles() if config else {})
        }
        body = {
            "ready": ready,
            "state": state.state,
            "required_roles": required_roles,
        }
        return JSONResponse(status_code=200 if ready else 503, content=body)

    @app.get("/health")
    async def health() -> dict:
        await refresh_role_observations()
        roles = {}
        configured_roles = [
            name for name in ("generation", "embedding")
            if config is None or config.role(name) is not None
        ]
        for name in configured_roles:
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
    async def runtime_manifest() -> dict:
        await refresh_role_observations()
        return manifest.build()

    @app.get("/runtime/errors")
    def runtime_errors() -> dict:
        return state.errors_payload()

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    async def managed_generation(request: Request, *, resume: bool) -> JSONResponse:
        if not api_key or request.headers.get("Authorization") != f"Bearer {api_key}":
            return _error(401, "invalid API key", "authentication_error")
        async for chunk in request.stream():
            if chunk:
                return _error(400, "request body is not allowed", "invalid_request_error")
        async with generation_control:
            backend.generation_paused = True
            try:
                manifest.write()
                if resume:
                    await backend.resume()
                    if backend.generation_paused is not False:
                        raise RuntimeError("engine resume was not acknowledged")
                else:
                    # Complete admitted responses before engine-specific idle
                    # proof. This also preserves native llama streams; HTTP
                    # drain alone never acknowledges safe disruption.
                    await asyncio.wait_for(admission.wait_generation_idle(), timeout=600)
                    await backend.quiesce()
                    if backend.generation_paused is not True:
                        raise RuntimeError("engine pause was not acknowledged")
                manifest.write()
            except BaseException as exc:
                backend.generation_paused = True
                try:
                    manifest.write()
                except Exception:
                    logger.error("failed to publish closed generation admission")
                if not isinstance(exc, Exception):
                    raise
                operation = "resume" if resume else "quiesce"
                return _error(
                    503, f"engine {operation} was not acknowledged", "server_error",
                    "ENGINE_RESUME_FAILED" if resume else "ENGINE_QUIESCE_FAILED",
                )
        return JSONResponse(content={"resumed" if resume else "quiesced": True})

    @app.post("/runtime/admin/generation/quiesce")
    async def quiesce_generation(request: Request) -> JSONResponse:
        return await managed_generation(request, resume=False)

    @app.post("/runtime/admin/generation/resume")
    async def resume_generation(request: Request) -> JSONResponse:
        return await managed_generation(request, resume=True)

    # ── OpenAI surface ───────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models() -> dict:
        await refresh_role_observations()
        data = [
            {"id": alias, "object": "model", "owned_by": "sovereign"}
            for alias, role in alias_map.items()
            if role_status(role) == "healthy"
        ]
        return {"object": "list", "data": data}

    def route(body: dict, expected_role: str, required_fields: tuple[str | tuple[str, ...], ...]):
        """Shared request gate: field presence, readiness, role routing.
        A tuple entry in required_fields means any-of (e.g. input|messages)."""
        for field_name in required_fields:
            names = field_name if isinstance(field_name, tuple) else (field_name,)
            if not any(name in body for name in names):
                return _error(400, f"{' or '.join(names)} is required", "invalid_request_error")
        model = body.get("model")
        role = alias_map.get(model)
        if role != expected_role:
            return _model_not_found(model)
        if not is_ready(expected_role):
            return _not_ready()
        return None

    async def forward(role: str, path: str, raw_body: bytes, admission_stack: AsyncExitStack):
        """Raw in-process forward to the role's vLLM app (streaming intact).
        Returns None when the backend has no per-role app (fake backend)."""
        role_client = getattr(backend, "role_client", None)
        client = role_client(role) if role_client else None
        if client is None:
            return None
        if role == "generation":
            admission_stack.push_async_callback(refresh_role_observations)
        upstream = client.build_request(
            "POST", path, content=raw_body, headers={"Content-Type": "application/json"}
        )
        resp = await client.send(upstream, stream=True)
        admission_stack.push_async_callback(resp.aclose)

        async def relay():
            async for chunk in resp.aiter_raw():
                yield chunk

        return AdmittedStreamingResponse(
            relay(),
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type"),
            admission=admission_stack.pop_all(),
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
        if role == "generation" and request.url.path.endswith("/chat/completions"):
            if denied := _unsupported_generation_content_error(body.get("messages")):
                return denied
        if request.url.path.endswith("/embeddings") and "messages" in body:
            if denied := _remote_media_error(body["messages"]):
                return denied
        alias = body["model"]
        try:
            async with AsyncExitStack() as admission_stack:
                await admission_stack.enter_async_context(admission.slot(role))
                # A queued request may outlive either admission or role proof.
                await refresh_role_observations()
                if not is_ready(role):
                    return _not_ready()
                if (response := await forward(role, request.url.path, raw_body, admission_stack)) is not None:
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
                    return AdmittedStreamingResponse(
                        sse(), media_type="text/event-stream", admission=admission_stack.pop_all()
                    )
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
        # Text uses standard OpenAI `input`; multimodal items arrive as a
        # chat-style `messages` array (runtime-contract extended schema).
        return await openai_endpoint(request, "embedding", ("model", ("input", "messages")))

    return app
