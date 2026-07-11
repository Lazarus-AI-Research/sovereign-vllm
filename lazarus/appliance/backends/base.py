"""Backend protocol: what the appliance needs from an inference engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

from lazarus.appliance.config import RuntimeConfig


class BackendStartError(Exception):
    """Raised by start() for load-time failures. Carries the error code for
    /runtime/errors; the launcher decides degraded vs configuration_error."""

    def __init__(self, code: str, message: str, *, role: str | None = None, recoverable: bool = True):
        super().__init__(message)
        self.code = code
        self.role = role
        self.recoverable = recoverable


@dataclass
class RoleInfo:
    """Live, observed per-role state — the manifest's source of truth."""

    status: str = "loading"  # healthy | unhealthy | loading | disabled
    error_code: str | None = None
    engine_model: str | None = None
    revision: str | None = None
    context_length: int | None = None
    dimensions: int | None = None  # embedding: probed, never assumed (§10.1)
    modalities: list[str] | None = None


class EngineBackend(ABC):
    """One backend instance per runtime process. start() loads roles
    strictly serially (generation before embedding, §24 steps 8–9)."""

    backend_id: str = "unknown"  # manifest "backend" field: cuda|rocm|xpu|metal|cpu|mock

    @abstractmethod
    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        """Load enabled roles. Per-role failures are recorded in role_info
        (status unhealthy + error_code) rather than raised; raise
        BackendStartError only for whole-backend failures."""

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    def role_info(self, role: str) -> RoleInfo: ...

    @abstractmethod
    def engine_version(self) -> str: ...

    # OpenAI surface. Bodies are already validated for role routing by the
    # appliance; backends may raise for engine errors — the API layer maps
    # exceptions to 500s and records runtime errors.

    @abstractmethod
    async def chat_completion(self, body: dict) -> dict: ...

    @abstractmethod
    def chat_completion_stream(self, body: dict) -> AsyncIterator[dict]:
        """Yield chat.completion.chunk dicts; the API layer adds SSE framing
        and the final [DONE]."""

    @abstractmethod
    async def completion(self, body: dict) -> dict: ...

    @abstractmethod
    async def embeddings(self, body: dict) -> dict: ...
