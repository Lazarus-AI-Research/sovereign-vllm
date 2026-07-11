"""Runtime state machine and error record keeping (design.md §3.2).

The state machine never terminates the process itself: exiting is a launcher
decision governed by the startup.* flags. Errors are deduplicated by
(code, role) and keep their first_seen timestamp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger("sovereign.state")

STATES = (
    "initializing",
    "downloading",
    "compiling",
    "loading",
    "smoke_testing",
    "healthy",
    "degraded",
    "configuration_error",
    "runtime_error",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ErrorRecord:
    code: str
    message: str
    recoverable: bool
    role: str | None = None
    first_seen: str = field(default_factory=_now)

    def payload(self) -> dict:
        body: dict = {
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
            "first_seen": self.first_seen,
        }
        if self.role:
            body["role"] = self.role
        return body


class StateMachine:
    def __init__(self) -> None:
        self._state = "initializing"
        self._errors: dict[tuple[str, str | None], ErrorRecord] = {}
        self._listeners: list = []

    @property
    def state(self) -> str:
        return self._state

    def on_change(self, listener) -> None:
        self._listeners.append(listener)
        listener(self._state)

    def transition(self, state: str) -> None:
        if state not in STATES:
            raise ValueError(f"unknown runtime state: {state!r}")
        if state == self._state:
            return
        logger.info("state: %s -> %s", self._state, state)
        self._state = state
        for listener in self._listeners:
            listener(state)

    def record_error(
        self, code: str, message: str, *, recoverable: bool, role: str | None = None
    ) -> None:
        key = (code, role)
        if key in self._errors:
            self._errors[key].message = message
            return
        logger.warning("error recorded: %s (role=%s, recoverable=%s): %s", code, role, recoverable, message)
        self._errors[key] = ErrorRecord(code=code, message=message, recoverable=recoverable, role=role)

    def clear_errors(self, role: str | None = None) -> None:
        if role is None:
            self._errors.clear()
        else:
            self._errors = {key: rec for key, rec in self._errors.items() if key[1] != role}

    @property
    def errors(self) -> list[ErrorRecord]:
        return list(self._errors.values())

    def errors_payload(self) -> dict:
        return {"errors": [record.payload() for record in self.errors]}
