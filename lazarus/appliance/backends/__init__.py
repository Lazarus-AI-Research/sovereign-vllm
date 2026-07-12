"""Engine backends behind the appliance layer.

The appliance owns the contract (state machine, health, manifest, routing);
a backend owns inference. Selection via SOVEREIGN_ENGINE_BACKEND:

  vllm  in-process vLLM engine(s) — the product backend (default)
  fake  deterministic canned engine for appliance tests and CI

The fake backend exists so every piece of appliance logic is testable on
machines that cannot run vLLM; it is never shipped as a product profile.
"""

from __future__ import annotations

import os

from lazarus.appliance.backends.base import BackendStartError, EngineBackend, RoleInfo
from lazarus.appliance.backends.fake import FakeBackend

__all__ = ["BackendStartError", "EngineBackend", "FakeBackend", "RoleInfo", "select_backend"]


def select_backend() -> EngineBackend:
    name = os.environ.get("SOVEREIGN_ENGINE_BACKEND", "vllm").lower()
    if name == "fake":
        return FakeBackend()
    if name == "vllm":
        from lazarus.appliance.backends.vllm_engine import VllmBackend

        return VllmBackend()
    if name == "agent":
        from lazarus.appliance.backends.agent import AgentBackend

        return AgentBackend()
    raise ValueError(f"unknown SOVEREIGN_ENGINE_BACKEND: {name!r}")
