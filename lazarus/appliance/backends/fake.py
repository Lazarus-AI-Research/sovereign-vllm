"""Deterministic canned backend for appliance tests and CI (never shipped).

Env knobs:
  SOVEREIGN_FAKE_FAIL_ROLE   role name whose load fails (tests degraded path)
  SOVEREIGN_FAKE_DELAY       seconds per simulated state (default 0.05)
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import time
from collections.abc import AsyncIterator, Callable

from lazarus.appliance.backends.base import EngineBackend, RoleInfo
from lazarus.appliance.config import RuntimeConfig

_DIM = 384


def _embed(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(_DIM)]
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


class FakeBackend(EngineBackend):
    backend_id = "mock"

    def __init__(self) -> None:
        self._roles: dict[str, RoleInfo] = {}
        self._aliases: dict[str, str] = {}

    async def start(self, config: RuntimeConfig, on_state: Callable[[str], None]) -> None:
        delay = float(os.environ.get("SOVEREIGN_FAKE_DELAY", "0.05"))
        fail_role = os.environ.get("SOVEREIGN_FAKE_FAIL_ROLE")
        on_state("downloading")
        await asyncio.sleep(delay)
        on_state("loading")
        for name, role in config.roles.items():
            if role is None:
                continue
            if not role.enabled:
                self._roles[name] = RoleInfo(status="disabled")
                continue
            await asyncio.sleep(delay)
            if name == fail_role:
                self._roles[name] = RoleInfo(status="unhealthy", error_code="MODEL_LOAD_FAILED")
                continue
            info = RoleInfo(
                status="healthy",
                engine_model=role.model,
                revision=role.revision or "fake",
                context_length=role.max_model_len or 32768,
            )
            if name == "embedding":
                info.dimensions = _DIM
                info.modalities = ["text"]
            self._roles[name] = info
            if role.served_model_name:
                self._aliases[name] = role.served_model_name

    async def shutdown(self) -> None:
        return

    def role_info(self, role: str) -> RoleInfo:
        return self._roles.get(role, RoleInfo(status="disabled"))

    def engine_version(self) -> str:
        return "fake"

    async def chat_completion(self, body: dict) -> dict:
        content = "This is the Sovereign appliance layer (fake backend). All systems nominal."
        return {
            "id": "chatcmpl-fake",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 14, "total_tokens": 22},
        }

    async def chat_completion_stream(self, body: dict) -> AsyncIterator[dict]:
        created = int(time.time())
        content = "This is the Sovereign appliance layer (fake backend)."
        deltas = [{"role": "assistant"}, {"content": content[:25]}, {"content": content[25:]}]
        finishes = [None, None, None, "stop"]
        for delta, finish in zip(deltas + [{}], finishes):
            yield {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": created,
                "model": body["model"],
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

    async def completion(self, body: dict) -> dict:
        return {
            "id": "cmpl-fake",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body["model"],
            "choices": [{"index": 0, "text": " a local-first AI appliance.", "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
        }

    async def embeddings(self, body: dict) -> dict:
        if "messages" in body:  # extended multimodal schema: one item per request
            inputs = [str(body["messages"])]
        else:
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        data = [
            {"object": "embedding", "index": i, "embedding": _embed(str(text))}
            for i, text in enumerate(inputs)
        ]
        return {
            "object": "list",
            "data": data,
            "model": body["model"],
            "usage": {"prompt_tokens": len(inputs) * 4, "total_tokens": len(inputs) * 4},
        }
