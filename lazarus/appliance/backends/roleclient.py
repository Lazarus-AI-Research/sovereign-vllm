"""Shared OpenAI-surface plumbing for backends that expose per-role HTTP
clients (in-process vLLM apps, the Metal host agent). The API layer forwards
live traffic raw via role_client(); these methods serve the startup smoke
test and probes."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class RoleClientMixin:
    def role_client(self, role: str) -> httpx.AsyncClient | None:  # pragma: no cover
        raise NotImplementedError

    async def _post(self, role: str, path: str, body: dict) -> dict:
        client = self.role_client(role)
        if client is None:
            raise RuntimeError(f"{role} role is not loaded")
        resp = await client.post(path, json=body)
        if resp.status_code != 200:
            raise RuntimeError(f"{path}: {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def chat_completion(self, body: dict) -> dict:
        return await self._post("generation", "/v1/chat/completions", body)

    async def chat_completion_stream(self, body: dict) -> AsyncIterator:
        client = self.role_client("generation")
        if client is None:
            raise RuntimeError("generation role is not loaded")
        async with client.stream(
            "POST", "/v1/chat/completions", json={**body, "stream": True}
        ) as resp:
            async for line in resp.aiter_lines():
                if line:
                    yield line + "\n\n"

    async def completion(self, body: dict) -> dict:
        return await self._post("generation", "/v1/completions", body)

    async def embeddings(self, body: dict) -> dict:
        return await self._post("embedding", "/v1/embeddings", body)
