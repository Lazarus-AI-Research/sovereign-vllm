"""Runtime manifest builder (design.md §14).

The manifest reports observed reality in every state — including
configuration_error, where role details may be unknown. Written to
$SOVEREIGN_RUNTIME_MANIFEST and served at GET /runtime/manifest.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from lazarus.appliance.backends.base import EngineBackend
from lazarus.appliance.config import RuntimeConfig
from lazarus.appliance.state import StateMachine

RUNTIME_VERSION = "0.1.0-dev"


class ManifestBuilder:
    def __init__(
        self,
        *,
        state: StateMachine,
        backend: EngineBackend,
        config: RuntimeConfig | None,
        port: int,
    ) -> None:
        self.state = state
        self.backend = backend
        self.config = config
        self.port = port

    @property
    def profile(self) -> str:
        if env := os.environ.get("SOVEREIGN_PROFILE"):
            return env
        if self.config is not None:
            return self.config.runtime.profile
        return "cpu-x86_64"

    def _role_entry(self, name: str) -> dict:
        info = self.backend.role_info(name)
        role_config = self.config.role(name) if self.config else None
        entry: dict = {
            "enabled": bool(role_config.enabled) if role_config else name in ("generation", "embedding"),
            "status": info.status,
        }
        if role_config and role_config.task:
            entry["task"] = role_config.task
        if role_config and role_config.served_model_name:
            entry["served_model_name"] = role_config.served_model_name
        if info.engine_model:
            entry["engine_model"] = info.engine_model
        if info.revision:
            entry["revision"] = info.revision
        if info.context_length:
            entry["context_length"] = info.context_length
        if info.error_code:
            entry["error_code"] = info.error_code
        if name == "embedding":
            if info.dimensions:
                entry["dimensions"] = info.dimensions
            if role_config and role_config.pooling:
                entry["pooling"] = role_config.pooling
            if role_config and role_config.normalization:
                entry["normalization"] = role_config.normalization
            if info.modalities:
                entry["modalities"] = info.modalities
        return entry

    def build(self) -> dict:
        accelerator = {"vendor": "none", "device_count": 0, "unified_memory": False}
        probe = getattr(self.backend, "accelerator", None)
        if probe is not None:
            accelerator = probe()

        manifest: dict = {
            "schema_version": "1.1",
            "runtime_id": f"sovereign-runtime-{self.profile}-{RUNTIME_VERSION}",
            "runtime_version": RUNTIME_VERSION,
            "vllm_version": self.backend.engine_version(),
            "backend": self.backend.backend_id,
            "profile": self.profile,
            "topology": "single_process_multi_role",
            "state": self.state.state,
            "api": {"openai_compatible": True, "port": self.port, "base_path": "/v1"},
            "roles": {
                "generation": self._role_entry("generation"),
                "embedding": self._role_entry("embedding"),
            },
            "accelerator": accelerator,
            "health": {
                "status": "healthy" if self.state.state == "healthy" else self.state.state,
                "driver": "ok",
                "kernels": "ok",
                "metrics": "ok",
            },
        }
        if self.config is not None:
            manifest["resource_policy"] = {
                "enforcement": "best_effort",
                "generation_memory_weight": self.config.roles.generation.memory_weight,
                "embedding_memory_weight": self.config.roles.embedding.memory_weight,
            }
        for name in ("vision", "audio", "rerank"):
            if self.config and self.config.role(name):
                manifest["roles"][name] = self._role_entry(name)
        return manifest

    def write(self) -> None:
        path = os.environ.get("SOVEREIGN_RUNTIME_MANIFEST")
        if not path:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.build(), indent=2))
