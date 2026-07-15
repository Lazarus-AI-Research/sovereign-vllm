"""Agent configuration (written by the installer)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AgentRole(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_path: str
    port: int = Field(ge=1, le=65535)
    context_length: int | None = None
    # Immutable upstream revision for runtime-manifest traceability.
    revision: str | None = None
    # multimodal projector (GGUF) for omni models, passed as --mmproj
    mmproj_path: str | None = None
    # extra llama-server flags, e.g. ["--embedding", "--pooling", "last"]
    args: list[str] = []


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str = "127.0.0.1"
    port: int = Field(default=9100, ge=1, le=65535)
    token_env: str = "SOVEREIGN_AGENT_TOKEN"
    llama_server: str = "llama-server"
    roles: dict[str, AgentRole]


def load_agent_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentConfig.model_validate(raw)
