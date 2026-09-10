"""Agent configuration (written by the installer)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lazarus.appliance.config import RuntimeConfig


# Keep this private native-file contract aligned with Control's manifest decoder.
# The bound includes /models/; host-root spelling is not part of the identity.
MAX_NATIVE_MODEL_IDENTITY_BYTES = 512


def valid_native_model_identity(value: object) -> bool:
    if not isinstance(value, str) or len(value) > MAX_NATIVE_MODEL_IDENTITY_BYTES or not value.startswith("/models/"):
        return False
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if len(encoded) > MAX_NATIVE_MODEL_IDENTITY_BYTES or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in value):
        return False
    return all(component not in {b"", b".", b".."} and len(component) <= 255 for component in encoded[8:].split(b"/"))


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

    @field_validator("model_path")
    @classmethod
    def canonical_model_path(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or str(path) != value or value.startswith("//") or ".." in path.parts:
            raise ValueError("model_path must be a canonical absolute path")
        # Root-relative bounds and managed-file existence require the Agent's
        # resolved model root and are checked before the child is started.
        return value


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str = "127.0.0.1"
    port: int = Field(default=9100, ge=1, le=65535)
    token_env: str = "SOVEREIGN_AGENT_TOKEN"
    llama_server: str = "llama-server"
    roles: dict[str, AgentRole]
    # Installer-owned llama roles remain available for an explicit engine cutback.
    # Persist the private Runtime path, never a caller-selected host path.
    slimserve_generation: RuntimeConfig | None = None


def load_agent_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentConfig.model_validate(raw)
