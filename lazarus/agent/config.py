"""Agent configuration (written by the installer)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    # Extra arguments are retained only for legacy llama.cpp roles.
    args: list[str] = []
    # The managed composite uses the product EmbeddingGemma executable, never
    # a llama.cpp embedding substitution.
    engine: Literal["llama.cpp", "embeddinggemma"] = "llama.cpp"

    @model_validator(mode="after")
    def engine_contract(self):
        if self.engine == "embeddinggemma" and self.args:
            raise ValueError("embeddinggemma does not accept llama.cpp arguments")
        return self

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
    hardware_profile: Literal["metal-arm64"] = "metal-arm64"
    # Generated managed configs name only the reviewed installed executables.
    # Their paths are private distribution-owned values, never API input.
    embeddinggemma: str = "embeddinggemma"
    # Installer-owned llama roles remain available for an explicit engine cutback.
    # Persist the private Runtime path, never a caller-selected host path.
    slimserve_generation: RuntimeConfig | None = None
    # These identities are immutable inputs from Control's durable binding.
    # Legacy singleton configurations intentionally omit both.
    runtime_instance_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )
    deployment_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    )

    @model_validator(mode="after")
    def managed_instance_identity(self):
        if (self.runtime_instance_id is None) != (self.deployment_id is None):
            raise ValueError("runtime_instance_id and deployment_id must be configured together")
        if self.runtime_instance_id is not None:
            if self.runtime_instance_id == self.deployment_id:
                raise ValueError("runtime_instance_id and deployment_id must be distinct")
            generation = self.roles.get("generation")
            embedding = self.roles.get("embedding")
            if set(self.roles) != {"generation", "embedding"} or generation is None or embedding is None:
                raise ValueError("managed instance requires one generation and one embedding role")
            if generation.engine != "llama.cpp" or embedding.engine != "embeddinggemma":
                raise ValueError("managed instance requires llama.cpp generation and embeddinggemma embedding")
        return self


def load_agent_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentConfig.model_validate(raw)
