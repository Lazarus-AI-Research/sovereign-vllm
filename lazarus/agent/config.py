"""Agent configuration (written by the installer, extended by Control)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lazarus.agent.deployments import AgentDeployment

logger = logging.getLogger("sovereign.agent.config")

# Keep this private native-file contract aligned with Control's manifest decoder.
# The bound includes /models/; host-root spelling is not part of the identity.
MAX_NATIVE_MODEL_IDENTITY_BYTES = 512

# Keys a configuration written by an earlier agent may still carry: the fixed
# roles, the embeddinggemma executable the embedding role used, the SlimServe
# generation path and the managed-instance identity. Every process the agent
# runs is a deployment now, so they are read past and dropped at the next save.
RETIRED_KEYS = ("roles", "embeddinggemma", "slimserve_generation", "runtime_instance_id", "deployment_id")


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


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen: str = "127.0.0.1"
    port: int = Field(default=9100, ge=1, le=65535)
    token_env: str = "SOVEREIGN_AGENT_TOKEN"
    llama_server: str = "llama-server"
    # Every served model: created by Control while the appliance runs and
    # persisted so a restarted agent serves them again.
    deployments: dict[str, AgentDeployment] = {}
    hardware_profile: Literal["metal-arm64"] = "metal-arm64"

    @model_validator(mode="before")
    @classmethod
    def drop_retired_keys(cls, data):
        if not isinstance(data, dict):
            return data
        retired = [key for key in RETIRED_KEYS if key in data]
        if retired:
            logger.warning("ignoring retired agent configuration keys: %s", ", ".join(retired))
            data = {key: value for key, value in data.items() if key not in RETIRED_KEYS}
        return data


def load_agent_config(path: str | Path) -> AgentConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return AgentConfig.model_validate(raw)
