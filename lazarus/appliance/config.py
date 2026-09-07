"""runtime.yaml parsing and validation (design.md §12).

The JSON Schema in the monorepo (schemas/runtime-config.schema.json) is the
external contract; these models are the appliance's operational parser and
must stay in sync with it — the conformance harness enforces the external
side. A ConfigError here must lead to state configuration_error with the
control API still serving (§3.2), never a crash loop.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ConfigError(Exception):
    """Human-readable configuration failure; message is shown in /runtime/errors."""


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    task: Literal["generate", "embed", "rerank"] | None = None
    source: Literal["huggingface", "modelscope", "local"] | None = None
    model: str | None = None
    revision: str | None = None
    served_model_name: str | None = None
    max_model_len: int | None = Field(default=None, ge=1)
    priority: Literal["high", "normal", "low"] = "normal"
    memory_weight: int = Field(default=50, ge=1, le=100)
    max_concurrent_requests: int = Field(default=8, ge=1)
    pooling: Literal["last", "mean", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None
    throttle_when_generation_queue_above: int | None = Field(default=None, ge=0)
    enforce_eager: bool = Field(default=False, strict=True)
    accelerator_device_ids: list[str] = Field(default_factory=list, min_length=1, max_length=4)
    tensor_parallel_size: int = Field(default=1, ge=1, le=4)
    # Tool calling is on by default for generation: unset = infer the parser
    # from the model; "off" disables; any other value = explicit parser name.
    tool_call_parser: str | None = None
    # Same contract for reasoning separation (thinking → reasoning_content).
    reasoning_parser: str | None = None

    @field_validator("tensor_parallel_size", mode="before")
    @classmethod
    def tensor_size_is_numeric(cls, value):
        # JSON Schema integers include integral numbers such as 2.0, but
        # neither boolean values nor numeric strings are integers.
        if type(value) not in (int, float):
            raise ValueError("tensor_parallel_size must be an integer")
        return value

    def missing_load_fields(self) -> list[str]:
        if not self.enabled:
            return []
        return [
            name
            for name in ("source", "model", "served_model_name")
            if getattr(self, name) in (None, "")
        ]


class RuntimeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    listen_address: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    api_key_env: str | None = None
    profile: str = "cpu-x86_64"


class StartupSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smoke_test_on_start: bool = True
    remain_alive_on_configuration_error: bool = True
    fail_process_on_generation_error: bool = False
    fail_process_on_embedding_error: bool = False


class ObservabilitySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prometheus: bool = True
    structured_logs: bool = True
    otlp_endpoint: str | None = None


class PrivacySection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_logging: bool = False
    response_logging: bool = False
    full_trace: bool = False


class RolesSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generation: RoleConfig
    # Embeddings are optional. SovereignStack uses its dedicated
    # EmbeddingGemma service unless an operator selects a custom model.
    embedding: RoleConfig | None = None
    vision: RoleConfig | None = None
    audio: RoleConfig | None = None
    rerank: RoleConfig | None = None

    def items(self) -> list[tuple[str, RoleConfig]]:
        pairs = []
        for name in ("generation", "embedding", "vision", "audio", "rerank"):
            role = getattr(self, name)
            if role is not None:
                pairs.append((name, role))
        return pairs


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.2"]
    runtime: RuntimeSection = RuntimeSection()
    startup: StartupSection = StartupSection()
    roles: RolesSection
    observability: ObservabilitySection = ObservabilitySection()
    privacy: PrivacySection = PrivacySection()

    @property
    def api_key(self) -> str | None:
        if not self.runtime.api_key_env:
            return None
        return os.environ.get(self.runtime.api_key_env) or None

    def role(self, name: str) -> RoleConfig | None:
        return getattr(self.roles, name, None)

    def enabled_roles(self) -> dict[str, RoleConfig]:
        return {name: role for name, role in self.roles.items() if role.enabled}

    def alias_to_role(self) -> dict[str, str]:
        return {
            role.served_model_name: name
            for name, role in self.enabled_roles().items()
            if role.served_model_name
        }


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "<root>"
        lines.append(f"{location}: {err['msg']}")
    return "; ".join(lines)


def load_config(path: str | Path) -> RuntimeConfig:
    """Parse and validate runtime.yaml. Raises ConfigError with a message
    suitable for /runtime/errors."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"runtime config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"runtime config is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("runtime config must be a YAML mapping")
    try:
        config = RuntimeConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"runtime config invalid: {_format_validation_error(exc)}") from exc

    problems = []
    gpu_uuid = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
    for name, role in config.roles.items():
        if role.enabled:
            for field in role.missing_load_fields():
                problems.append(f"roles.{name}.{field} is required when the role is enabled")
        if name != "generation" and role.model_fields_set.intersection(
            {"accelerator_device_ids", "tensor_parallel_size"}
        ):
            problems.append(f"roles.{name} cannot define generation accelerator placement")
    generation = config.roles.generation
    has_devices = "accelerator_device_ids" in generation.model_fields_set
    has_tensor_size = "tensor_parallel_size" in generation.model_fields_set
    if has_devices != has_tensor_size:
        problems.append(
            "roles.generation accelerator_device_ids and tensor_parallel_size must be defined together"
        )
    if (has_devices or has_tensor_size) and config.runtime.profile not in (
        "cuda-x86_64",
        "cuda-arm64-dgx-spark",
    ):
        problems.append("roles.generation accelerator placement requires a CUDA runtime profile")
    if generation.tensor_parallel_size not in (1, 2, 4):
        problems.append("roles.generation tensor_parallel_size must be one of 1, 2, or 4")
    if generation.accelerator_device_ids:
        if len(generation.accelerator_device_ids) != generation.tensor_parallel_size:
            problems.append(
                "roles.generation accelerator_device_ids must match tensor_parallel_size"
            )
        if len(set(generation.accelerator_device_ids)) != len(generation.accelerator_device_ids):
            problems.append("roles.generation accelerator_device_ids must be unique")
        if any(not gpu_uuid.fullmatch(value) for value in generation.accelerator_device_ids):
            problems.append(
                "roles.generation accelerator_device_ids must contain canonical NVIDIA GPU UUIDs"
            )
    if problems:
        raise ConfigError("runtime config invalid: " + "; ".join(problems))
    return config
