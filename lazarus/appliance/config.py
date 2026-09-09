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
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class ConfigError(Exception):
    """Human-readable configuration failure; message is shown in /runtime/errors."""


SLIMSERVE_COMMIT = "44a7d1a21851c7164c098c93fbc1baa12ab99847"
QUIXICORE_CUDA_COMMIT = "08780aaa22cdc2d144b6beacb24df953134b34be"
QUIXICORE_METAL_COMMIT = "71a08cd4cbcdc622ce31b3fc91e1f505e144b516"


class SlimServeFile(BaseModel):
    """One Control-verified member; never an arbitrary launch path."""

    model_config = ConfigDict(extra="forbid")
    file: str = Field(min_length=1, max_length=192, pattern=r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
    size_bytes: int = Field(strict=True, gt=0, le=1024**4)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("file")
    @classmethod
    def bounded_member(cls, value: str) -> str:
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in value.split("/")):
            raise ValueError("artifact member must not contain traversal or hidden components")
        if len(value.split("/")) > 4:
            raise ValueError("artifact member is too deeply nested")
        return value


class SlimServeArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["model", "drafter", "tokenizer"]
    repository: str = Field(max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    files: list[SlimServeFile] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def unique_members(self):
        if len({entry.file for entry in self.files}) != len(self.files):
            raise ValueError("artifact members must be unique")
        return self


class SlimServeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_commit: Literal["44a7d1a21851c7164c098c93fbc1baa12ab99847"]
    profile_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    variant: Literal["a100", "rtx3090", "metal"]
    quant: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    artifacts: list[SlimServeArtifact] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_roles(self):
        roles = [artifact.role for artifact in self.artifacts]
        if len(set(roles)) != len(roles) or "model" not in roles:
            raise ValueError("SlimServe requires one model closure and unique artifact roles")
        if sum(entry.size_bytes for artifact in self.artifacts for entry in artifact.files) > 2 * 1024**4:
            raise ValueError("SlimServe artifact closure exceeds 2 TiB")
        return self


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    engine: Literal["vllm", "slimserve"] = "vllm"
    engine_profile_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    slimserve: SlimServeConfig | None = None
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

    @model_validator(mode="after")
    def engine_contract(self):
        if self.engine == "slimserve":
            if not self.slimserve or not self.engine_profile_id:
                raise ValueError("SlimServe requires engine_profile_id and its sealed configuration")
            if self.source != "local" or not self.model or not Path(self.model).is_absolute():
                raise ValueError("SlimServe requires a prepared local model directory")
            if not self.revision or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
                raise ValueError("SlimServe requires an immutable model revision")
            if self.max_model_len is None or self.max_model_len > 1048576:
                raise ValueError("SlimServe requires bounded context at most 1048576")
            if self.max_concurrent_requests > 256:
                raise ValueError("SlimServe concurrency must not exceed 256")
            if not self.served_model_name or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", self.served_model_name):
                raise ValueError("SlimServe requires a bounded serving alias")
        elif self.slimserve is not None:
            raise ValueError("SlimServe configuration requires engine slimserve")
        return self

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

    @model_validator(mode="after")
    def slimserve_placement(self):
        generation = self.roles.generation
        for name, role in self.roles.items():
            if name != "generation" and role.model_fields_set.intersection({"engine", "engine_profile_id", "slimserve"}):
                raise ValueError("engine selection is generation-only")
        if generation.engine != "slimserve":
            return self
        if any(role.enabled for name, role in self.roles.items() if name != "generation"):
            raise ValueError("SlimServe uses the independent product embedding service")
        if not generation.enabled:
            raise ValueError("SlimServe generation must be enabled")
        if generation.slimserve.variant == "metal":
            if self.runtime.profile != "metal-arm64" or generation.tensor_parallel_size != 1 or generation.accelerator_device_ids:
                raise ValueError("Metal SlimServe requires native Metal and exactly one device")
        elif self.runtime.profile != "cuda-x86_64" or not generation.accelerator_device_ids:
            raise ValueError("CUDA SlimServe requires explicit managed CUDA placement")
        elif generation.tensor_parallel_size not in (1, 2, 4) or len(generation.accelerator_device_ids) != generation.tensor_parallel_size:
            raise ValueError("SlimServe device count must equal managed tensor parallelism")
        elif not {"accelerator_device_ids", "tensor_parallel_size"} <= generation.model_fields_set:
            raise ValueError("SlimServe managed placement fields must be explicit together")
        elif len(set(generation.accelerator_device_ids)) != generation.tensor_parallel_size or any(
            not re.fullmatch(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value)
            for value in generation.accelerator_device_ids
        ):
            raise ValueError("SlimServe requires unique canonical NVIDIA UUIDs")
        return self

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
