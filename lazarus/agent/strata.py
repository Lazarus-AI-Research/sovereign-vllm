"""Strata serves Qwen3.8-Flash-Next with its experts in memory and only the
busiest on the GPU. A deployment names the files Strata builds its pack
from: the model's two GGUF shards and the draft layer's tensors, read out of
Qwen's checkpoint, each pinned by its checksum. The agent builds the pack and
the draft layer from them once per snapshot (strata_prepare.py, in Strata's
own Python, offline), writes the run settings at every start, and runs
Strata's server on loopback behind its guard like every other."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, field_validator

if TYPE_CHECKING:
    from lazarus.agent.deployments import AgentDeployment
    from lazarus.agent.server import Agent

STRATA = "strata"
# Loading maps tens of gigabytes, and a first start builds the pack and the
# draft layer before the server starts at all.
STRATA_READY_TIMEOUT = 1800.0
# The most requests one Strata process decodes together.
STRATA_MAX_REQUESTS = 8
PREPARE_TIMEOUT = 3600.0
TENSOR_NAME = r"^mtp\.[A-Za-z0-9_.]{1,200}$"


class Tensor(BaseModel):
    """The checkpoint tensor a draft-layer file holds."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: Literal["BF16", "F16", "F32"]
    shape: list[int]

    @field_validator("name")
    @classmethod
    def mtp_tensor(cls, value: str) -> str:
        if not re.fullmatch(TENSOR_NAME, value):
            raise ValueError("a draft-layer tensor is one of the checkpoint's mtp tensors")
        return value

    @field_validator("shape")
    @classmethod
    def positive(cls, value: list[int]) -> list[int]:
        if not value or any(size < 1 for size in value):
            raise ValueError("a tensor's shape is one or more positive sizes")
        return value


class StrataPack(BaseModel):
    """The files Strata's pack is built from by their path in the snapshot
    directory, each with its sha256, and the tensor each draft-layer file
    holds."""

    model_config = ConfigDict(extra="forbid")

    files: dict[str, str]
    tensors: dict[str, Tensor]

    @field_validator("files")
    @classmethod
    def pinned_files(cls, files: dict[str, str]) -> dict[str, str]:
        for name, digest in files.items():
            parts = PurePosixPath(name).parts
            if not parts or name.startswith("/") or str(PurePosixPath(name)) != name or any(part in (".", "..") for part in parts):
                raise ValueError("Strata's files are named by canonical relative paths")
            if any(ord(char) < 32 for char in name):
                raise ValueError("Strata's file names hold no control characters")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError("Strata's files are pinned by their sha256")
        if sum(name.endswith(".gguf") for name in files) != 2:
            raise ValueError("Strata's files hold the model's two GGUF shards")
        return {name: digest.lower() for name, digest in files.items()}

    @field_validator("tensors")
    @classmethod
    def named_once(cls, tensors: dict[str, Tensor]) -> dict[str, Tensor]:
        names = [tensor.name for tensor in tensors.values()]
        if len(set(names)) != len(names):
            raise ValueError("each draft-layer tensor is held by one file")
        return tensors

    def digest(self) -> str:
        """The snapshot's own checksum, as Control computes it."""
        return hashlib.sha256("".join(f"{name}\t{self.files[name]}\n" for name in sorted(self.files)).encode()).hexdigest()


def strata_options(kind: str, strata: StrataPack | None, others: list, components: dict, projector: str | None, budget: int | None, sha256: str) -> None:
    if strata is None:
        return
    if kind != "generation":
        raise ValueError("Strata serves language model deployments only")
    if any(other is not None for other in others) or components or projector is not None:
        raise ValueError("a Strata deployment is its files alone")
    if budget is not None:
        raise ValueError("Strata takes no thinking budget")
    if set(strata.tensors) - set(strata.files):
        raise ValueError("every draft-layer tensor is one of Strata's files")
    if sha256.lower() != strata.digest():
        raise ValueError("a Strata deployment's sha256 is its files' own checksum")


def strata_engine(directory: str) -> dict | None:
    """Strata as the manifest names it: installed when its server, its Python
    and its engine are, at the engine's version."""
    if not directory:
        return None
    root = Path(directory)
    if not all(os.access(root / path, os.X_OK) for path in ("python/bin/python3", "engine/strata")) or not (root / "serve" / "server.py").is_file():
        return None
    try:
        version = json.loads((root / "engine" / "BUILD.json").read_text())["version"]
    except (OSError, ValueError, KeyError):
        version = "unknown"
    return {"name": STRATA, "version": version, "adapter": "metal-host-agent", "variants": ["metal-arm64"]}


def strata_home(agent: Agent) -> Path:
    home = agent.config_path.parent if agent.config_path else Path.home() / ".sovereign"
    return home / "strata"


def unified_memory_gb() -> float:
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout) / 1e9


# What the run settings are written from: a deployment's files, window,
# requests and thinking, and nothing that changes from one start to the next.
SETTINGS_FIELDS = ("model_path", "sha256", "strata", "context_length", "served_model_name", "requests", "thinking")


def settings_path(agent: Agent, name: str, record: dict) -> Path:
    """Where the run settings for this deployment as recorded are kept: one
    file for each set of what they are written from, so settings written for
    a replacement never stand in for the deployment it would have replaced."""
    written_from = json.dumps({field: record.get(field) for field in SETTINGS_FIELDS}, sort_keys=True)
    return strata_home(agent) / "deployments" / f"{name}.{hashlib.sha256(written_from.encode()).hexdigest()[:16]}.json"


def prepare_strata(agent: Agent, name: str, record: dict) -> Path:
    """The pack and the draft layer built for the record's files where they
    are not yet, and the run settings written from the record. Called off
    the event loop; a first build takes minutes, a later one seconds."""
    root = Path(agent.config.strata)
    settings = settings_path(agent, name, record)
    settings.parent.mkdir(parents=True, exist_ok=True)
    request = settings.with_suffix(".request.json")
    request.write_text(json.dumps({field: record.get(field) for field in SETTINGS_FIELDS}))
    command = [
        str(root / "python" / "bin" / "python3"), "-I", str(Path(__file__).with_name("strata_prepare.py")),
        "--strata", str(root), "--request", str(request), "--out", str(strata_home(agent) / "packs" / record["sha256"]),
        "--memory-gb", f"{unified_memory_gb():.1f}", "--settings", str(settings),
    ]
    with open(strata_home(agent) / "deployments" / f"{name}.prepare.log", "ab") as log:
        subprocess.run(command, stdout=log, stderr=log, check=True, timeout=PREPARE_TIMEOUT)
    return settings


# Strata's server on loopback, offline, holding as many requests at once as
# the deployment's memory does; its key is the agent's, given through its
# environment.
def strata_command(agent: Agent, deployment: AgentDeployment, settings: Path) -> list[str]:
    root = Path(agent.config.strata)
    return [
        str(root / "python" / "bin" / "python3"), "-I", str(root / "serve" / "server.py"),
        "--engine", "strata", "--config", str(settings),
        "--host", "127.0.0.1", "--port", str(deployment.port),
        "--parallel", str(deployment.requests), "--offline",
    ]
