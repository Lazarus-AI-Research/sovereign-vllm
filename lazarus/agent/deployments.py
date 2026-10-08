"""Deployments: one supervised process per served model, created and removed
by Sovereign Control through the admin API and reached through
``/deployments/{id}/v1``. An LLM or embedding model is a llama-server child;
an image model is a stable-diffusion.cpp server child. Each has its own port,
admission gate and lifecycle, so none restarts another."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shlex
import socket
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lazarus.agent import log_level

if TYPE_CHECKING:
    from lazarus.agent.server import Agent, ServerProcess

DEPLOYMENT_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
# The agent listens on 9100 and the retired fixed roles held 9101 and 9102;
# deployments take a block above both, so an agent upgraded in place never
# collides with a role process still winding down.
DEPLOYMENT_PORTS = range(9110, 9200)
# A request that has not finished in this long is not worth keeping a
# replacement waiting for.
IDLE_TIMEOUT = 600.0
READY_TIMEOUT = 300.0
# The manifest is read by clients with a five-second budget; every deployment
# is probed at once and each probe is cut off well inside it, so a hung
# deployment can never make the manifest fail for the ones beside it.
OBSERVE_TIMEOUT = 1.5

ALLOWED_PATHS = {
    "generation": {"chat/completions", "completions", "models"},
    "embedding": {"embeddings", "models"},
    "image": {"images/generations", "models"},
    "transcription": {"audio/transcriptions"},
    "speech": {"audio/speech"},
    # A video is made as a job, asked after by its id, and fetched when done.
    "video": {"videos"},
}
VIDEO_PATH = re.compile(r"^videos/(video_[A-Za-z0-9_-]{1,256})(/content)?$")
KINDS = tuple(ALLOWED_PATHS)
# The kinds served without a context window: a diffusion, transcription or
# speech model has no prompt to size.
NO_CONTEXT = ("image", "transcription", "speech", "video")

# The files an image model is served with beside its diffusion weights, by
# the flag stable-diffusion.cpp takes them under. A model that needs none of
# them (a single-file checkpoint) names none.
IMAGE_COMPONENTS = {"clip_l": "--clip_l", "t5xxl": "--t5xxl", "vae": "--vae"}
IMAGE_SAMPLERS = ("euler", "euler_a", "heun", "dpm2", "dpm++2m", "lcm")
# sd-server loads a LoRA a prompt names from its LoRA directory, which is its
# working directory unless given; launchd starts the agent at the root of
# the disk. An empty directory leaves a prompt nothing to load, and the
# server nothing to scan when it lists what it has.
LORA_DIRECTORY = "/var/empty"
# The files a video model is served with beside its diffusion weights: an
# autoencoder, a text encoder (a T5 for Wan, a language model for LTX-2 and
# MiniMax-H3), and for models that make sound an audio autoencoder, and for
# LTX-2 the connectors between its text encoder and its transformer. LTX's
# spatial upscaler is found by name in the directory the server is given.
VIDEO_COMPONENTS = {
    "vae": "--vae", "t5xxl": "--t5xxl", "llm": "--llm", "audio_vae": "--audio-vae",
    "embeddings_connectors": "--embeddings-connectors", "spatial_upscaler": "--hires-upscalers-dir",
}
DIFFUSION = ("image", "video")
# The types stable-diffusion.cpp converts a text encoder to as it loads.
TEXT_ENCODER_TYPES = ("q8_0", "f16")
VIDEO_SIZE = r"^[1-9][0-9]{2,3}x[1-9][0-9]{2,3}$"
# A weight file the agent loads: GGUF for llama.cpp, GGUF or safetensors for
# stable-diffusion.cpp, whose text encoders and autoencoder ship as either.
WEIGHT_SUFFIXES = (".gguf", ".safetensors")

# The voice configuration piper reads beside a speech model's weights: the
# one component a speech deployment names, and it must be the file piper
# finds by name.
SPEECH_COMPONENTS = {"config"}
# The language a transcription deployment listens for; "auto" lets the
# model detect it.
LANGUAGE = r"^(auto|[a-z]{2,3})$"

# Where each kind's server says it is up. llama-server answers /health once
# its model is loaded; sd-server listens only once its model is loaded and
# answers the models listing (its capabilities route rescans directories on
# every call); whisper-server answers /health with 503 while
# it loads; piper's server listens only once its voice is loaded.
HEALTH_PATHS = {
    "generation": "/health", "embedding": "/health", "image": "/v1/models",
    "transcription": "/health", "speech": "/voices", "video": "/v1/models",
}
ENGINES = {
    "generation": "llama.cpp", "embedding": "llama.cpp", "image": "stable-diffusion.cpp",
    "transcription": "whisper.cpp", "speech": "piper", "video": "stable-diffusion.cpp",
}

# SlimServe serves a language model from one of its profiles: a model,
# quantization and engine settings tuned and measured together for this
# hardware, which it does not let a caller vary. It reads its files from a
# directory laid out as its profile list names them, and opens its port only
# once the model is loaded, which for a large model takes many minutes.
SLIMSERVE = "slimserve"
# mlx-lm serves a language model from its MLX snapshot: a repository's
# files, each pinned by its checksum, in one directory.
MLX = "mlx-lm"
SLIMSERVE_FILES = ("model", "projector", "drafter")
SLIMSERVE_READY_TIMEOUT = 1800.0
PROFILE_ID = r"^[a-z0-9][a-z0-9.-]{0,63}$"
PROFILE_QUANT = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$"
# The largest window the agent accepts for llama-server and for a SlimServe
# profile: a million tokens (GLM-5.3-Flash's, DeepSeek V4's). Control sizes
# each deployment's window to its model and memory.
LLAMA_WINDOW_LIMIT = 1048576
LLAMA_DEFAULT_WINDOW = 8192
SLIMSERVE_WINDOW_LIMIT = 1048576


# What each kind's loader accepts: GGUF for llama-server, GGUF or safetensors
# for stable-diffusion.cpp, ggml for whisper-server, ONNX for piper.
def weight_suffixes(kind: str) -> tuple[str, ...]:
    return {"image": WEIGHT_SUFFIXES, "video": WEIGHT_SUFFIXES, "transcription": (".bin",), "speech": (".onnx",)}.get(kind, (".gguf",))


def component_suffixes(kind: str) -> tuple[str, ...]:
    return (".json",) if kind == "speech" else weight_suffixes(kind)


class Component(BaseModel):
    """One named file an image deployment loads beside its diffusion model."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str

    @field_validator("path")
    @classmethod
    def canonical_path(cls, value: str) -> str:
        from pathlib import Path

        path = Path(value)
        if not path.is_absolute() or str(path) != value or value.startswith("//") or ".." in path.parts:
            raise ValueError("model paths must be canonical absolute paths")
        return value


class SlimServeProfile(BaseModel):
    """The SlimServe profile a language model deployment is served from, and
    where SlimServe looks for each of the deployment's files: its path under
    SlimServe's model directory, naming the weights, the projector or the
    drafter."""

    model_config = ConfigDict(extra="forbid")

    profile: str = Field(pattern=PROFILE_ID)
    quant: str = Field(pattern=PROFILE_QUANT)
    layout: dict[str, Literal[SLIMSERVE_FILES]]

    @field_validator("layout")
    @classmethod
    def relative_paths(cls, layout: dict[str, str]) -> dict[str, str]:
        from pathlib import PurePosixPath

        for name in layout:
            parts = PurePosixPath(name).parts
            if not parts or name.startswith("/") or str(PurePosixPath(name)) != name or any(part in (".", "..") for part in parts):
                raise ValueError("a SlimServe layout names each file by a canonical relative path")
        # Every link has a place of its own: none inside another's path, and
        # none differing only by case, which the Mac's disk takes as the same.
        folded = sorted(name.casefold() for name in layout)
        for name, following in zip(folded, folded[1:]):
            if following == name or following.startswith(name + "/"):
                raise ValueError("a SlimServe layout names each file at a place of its own")
        files = list(layout.values())
        if files.count("model") != 1 or any(files.count(file) > 1 for file in SLIMSERVE_FILES):
            raise ValueError("a SlimServe layout names the weights once, and the projector and drafter at most once each")
        return layout


class MLXSnapshot(BaseModel):
    """An MLX model's files by their path in its snapshot directory, each
    with its sha256: its configuration, tokenizer and safetensors weights."""

    model_config = ConfigDict(extra="forbid")

    files: dict[str, str]

    @field_validator("files")
    @classmethod
    def pinned_files(cls, files: dict[str, str]) -> dict[str, str]:
        from pathlib import PurePosixPath

        for name, digest in files.items():
            parts = PurePosixPath(name).parts
            if not parts or name.startswith("/") or str(PurePosixPath(name)) != name or any(part in (".", "..") for part in parts):
                raise ValueError("an MLX snapshot names each file by a canonical relative path")
            if any(ord(char) < 32 for char in name):
                raise ValueError("an MLX snapshot's file names hold no control characters")
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError("an MLX snapshot pins each file by its sha256")
        if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
            raise ValueError("an MLX snapshot holds its config.json and safetensors weights")
        return {name: digest.lower() for name, digest in files.items()}

    def digest(self) -> str:
        """The snapshot's own checksum: its files' paths and checksums, in
        order, as Control computes it."""
        return hashlib.sha256("".join(f"{name}\t{self.files[name]}\n" for name in sorted(self.files)).encode()).hexdigest()


# An MLX deployment is a language model served from its snapshot alone: no
# projector, no components, no SlimServe profile, and no thinking budget,
# which mlx-lm does not take.
def mlx_options(kind: str, mlx: MLXSnapshot | None, slimserve, components: dict, projector: str | None, budget: int | None, sha256: str) -> None:
    if mlx is None:
        return
    if kind != "generation":
        raise ValueError("MLX serves language model deployments only")
    if slimserve is not None or components or projector is not None:
        raise ValueError("an MLX deployment is its snapshot alone")
    if budget is not None:
        raise ValueError("mlx-lm takes no thinking budget")
    if sha256.lower() != mlx.digest():
        raise ValueError("an MLX deployment's sha256 is its snapshot's own checksum")


# A SlimServe deployment is a language model whose files are the weights, a
# projector where the model reads images, and a drafter where the profile
# speculates, each named in the layout exactly when it is given. Its thinking
# is the profile's, which SlimServe does not let a launch change.
def slimserve_options(kind: str, slimserve: SlimServeProfile | None, components: dict, projector: str | None, thinking: str | None, budget: int | None) -> None:
    if slimserve is None:
        if kind == "generation" and components:
            raise ValueError("components apply to image, video, speech and SlimServe deployments only")
        return
    if kind != "generation":
        raise ValueError("SlimServe serves language model deployments only")
    if set(components) - {"drafter"}:
        raise ValueError("a SlimServe deployment's one component is its drafter")
    files = set(slimserve.layout.values())
    if ("projector" in files) != (projector is not None) or ("drafter" in files) != ("drafter" in components):
        raise ValueError("a SlimServe layout names exactly the files the deployment gives")
    if thinking is not None or budget is not None:
        raise ValueError("a SlimServe deployment thinks as its profile does")


class AgentDeployment(BaseModel):
    """What agent.yaml records for a deployment; enough to start it again."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    kind: Literal[KINDS]
    model_path: str
    mmproj_path: str | None = None
    mmproj_sha256: str | None = None
    revision: str
    sha256: str
    port: int = Field(ge=1, le=65535)
    served_model_name: str
    # Zero for a model with no window, and for a SlimServe deployment served
    # with its profile's own.
    context_length: int = Field(default=0, ge=0, le=SLIMSERVE_WINDOW_LIMIT)
    pooling: Literal["mean", "last", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None
    # Whether a language model thinks before it answers, and for how many
    # tokens at most when it does; unset leaves the model's template to
    # decide. Thinking counts against a caller's token limit, so a model
    # that thinks on can answer nothing within a tight one.
    thinking: Literal["on", "off"] | None = None
    thinking_budget: int | None = Field(default=None, ge=1, le=65536)
    # An image deployment's companions and the sampling it was pinned with;
    # a speech deployment's one companion is its voice configuration.
    components: dict[str, Component] = {}
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg_scale: float | None = Field(default=None, ge=0, le=30)
    sampler: Literal[IMAGE_SAMPLERS] | None = None
    # A video deployment's pinned sampling beyond an image's, and the clip a
    # request gets unless it asks for another: its length and size.
    flow_shift: float | None = Field(default=None, ge=0, le=20)
    fps: int | None = Field(default=None, ge=1, le=60)
    seconds: int | None = Field(default=None, ge=1, le=10)
    size: str | None = Field(default=None, pattern=VIDEO_SIZE)
    # A video model with a spatial upscaler makes each clip at half its size
    # and doubles it, then refines it through these noise levels, or the
    # engine's own when none are given.
    upscale_sigmas: list[float] | None = None
    # The type a video model's language-model text encoder is converted to
    # as it loads, where it is published only in bf16; unset keeps it.
    text_encoder_type: Literal[TEXT_ENCODER_TYPES] | None = None
    # A transcription deployment's spoken language.
    language: str | None = Field(default=None, pattern=LANGUAGE)
    # A language model SlimServe serves instead of llama-server.
    slimserve: SlimServeProfile | None = None
    # A language model mlx-lm serves from its snapshot directory, model_path.
    mlx: MLXSnapshot | None = None

    @field_validator("model_path", "mmproj_path")
    @classmethod
    def canonical_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from pathlib import Path

        path = Path(value)
        if not path.is_absolute() or str(path) != value or value.startswith("//") or ".." in path.parts:
            raise ValueError("model paths must be canonical absolute paths")
        return value

    @model_validator(mode="after")
    def kind_options(self):
        if self.kind == "embedding":
            self.pooling = self.pooling or "mean"
            self.normalization = self.normalization or "l2"
        elif self.pooling is not None or self.normalization is not None:
            raise ValueError("pooling and normalization apply to embedding deployments only")
        # A multimodal embedding model's projector holds its image and audio
        # encoders, as a vision model's does for generation.
        if self.kind not in ("generation", "embedding") and self.mmproj_path is not None:
            raise ValueError(f"a {self.kind} deployment has no projector")
        thinking_options(self.kind, self.thinking, self.thinking_budget)
        video_options(self.kind, self.flow_shift, self.fps, self.seconds, self.size)
        upscale_options(self.kind, self.components, self.size, self.upscale_sigmas)
        text_encoder_options(self.kind, self.components, self.text_encoder_type)
        if self.kind in DIFFUSION:
            diffusion_components(self.kind, self.components)
            self.steps = self.steps or 20
            self.cfg_scale = (7.0 if self.kind == "image" else 5.0) if self.cfg_scale is None else self.cfg_scale
            self.sampler = self.sampler or "euler"
            if self.kind == "video":
                self.fps, self.seconds, self.size = self.fps or 16, self.seconds or 2, self.size or "640x352"
        else:
            if self.steps is not None or self.cfg_scale is not None or self.sampler is not None:
                raise ValueError("steps, cfg_scale and sampler apply to image and video deployments only")
            if self.kind == "speech":
                speech_components(self.components, self.model_path)
            elif self.kind != "generation" and self.components:
                raise ValueError("components apply to image, video, speech and SlimServe deployments only")
        slimserve_options(self.kind, self.slimserve, self.components, self.mmproj_path, self.thinking, self.thinking_budget)
        mlx_options(self.kind, self.mlx, self.slimserve, self.components, self.mmproj_path, self.thinking_budget, self.sha256)
        if self.kind == "transcription":
            self.language = self.language or "auto"
        elif self.language is not None:
            raise ValueError("language applies to transcription deployments only")
        if self.slimserve is None:
            llama_window(self.kind, self.context_length)
        elif self.context_length and self.context_length < 128:
            raise ValueError("a SlimServe deployment's window, when it names one, is at least 128 tokens")
        return self


def llama_window(kind: str, window: int) -> None:
    if kind not in NO_CONTEXT and not 128 <= window <= LLAMA_WINDOW_LIMIT:
        raise ValueError(f"a llama-server deployment's window is between 128 and {LLAMA_WINDOW_LIMIT} tokens")


# The window a request is served with: the one it names, or llama-server's
# default; a SlimServe profile's own when it names none.
def request_window(request: DeploymentRequest) -> int:
    if request.context_length is not None:
        return request.context_length
    return 0 if request.slimserve is not None else LLAMA_DEFAULT_WINDOW


def video_options(kind: str, flow_shift, fps, seconds, size) -> None:
    if kind != "video" and any(value is not None for value in (flow_shift, fps, seconds, size)):
        raise ValueError("flow_shift, fps, seconds and size apply to video deployments only")
    # Every video model the engine serves takes sides a multiple of 32; a
    # size it could not make would refuse every request that asks for none.
    if size is not None and any(int(side) % 32 for side in size.split("x")):
        raise ValueError("each side of a video's size is a multiple of 32")


def upscale_options(kind: str, components: dict, size: str | None, sigmas: list[float] | None) -> None:
    upscaled = kind == "video" and "spatial_upscaler" in components
    if sigmas is not None and not upscaled:
        raise ValueError("only a video model with a spatial upscaler refines after it")
    falling = sigmas is not None and all(later < earlier for earlier, later in zip(sigmas, sigmas[1:]))
    if sigmas is not None and (len(sigmas) < 2 or any(not 0 <= sigma <= 1 for sigma in sigmas) or sigmas[-1] != 0 or not falling):
        raise ValueError("a refine pass's noise levels are 0 to 1 and fall to 0")
    # The clip is made at half its size, which the engine takes in
    # multiples of 32; with no size given there is none it was reviewed at.
    if upscaled and size is None:
        raise ValueError("an upscaled video names its size")
    if upscaled and any(int(side) % 64 for side in size.split("x")):
        raise ValueError("each side of an upscaled video's size is a multiple of 64")


def text_encoder_options(kind: str, components: dict, text_encoder_type: str | None) -> None:
    if text_encoder_type is not None and (kind != "video" or "llm" not in components):
        raise ValueError("a text encoder type applies to a video model whose text encoder is a language model")


def diffusion_components(kind: str, components: dict) -> None:
    known = IMAGE_COMPONENTS if kind == "image" else VIDEO_COMPONENTS
    unknown = set(components) - set(known)
    if unknown:
        raise ValueError(f"unknown {kind} components: {', '.join(sorted(unknown))}")
    # Every video model reads its prompt through a text encoder and decodes
    # through an autoencoder published beside it.
    if kind == "video" and ("vae" not in components or not {"t5xxl", "llm"} & set(components)):
        raise ValueError("a video model names its autoencoder, vae, and its text encoder, t5xxl or llm")


def thinking_options(kind: str, thinking: str | None, budget: int | None) -> None:
    if kind != "generation" and (thinking is not None or budget is not None):
        raise ValueError("thinking applies to generation deployments only")
    if budget is not None and thinking != "on":
        raise ValueError("a thinking budget applies when thinking is on")


# piper finds a voice's configuration by the weights' own name with .json
# appended; the one component a speech deployment names must be that file.
def speech_components(components: dict, model_path: str) -> None:
    if set(components) != SPEECH_COMPONENTS:
        raise ValueError("a speech deployment names its voice configuration as its one component, config")
    if components["config"].path != model_path + ".json":
        raise ValueError("a voice's configuration is the weights' name with .json appended")


class ComponentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact: str
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")


class DeploymentRequest(BaseModel):
    """Constrained input; arbitrary llama.cpp flags never cross this boundary."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[KINDS]
    artifact: str
    mmproj: str | None = None
    revision: str = Field(pattern=r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    mmproj_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    served_model_name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    # Unset is 8192 for llama-server, and a SlimServe profile's own window.
    context_length: int | None = Field(default=None, ge=128, le=SLIMSERVE_WINDOW_LIMIT)
    pooling: Literal["mean", "last", "cls"] | None = None
    normalization: Literal["l2", "none"] | None = None
    thinking: Literal["on", "off"] | None = None
    thinking_budget: int | None = Field(default=None, ge=1, le=65536)
    components: dict[str, ComponentRequest] = {}
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg_scale: float | None = Field(default=None, ge=0, le=30)
    sampler: Literal[IMAGE_SAMPLERS] | None = None
    flow_shift: float | None = Field(default=None, ge=0, le=20)
    fps: int | None = Field(default=None, ge=1, le=60)
    seconds: int | None = Field(default=None, ge=1, le=10)
    size: str | None = Field(default=None, pattern=VIDEO_SIZE)
    upscale_sigmas: list[float] | None = None
    text_encoder_type: Literal[TEXT_ENCODER_TYPES] | None = None
    language: str | None = Field(default=None, pattern=LANGUAGE)
    slimserve: SlimServeProfile | None = None
    # The artifact is then the snapshot's directory.
    mlx: MLXSnapshot | None = None

    @model_validator(mode="after")
    def projector_checksum(self):
        if (self.mmproj is None) != (self.mmproj_sha256 is None):
            raise ValueError("a projector is named together with its sha256")
        if self.kind not in DIFFUSION and (self.steps is not None or self.cfg_scale is not None or self.sampler is not None):
            raise ValueError("steps, cfg_scale and sampler apply to image and video deployments only")
        thinking_options(self.kind, self.thinking, self.thinking_budget)
        video_options(self.kind, self.flow_shift, self.fps, self.seconds, self.size)
        upscale_options(self.kind, self.components, self.size, self.upscale_sigmas)
        text_encoder_options(self.kind, self.components, self.text_encoder_type)
        if self.kind in DIFFUSION:
            diffusion_components(self.kind, self.components)
        elif self.kind == "speech":
            if set(self.components) != SPEECH_COMPONENTS:
                raise ValueError("a speech deployment names its voice configuration as its one component, config")
        elif self.kind != "generation" and self.components:
            raise ValueError("components apply to image, video, speech and SlimServe deployments only")
        slimserve_options(self.kind, self.slimserve, self.components, self.mmproj, self.thinking, self.thinking_budget)
        mlx_options(self.kind, self.mlx, self.slimserve, self.components, self.mmproj, self.thinking_budget, self.sha256)
        if self.kind != "transcription" and self.language is not None:
            raise ValueError("language applies to transcription deployments only")
        return self


class Admission:
    """Per-deployment ingress gate: closed while the process is being replaced
    or removed, and a count of requests in flight so a replacement waits for
    them rather than cutting an answer short."""

    def __init__(self) -> None:
        self.paused = False
        self.requests = 0
        self.idle = asyncio.Event()
        self.idle.set()

    def enter(self) -> None:
        self.requests += 1
        self.idle.clear()

    def leave(self) -> None:
        self.requests -= 1
        if self.requests == 0:
            self.idle.set()


def deployment_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    return server_command(agent, deployment) + log_level_arguments(deployment.kind, log_level.current())


# The appliance's log level as a server takes it on its command line, at the
# server's start; a server started before a change keeps the level it had.
def log_level_arguments(kind: str, level: str | None) -> list[str]:
    if kind in DIFFUSION and level in log_level.DIFFUSION_LEVELS:
        return ["--log-level", log_level.DIFFUSION_LEVELS[level]]
    if kind == "transcription" and level == "debug":
        return ["--print-progress"]
    return []


def server_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    if deployment.kind == "image":
        return image_command(agent, deployment)
    if deployment.kind == "video":
        return video_command(agent, deployment)
    if deployment.kind == "transcription":
        return transcription_command(agent, deployment)
    if deployment.kind == "speech":
        return speech_command(agent, deployment)
    command = [agent.config.llama_server]
    if deployment.kind == "embedding":
        command += [
            "--embedding", "--pooling", deployment.pooling or "mean",
            "--embd-normalize", "2" if (deployment.normalization or "l2") == "l2" else "-1",
        ]
    else:
        command += ["--jinja"]
    command += [
        "--alias", deployment.served_model_name,
        "--host", "127.0.0.1",
        "--port", str(deployment.port),
        # b11459 applies env before argv, then remote selection after argv; the
        # selectors are cleared so the final -m is the model that loads.
        "--model-url", "", "--hf-repo", "", "--docker-repo", "",
        "-m", deployment.model_path,
    ]
    if deployment.mmproj_path:
        command += ["--mmproj", deployment.mmproj_path]
    command += ["-c", str(deployment.context_length)]
    if deployment.kind == "embedding":
        # An embedding model reads each input whole in one batch, so a batch
        # as large as the window takes any input the window does: a recording
        # or an image runs to hundreds of tokens, past the default 512.
        command += ["-b", str(deployment.context_length), "-ub", str(deployment.context_length)]
    return command


def engine_of(deployment: AgentDeployment) -> str:
    if deployment.mlx is not None:
        return MLX
    return SLIMSERVE if deployment.slimserve is not None else ENGINES[deployment.kind]


# mlx-lm's server behind the agent's guard (lazarus.agent.mlx_server), in the
# agent's own Python, which carries mlx-lm. Its thinking switch is passed
# to the model's template as llama-server's is; a reply runs to the window
# unless the caller asks for less. mlx-lm takes no window of its own, so
# the window is the default reply's length, not a bound on a request; what
# it holds beside the weights is bounded instead: four requests decoded at
# once, and the prompts it keeps for reuse held to MLX_PROMPT_CACHE_BYTES.
MLX_DECODE_CONCURRENCY = 4
MLX_PROMPT_CACHE_BYTES = 2 << 30


def mlx_command(deployment: AgentDeployment) -> list[str]:
    command = [
        sys.executable, "-m", "lazarus.agent.mlx_server",
        "--model", deployment.model_path,
        "--host", "127.0.0.1",
        "--port", str(deployment.port),
        "--max-tokens", str(deployment.context_length or LLAMA_DEFAULT_WINDOW),
        "--decode-concurrency", str(MLX_DECODE_CONCURRENCY),
        "--prompt-cache-bytes", str(MLX_PROMPT_CACHE_BYTES),
    ]
    if deployment.thinking is not None:
        command += ["--chat-template-args", json.dumps({"enable_thinking": deployment.thinking == "on"})]
    return command


# SlimServe finds a profile's files under its model directory by the paths
# its profile list gives them. Each deployment gets a directory of its own
# holding links to the verified files it was given, laid out afresh at every
# start; SlimServe keeps the KV cache it spills to disk there too.
def lay_out_slimserve(agent: Agent, deployment_id: str, deployment: AgentDeployment) -> Path:
    import shutil

    home = agent.config_path.parent if agent.config_path else Path.home() / ".sovereign"
    directory = home / "slimserve" / deployment_id
    shutil.rmtree(directory, ignore_errors=True)
    sources = {"model": deployment.model_path, "projector": deployment.mmproj_path}
    if "drafter" in deployment.components:
        sources["drafter"] = deployment.components["drafter"].path
    for name, file in deployment.slimserve.layout.items():
        link = directory / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(sources[file])
    return directory


# The profile is served as SlimServe tuned it, with its own window unless the
# deployment names a smaller one. Every file is in place, so SlimServe downloads nothing,
# and it is not told it may: one found missing or changed fails the launch
# rather than being fetched.
def slimserve_command(agent: Agent, deployment: AgentDeployment, directory: Path) -> list[str]:
    return [
        agent.config.slimserve, deployment.slimserve.profile,
        "--quant", deployment.slimserve.quant,
        "--cache", str(directory),
        "--host", "127.0.0.1",
        "--port", str(deployment.port),
        "--served-model-name", deployment.served_model_name,
        *(["--ctx", str(deployment.context_length)] if deployment.context_length else []),
    ]


# What a language model server takes through its environment: the server's
# reasoning switches are read from it (LLAMA_ARG_REASONING, the budget as
# LLAMA_ARG_THINK_BUDGET) rather than from its arguments, and its verbosity
# at the appliance's log level as it starts. SlimServe is told it is offline,
# and logs at the appliance's level.
def deployment_environment(deployment: AgentDeployment) -> dict[str, str]:
    environment: dict[str, str] = {}
    if deployment.slimserve is not None:
        environment["HF_HUB_OFFLINE"] = "1"
        level = log_level.current()
        if level in log_level.LEVELS:
            environment["VLLM_LOGGING_LEVEL"] = {"warn": "WARNING"}.get(level, level.upper())
        return environment
    verbosity = log_level.LLAMA_VERBOSITY.get(log_level.current() or "")
    if deployment.kind in ("generation", "embedding") and verbosity is not None:
        environment["LLAMA_ARG_LOG_VERBOSITY"] = verbosity
    if deployment.thinking is not None:
        environment["LLAMA_ARG_REASONING"] = deployment.thinking
    if deployment.thinking_budget is not None:
        environment["LLAMA_ARG_THINK_BUDGET"] = str(deployment.thinking_budget)
    return environment


# The server has no API key of its own; it listens on loopback and is
# reached through the agent's proxy, which gates admission. A model with
# components is standalone diffusion weights; one without is a full
# checkpoint carrying its own text encoder and autoencoder, which the server
# loads under a different flag.
def image_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    command = [
        agent.config.sd_server,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(deployment.port),
        "--diffusion-model" if deployment.components else "--model", deployment.model_path,
        "--lora-model-dir", LORA_DIRECTORY,
    ]
    for name, flag in IMAGE_COMPONENTS.items():
        component = deployment.components.get(name)
        if component is not None:
            command += [flag, component.path]
    command += [
        "--steps", str(deployment.steps),
        "--cfg-scale", str(deployment.cfg_scale),
        "--sampling-method", deployment.sampler,
    ]
    return command


# A video model loads its diffusion weights with every component beside them,
# its transformer with flash attention. Its autoencoder runs on the
# processors, a few frames at a time: Metal has no 3D im2col, so its 3D
# convolutions run one output at a time, and LTX-2.5's decoder took 855 s
# for a two-second 1280x704 clip there against 317 s on the processors of an
# M3 Max; a whole clip at that size does not fit at once. A text encoder
# published only in bf16 is converted as it loads. Sampling is per request,
# from the pinned values the videos API sends.
def video_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    command = [
        agent.config.sd_server,
        "--listen-ip", "127.0.0.1",
        "--listen-port", str(deployment.port),
        "--diffusion-model", deployment.model_path,
        "--lora-model-dir", LORA_DIRECTORY,
    ]
    for name, flag in VIDEO_COMPONENTS.items():
        component = deployment.components.get(name)
        if component is not None:
            command += [flag, os.path.dirname(component.path) if name == "spatial_upscaler" else component.path]
    if deployment.text_encoder_type is not None:
        command += ["--tensor-type-rules", rf"^text_encoders\.llm\.={deployment.text_encoder_type}"]
    return command + ["--diffusion-fa", "--backend", "vae=cpu", "--temporal-tiling"]


# whisper-server answers its inference route under the OpenAI transcription
# path, as the gateway and the workspace call it; timestamps are left out of
# the text it returns.
def transcription_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    return [
        agent.config.whisper_server,
        "--host", "127.0.0.1",
        "--port", str(deployment.port),
        "-m", deployment.model_path,
        "--inference-path", "/v1/audio/transcriptions",
        "--no-timestamps",
        "--language", deployment.language or "auto",
    ]


# piper's own HTTP server, run by the agent's interpreter unless the
# configuration names another; it loads the voice named by path and finds
# the configuration beside it.
def speech_command(agent: Agent, deployment: AgentDeployment) -> list[str]:
    server = shlex.split(agent.config.piper_server) or [sys.executable, "-m", "piper.http_server"]
    return [*server, "--host", "127.0.0.1", "--port", str(deployment.port), "-m", deployment.model_path]


def deployment_lock(agent: Agent, deployment_id: str) -> asyncio.Lock:
    return agent.deployment_locks.setdefault(deployment_id, asyncio.Lock())


def free_port(agent: Agent) -> int:
    """Called with records_lock held; a port handed out is reserved until the
    transition that took it commits or gives it back."""
    taken = {deployment.port for deployment in agent.config.deployments.values()}
    # A child kept registered without a record (a failed creation whose stop
    # failed) still owns its port until its termination is confirmed.
    taken |= {process.port for process in agent.deployments.values()}
    taken |= agent.port_reservations
    taken.add(agent.config.port)
    for port in DEPLOYMENT_PORTS:
        if port not in taken and port_available(port):
            return port
    raise RuntimeError("no free deployment port")


def port_available(port: int) -> bool:
    """A port nothing on this host listens on, whatever the records say."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def status_of(agent: Agent, deployment_id: str, deployment: AgentDeployment, healthy: bool) -> dict:
    process = agent.deployments.get(deployment_id)
    running = process is not None and process.running()
    try:
        model = agent.observed_model(deployment.model_path, directory=deployment.mlx is not None)
    except (OSError, ValueError, TypeError, RuntimeError):
        return {"status": "unhealthy", "error_code": "MODEL_LOAD_FAILED", "kind": deployment.kind}
    admission = agent.deployment_admission.get(deployment_id)
    paused = admission is not None and admission.paused
    if healthy and running:
        # A child that answers behind a closed gate is not serving: nothing
        # reaches it until the gate reopens.
        status = "paused" if paused else "healthy"
    else:
        status = "loading" if running else "unhealthy"
    return {
        "status": status,
        "admission": "paused" if paused else "open",
        "kind": deployment.kind,
        "model": model,
        "port": deployment.port,
        "served_model_name": deployment.served_model_name,
        "context_length": deployment.context_length,
        "thinking": deployment.thinking,
        "thinking_budget": deployment.thinking_budget,
        "revision": deployment.revision,
        "engine": engine_of(deployment),
        "memory_bytes": process.memory_bytes() if running else None,
    }


async def observe_deployments(agent: Agent) -> dict[str, dict]:
    snapshot = list(agent.config.deployments.items())

    async def probe(deployment_id: str) -> bool:
        process = agent.deployments.get(deployment_id)
        if process is None:
            return False
        try:
            return await asyncio.wait_for(process.healthy(), timeout=OBSERVE_TIMEOUT)
        except asyncio.TimeoutError:
            return False

    healthy = await asyncio.gather(*(probe(deployment_id) for deployment_id, _ in snapshot))
    result = {}
    for (deployment_id, deployment), is_healthy in zip(snapshot, healthy):
        # A deployment removed while it was being probed is not reported.
        if agent.config.deployments.get(deployment_id) is not deployment:
            continue
        result[deployment_id] = status_of(agent, deployment_id, deployment, is_healthy)
    return result


async def wait_deployment_ready(agent: Agent, deployment: AgentDeployment, process: ServerProcess) -> None:
    timeout = getattr(agent, "deployment_ready_timeout", SLIMSERVE_READY_TIMEOUT if deployment.slimserve else READY_TIMEOUT)
    if deployment.kind == "embedding":
        await agent.wait_embedding_ready(process, timeout=timeout)
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await process.healthy():
            return
        if not process.running():
            break
        await asyncio.sleep(1)
    raise RuntimeError("deployment did not become healthy before timeout")


# mlx-lm loads every weight file it finds in the directory, so the
# directory holds the pinned files and nothing else, none of them a link.
def snapshot_holds_its_pins(directory: Path, files: dict[str, str]) -> None:
    present = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{path} is a link; an MLX snapshot holds its files themselves")
        if path.is_file():
            present.add(path.relative_to(directory).as_posix())
    if present != set(files):
        extra, missing = sorted(present - set(files)), sorted(set(files) - present)
        raise ValueError(f"the MLX snapshot in {directory} holds {extra or 'nothing'} beyond its pins and lacks {missing or 'nothing'}")


def file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_deployment_files(deployment: AgentDeployment) -> None:
    """The bytes on disk are the bytes the record was made for. A weight
    overwritten since is refused, never loaded under the recorded revision."""
    if deployment.mlx is not None:
        snapshot_holds_its_pins(Path(deployment.model_path), deployment.mlx.files)
        for name, digest in deployment.mlx.files.items():
            if file_digest(str(Path(deployment.model_path) / name)) != digest:
                raise ValueError(f"{deployment.model_path}/{name} no longer matches its recorded checksum")
        return
    if file_digest(deployment.model_path) != deployment.sha256.lower():
        raise ValueError(f"{deployment.model_path} no longer matches its recorded checksum")
    if deployment.mmproj_path and deployment.mmproj_sha256 and file_digest(deployment.mmproj_path) != deployment.mmproj_sha256.lower():
        raise ValueError(f"{deployment.mmproj_path} no longer matches its recorded checksum")
    for component in deployment.components.values():
        if file_digest(component.path) != component.sha256.lower():
            raise ValueError(f"{component.path} no longer matches its recorded checksum")


def start_deployment(agent: Agent, deployment_id: str, verify: bool = False, record: AgentDeployment | None = None) -> ServerProcess:
    """Starts the recorded deployment, or a candidate record not yet committed
    to the configuration."""
    from lazarus.agent.server import ServerProcess

    if agent.stopping:
        raise RuntimeError("the agent is shutting down")
    deployment = record or agent.config.deployments[deployment_id]
    agent.observed_model(deployment.model_path, directory=deployment.mlx is not None)
    # A request's files were checked as it arrived; a restart from agent.yaml
    # checks them again, since the disk may have changed meanwhile.
    if verify:
        verify_deployment_files(deployment)
    if deployment.slimserve is not None:
        if not agent.config.slimserve:
            raise ValueError("SlimServe is not installed on this host")
        command = slimserve_command(agent, deployment, lay_out_slimserve(agent, deployment_id, deployment))
    elif deployment.mlx is not None:
        command = mlx_command(deployment)
    else:
        command = deployment_command(agent, deployment)
    return ServerProcess(
        deployment_id, command, deployment.port, deployment.model_path,
        revision=deployment.revision, context_length=deployment.context_length,
        authenticated=deployment.kind == "generation", environment=deployment_environment(deployment),
        health_path=HEALTH_PATHS[deployment.kind], engine=engine_of(deployment),
    )


async def quiesce(agent: Agent, deployment_id: str, transition: "Transition | None" = None) -> bool:
    """Close the gate and wait for what is in flight, or for the transition
    to be abandoned, whichever comes first; returns the gate's state before,
    so a transition that changes nothing can put it back."""
    admission = agent.deployment_admission.setdefault(deployment_id, Admission())
    was_paused = admission.paused
    admission.paused = True
    waits = [asyncio.ensure_future(admission.idle.wait())]
    if transition is not None:
        waits.append(asyncio.ensure_future(transition.gone.wait()))
    try:
        await asyncio.wait(waits, timeout=IDLE_TIMEOUT, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for wait in waits:
            wait.cancel()
    return was_paused


async def stop_deployment(agent: Agent, deployment_id: str) -> None:
    # Terminating a child can take up to ten seconds of blocking waits; that
    # runs off the event loop so every other deployment keeps streaming. Only
    # a transition worker calls this, and no request's cancel scope reaches a
    # worker, so the wait always runs to the child's end.
    process = agent.deployments.get(deployment_id)
    if process is None:
        return
    # The child stays registered until it is confirmed gone, so a failed stop
    # can be retried and an agent shutdown still finds it.
    await asyncio.to_thread(process.stop)
    if agent.deployments.get(deployment_id) is process:
        del agent.deployments[deployment_id]


class Transition:
    """One replacement or removal. It runs in a task of its own so the request
    that asked for it can be cancelled without leaving it half done: the
    worker notices the request is gone at its next safe point and either
    changes nothing or rolls back, to completion."""

    def __init__(self) -> None:
        self.abandoned = False
        # Set with abandoned, so a drain waiting on in-flight requests wakes
        # at once and puts the gate back instead of holding it for the
        # whole idle timeout.
        self.gone = asyncio.Event()
        # Set once the candidate is confirmed and the worker waits to commit
        # it; a request cancelled after this point is still rolled back.
        self.committing = False


async def run_transition(agent: Agent, worker_coroutine, transition: Transition, http_request: Request | None = None) -> dict:
    worker = asyncio.create_task(worker_coroutine)
    # A worker abandoned by its request still finishes and is joined at
    # shutdown; its outcome is read so a failure there is never an
    # unretrieved exception.
    agent.transitions.add(worker)
    worker.add_done_callback(agent.transitions.discard)
    worker.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    # A client that went away is not cancelled by the server; it is watched
    # for, and its transition abandoned like a cancelled one.
    watcher = asyncio.create_task(abandon_on_disconnect(http_request, worker, transition)) if http_request is not None else None
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        transition.abandoned = True
        transition.gone.set()
        raise
    finally:
        if watcher is not None:
            watcher.cancel()


DISCONNECT_POLL = 0.5


async def abandon_on_disconnect(http_request: Request, worker: asyncio.Task, transition: Transition) -> None:
    while not worker.done():
        if await http_request.is_disconnected():
            transition.abandoned = True
            transition.gone.set()
            return
        await asyncio.sleep(DISCONNECT_POLL)


async def apply_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest, http_request: Request | None = None) -> dict:
    # Checksumming multi-gigabyte weights must not stall every other
    # deployment's stream, so it runs off the event loop.
    # llama-server loads GGUF alone; stable-diffusion.cpp takes its encoders
    # and autoencoder as safetensors too. A file the loader would refuse is
    # refused here, before a serving process is touched.
    if request.slimserve is not None and not agent.config.slimserve:
        raise ValueError("SlimServe is not installed on this host")
    suffixes = weight_suffixes(request.kind)
    if request.mlx is not None:
        from lazarus.agent.server import mlx_engine

        if mlx_engine() is None:
            raise ValueError("mlx-lm is not installed on this host")
        model = await asyncio.to_thread(agent.resolve_snapshot, request.artifact, request.mlx.files)
    else:
        model = await asyncio.to_thread(agent.resolve_model, request.artifact, request.sha256, suffixes)
    mmproj = None
    if request.mmproj:
        mmproj = await asyncio.to_thread(agent.resolve_model, request.mmproj, request.mmproj_sha256, suffixes)
    components = {}
    for name, component in request.components.items():
        path = await asyncio.to_thread(agent.resolve_model, component.artifact, component.sha256, component_suffixes(request.kind))
        components[name] = Component(path=str(path), sha256=component.sha256.lower())
    transition = Transition()
    return await run_transition(agent, replace_deployment(agent, deployment_id, request, model, mmproj, components, transition), transition, http_request)


async def replace_deployment(agent: Agent, deployment_id: str, request: DeploymentRequest, model, mmproj, components: dict[str, Component], transition: Transition) -> dict:
    async with deployment_lock(agent, deployment_id):
        if transition.abandoned:
            # The request went away while this waited for the lock; nothing
            # has been touched, and nothing will be.
            return {"status": "unchanged", "id": deployment_id}
        previous = agent.config.deployments.get(deployment_id)
        async with agent.records_lock:
            if transition.abandoned:
                # The request went away while this waited for the shared
                # lock; no drain, no process, nothing.
                return {"status": "unchanged", "id": deployment_id}
            port = previous.port if previous else free_port(agent)
            agent.port_reservations.add(port)
        try:
            return await replace_on_port(agent, deployment_id, request, model, mmproj, components, transition, previous, port)
        finally:
            agent.port_reservations.discard(port)


async def replace_on_port(agent: Agent, deployment_id: str, request: DeploymentRequest, model, mmproj, components: dict[str, Component], transition: Transition, previous, port: int) -> dict:
    """The transition proper, with the deployment's own lock held and its
    port reserved by the caller."""
    candidate = AgentDeployment(
        kind=request.kind, model_path=str(model), mmproj_path=str(mmproj) if mmproj else None,
        mmproj_sha256=request.mmproj_sha256.lower() if request.mmproj_sha256 else None,
        revision=request.revision.lower(), sha256=request.sha256.lower(),
        port=port,
        served_model_name=request.served_model_name,
        context_length=0 if request.kind in NO_CONTEXT else request_window(request),
        pooling=request.pooling, normalization=request.normalization,
        thinking=request.thinking, thinking_budget=request.thinking_budget,
        components=components, steps=request.steps, cfg_scale=request.cfg_scale, sampler=request.sampler,
        flow_shift=request.flow_shift, fps=request.fps, seconds=request.seconds, size=request.size,
        upscale_sigmas=request.upscale_sigmas, text_encoder_type=request.text_encoder_type, language=request.language,
        slimserve=request.slimserve, mlx=request.mlx,
    )
    if previous is not None:
        was_paused = await quiesce(agent, deployment_id, transition)
        if transition.abandoned:
            # Nothing has changed: the process that was serving keeps
            # serving, behind the gate it had before the drain.
            agent.deployment_admission[deployment_id].paused = was_paused
            return {"status": "unchanged", "id": deployment_id}
    agent.deployment_admission[deployment_id] = Admission()
    agent.deployment_admission[deployment_id].paused = True
    try:
        # The previous process goes, and so does a child a failed creation
        # left registered without a record: a handle is never overwritten
        # while its child may still run.
        if previous is not None or deployment_id in agent.deployments:
            await stop_deployment(agent, deployment_id)
        # The drain may have taken minutes; the files are checked again
        # right before they are loaded, so what starts is what was pinned.
        await asyncio.to_thread(verify_deployment_files, candidate)
        # The candidate stays out of the configuration until it is confirmed:
        # another deployment's save meanwhile persists only what was verified.
        process = start_deployment(agent, deployment_id, record=candidate)
        agent.deployments[deployment_id] = process
        await wait_deployment_ready(agent, candidate, process)
        transition.committing = True
        async with agent.records_lock:
            # Checked again under the lock: the request may have gone while
            # this waited for it, and so may the candidate. Either is the
            # failure the rollback below handles, never a record.
            if transition.abandoned:
                raise RuntimeError("the request was cancelled before the deployment was confirmed")
            if not process.running():
                raise RuntimeError("the deployment exited before it could be recorded")
            committed = agent.config.deployments
            agent.config.deployments = {**committed, deployment_id: candidate}
            try:
                agent.save_config()
            except Exception:
                # A record that could not be saved is not a record: what was
                # committed before stays, in memory as on disk.
                agent.config.deployments = committed
                raise
    except Exception as exc:
        # A cancelled request is a failed one. The configuration never held
        # the candidate, so whatever cleanup does next, the rejected
        # candidate is never what is persisted. The previous process is
        # already gone, so it is restored from files checked again against
        # their checksums.
        rolled_back, rollback_error = False, None
        try:
            await stop_deployment(agent, deployment_id)
            if previous is None:
                agent.deployment_admission.pop(deployment_id, None)
            else:
                await asyncio.to_thread(verify_deployment_files, previous)
                restored = start_deployment(agent, deployment_id, record=previous)
                agent.deployments[deployment_id] = restored
                await wait_deployment_ready(agent, previous, restored)
                agent.deployment_admission[deployment_id].paused = False
            async with agent.records_lock:
                agent.save_config()
            rolled_back = True
        except Exception as rollback_exc:
            rollback_error = str(rollback_exc)
        return {
            "status": "unhealthy", "id": deployment_id, "error": str(exc),
            "rolled_back": rolled_back, "rollback_verified": rolled_back, "rollback_error": rollback_error,
        }
    agent.deployment_admission[deployment_id].paused = False
    return {"status": "healthy", "id": deployment_id, **status_of(agent, deployment_id, candidate, True)}


class PersistenceError(RuntimeError):
    """The process is gone but agent.yaml still records it; a retry finishes
    the removal rather than reporting a deployment absent that a restarted
    agent would serve again."""


async def remove_deployment(agent: Agent, deployment_id: str, http_request: Request | None = None) -> dict:
    transition = Transition()
    return await run_transition(agent, forget_deployment(agent, deployment_id, transition), transition, http_request)


async def forget_deployment(agent: Agent, deployment_id: str, transition: Transition) -> dict:
    async with deployment_lock(agent, deployment_id):
        if transition.abandoned:
            return {"status": "unchanged", "id": deployment_id}
        previous = agent.config.deployments.get(deployment_id)
        if previous is None:
            # No record, but a child a failed creation could not stop may
            # still be registered; it is stopped rather than called absent.
            if deployment_id in agent.deployments:
                await stop_deployment(agent, deployment_id)
                agent.deployment_admission.pop(deployment_id, None)
                return {"status": "stopped", "id": deployment_id}
            return {"status": "absent", "id": deployment_id}
        was_paused = await quiesce(agent, deployment_id, transition)
        if transition.abandoned:
            agent.deployment_admission[deployment_id].paused = was_paused
            return {"status": "unchanged", "id": deployment_id}
        await stop_deployment(agent, deployment_id)
        # The record goes, is saved, or comes back, under one acquisition of
        # the shared lock: no creation can take the port in between.
        async with agent.records_lock:
            agent.config.deployments = {k: v for k, v in agent.config.deployments.items() if k != deployment_id}
            try:
                agent.save_config()
            except Exception as exc:
                agent.config.deployments = {**agent.config.deployments, deployment_id: previous}
                raise PersistenceError(f"deployment stopped but not forgotten: {exc}") from exc
        agent.deployment_admission.pop(deployment_id, None)
        return {"status": "stopped", "id": deployment_id}


# The containers whisper-server cannot open: it decodes uploads in memory
# with miniaudio, which reads WAV, MP3, FLAC and Ogg Vorbis and nothing a
# browser records. The workspace converts its recordings to WAV before they
# arrive; an API client sending one of these is told what to send instead of
# a bare decode failure.
UNSUPPORTED_CONTAINERS = ((b"\x1a\x45\xdf\xa3", "WebM"), (b"ftyp", "MP4"))


def unsupported_container(body: bytes) -> str | None:
    marker = body.find(b'name="file"')
    if marker < 0:
        return None
    start = body.find(b"\r\n\r\n", marker)
    if start < 0:
        return None
    head = body[start + 4 : start + 16]
    for magic, name in UNSUPPORTED_CONTAINERS:
        if head.startswith(magic) or (name == "MP4" and head[4:8] == magic):
            return name
    return None


# What a voice reads aloud is the visible answer: a model's thinking, which
# Chat folds away, and the markdown that shapes text on a screen are not
# speech. Thinking opens the answer and goes, closed or cut off; a tag named
# later in prose is a word like any other. Fences go; inline code, headings,
# emphasis, links and list markers leave their words behind. Emphasis is
# recognised only where markdown would: an asterisk hugging its text on the
# inside, so "2 * 3 * 4" stays arithmetic.
THINKING = re.compile(r"^\s*<(think|thought|reasoning)>.*?</\1>\s*", re.DOTALL | re.IGNORECASE)
UNFINISHED_THINKING = re.compile(r"^\s*<(think|thought|reasoning)>.*$", re.DOTALL | re.IGNORECASE)
MARKDOWN = (
    (re.compile(r"```.*?```", re.DOTALL), " "),
    (re.compile(r"`([^`\n]*)`"), r"\1"),
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), " "),
    (re.compile(r"\[([^\]]+)\]\([^)]*\)"), r"\1"),
    (re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.MULTILINE), ""),
    (re.compile(r"^[ \t]*>[ \t]?", re.MULTILINE), ""),
    (re.compile(r"(\*\*|__|~~)(?=\S)(.+?)(?<=\S)\1", re.DOTALL), r"\2"),
    (re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])"), r"\1"),
    (re.compile(r"(?<![\w_])_(?=\S)([^_\n]+?)(?<=\S)_(?![\w_])"), r"\1"),
    (re.compile(r"^[ \t]*[-*_]{3,}[ \t]*$", re.MULTILINE), ""),
    (re.compile(r"[ \t]+"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
)


def spoken_text(text: str) -> str:
    text = UNFINISHED_THINKING.sub("", THINKING.sub("", text))
    for pattern, replacement in MARKDOWN:
        text = pattern.sub(replacement, text)
    return text.strip()


# The OpenAI speech request as piper takes it: the visible answer as text,
# and the speed as a length scale. The voice named is the deployment's; the
# answer is WAV, whatever format was asked for, since that is what the
# voice produces.
def speech_request(body: bytes) -> dict:
    try:
        request = json.loads(body or b"{}")
    except ValueError:
        raise ValueError("the request is not JSON")
    text = request.get("input") if isinstance(request, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError("input is required")
    if len(text) > 4096:
        raise ValueError("input is at most 4096 characters")
    text = spoken_text(text)
    if not text:
        raise ValueError("input holds nothing to say once thinking and markup are set aside")
    speed = request.get("speed", 1.0)
    if isinstance(speed, bool) or not isinstance(speed, (int, float)) or not 0.25 <= speed <= 4.0:
        raise ValueError("speed is between 0.25 and 4.0")
    return {"text": text, "length_scale": round(1.0 / float(speed), 4)}


async def synthesize(process: "ServerProcess", admission: Admission, body: bytes):
    try:
        payload = speech_request(body)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    admission.enter()
    try:
        async with httpx.AsyncClient(timeout=600.0, trust_env=False) as client:
            answer = await client.post(f"http://127.0.0.1:{process.port}/synthesize", json=payload)
    except httpx.HTTPError:
        return JSONResponse(status_code=503, content={"error": "deployment engine unavailable"})
    finally:
        admission.leave()
    if answer.status_code != 200:
        return JSONResponse(status_code=502, content={"error": f"the voice answered {answer.status_code}"})
    return Response(content=answer.content, media_type="audio/wav")


# The videos API: a create starts an engine job and answers with the video
# at once; the video is asked after, fetched when done, or cancelled by its
# id. See lazarus.agent.videos.
async def video_request(secret: str, process: "ServerProcess", admission: Admission, deployment: AgentDeployment, path: str, method: str, body: bytes):
    from lazarus.agent import videos

    engine = f"http://127.0.0.1:{process.port}/sdcpp/v1"
    admission.enter()
    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            if path == "videos":
                if method != "POST":
                    return JSONResponse(status_code=405, content={"error": "method not allowed"})
                job, video = videos.job_request(body, deployment)
                answer = await client.post(f"{engine}/vid_gen", json=job)
                if answer.status_code != 202:
                    return JSONResponse(status_code=502 if answer.status_code >= 500 else answer.status_code, content={"error": engine_error(answer)})
                started = answer.json()
                video["created_at"] = started.get("created", video["created_at"])
                return JSONResponse(content={"id": videos.video_id(secret, deployment.served_model_name, started["id"]), **video})
            match = VIDEO_PATH.match(path)
            if method == "POST":
                return JSONResponse(status_code=405, content={"error": "method not allowed"})
            job_id = videos.job_of(secret, deployment.served_model_name, match.group(1))
            if method == "DELETE":
                return cancelled(match.group(1), await client.post(f"{engine}/jobs/{job_id}/cancel"))
            answer = await client.get(f"{engine}/jobs/{job_id}")
            if answer.status_code in (404, 410):
                return JSONResponse(status_code=404, content={"error": "no such video"})
            if answer.status_code != 200:
                return JSONResponse(status_code=502, content={"error": engine_error(answer)})
            if match.group(2):
                data, media_type = videos.content_of(answer.json())
                return Response(content=data, media_type=media_type)
            return JSONResponse(content=videos.video_of(secret, deployment.served_model_name, answer.json()))
    except videos.VideoError as exc:
        return JSONResponse(status_code=exc.status, content={"error": str(exc)})
    except (httpx.HTTPError, ValueError, KeyError):
        return JSONResponse(status_code=503, content={"error": "deployment engine unavailable"})
    finally:
        admission.leave()


# The engine cancels a job only before it starts: one being made runs to the
# end, and one made already is kept until it expires.
def cancelled(identifier: str, answer) -> JSONResponse:
    if answer.status_code == 200:
        return JSONResponse(content={"id": identifier, "object": "video.deleted", "deleted": True})
    if answer.status_code in (404, 410):
        return JSONResponse(status_code=404, content={"error": "no such video"})
    if answer.status_code == 409:
        return JSONResponse(status_code=409, content={"error": "the video is being made and cannot be cancelled"})
    return JSONResponse(status_code=502, content={"error": engine_error(answer)})


# sd-server says why it refused as a string, or as an object with a message.
def engine_error(answer) -> str:
    try:
        error = answer.json()["error"]
    except (ValueError, KeyError, TypeError):
        return f"the engine answered {answer.status_code}"
    if isinstance(error, dict):
        error = error.get("message")
    return error if isinstance(error, str) and error else f"the engine answered {answer.status_code}"


def register_deployment_routes(app: FastAPI, agent: Agent) -> None:
    @app.get("/agent/deployments")
    async def list_deployments():
        return {"deployments": await observe_deployments(agent)}

    @app.put("/agent/admin/deployments/{deployment_id}")
    async def put_deployment(deployment_id: str, request: DeploymentRequest, http_request: Request):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        try:
            result = await apply_deployment(agent, deployment_id, request, http_request)
        except (OSError, ValueError, RuntimeError) as exc:
            return JSONResponse(status_code=422, content={"error": str(exc)})
        return JSONResponse(status_code=200 if result["status"] == "healthy" else 422, content=result)

    @app.delete("/agent/admin/deployments/{deployment_id}")
    async def delete_deployment(deployment_id: str, http_request: Request):
        if not DEPLOYMENT_ID.match(deployment_id):
            return JSONResponse(status_code=422, content={"error": "deployment id must be a short lowercase slug"})
        try:
            return await remove_deployment(agent, deployment_id, http_request)
        except (PersistenceError, OSError, RuntimeError) as exc:
            # The child may still be there; the record says so, and a retry
            # stops it again.
            return JSONResponse(status_code=500, content={"error": str(exc), "id": deployment_id})

    @app.api_route("/deployments/{deployment_id}/v1/{path:path}", methods=["GET", "POST", "DELETE"])
    async def proxy_deployment(deployment_id: str, path: str, request: Request):
        deployment = agent.config.deployments.get(deployment_id)
        if deployment is None:
            return JSONResponse(status_code=404, content={"error": f"unknown deployment {deployment_id!r}"})
        video = deployment.kind == "video" and VIDEO_PATH.match(path)
        if path not in ALLOWED_PATHS[deployment.kind] and not video:
            return JSONResponse(status_code=404, content={"error": "unsupported deployment endpoint"})
        # mlx-lm lists the host's own model cache, under its paths; the
        # deployment is its one model, by the name it is served under.
        if deployment.mlx is not None and path == "models" and request.method == "GET":
            return JSONResponse(content={"object": "list", "data": [{"id": deployment.served_model_name, "object": "model", "owned_by": "sovereign"}]})
        # Only a video, not its file or anything else, is deleted.
        if request.method == "DELETE" and not (video and not video.group(2)):
            return JSONResponse(status_code=405, content={"error": "method not allowed"})
        admission = agent.deployment_admission.setdefault(deployment_id, Admission())
        process = agent.deployments.get(deployment_id)
        # A configured deployment with no process is one being replaced or
        # removed: a retryable pause, never an unknown name.
        if admission.paused or process is None:
            return JSONResponse(status_code=503, content={"error": "deployment admission is paused"})
        body = await request.body()
        if admission.paused or agent.deployments.get(deployment_id) is not process:
            return JSONResponse(status_code=503, content={"error": "deployment is being replaced"})
        # A child that exited still owns its record until it is stopped; its
        # port may meanwhile belong to something else, so nothing is forwarded.
        if not process.running():
            return JSONResponse(status_code=503, content={"error": "deployment process is not running"})
        if deployment.kind == "speech":
            return await synthesize(process, admission, body)
        if deployment.kind == "video":
            return await video_request(agent.token + process.instance, process, admission, deployment, path, request.method, body)
        if deployment.kind == "transcription" and (container := unsupported_container(body)):
            return JSONResponse(status_code=415, content={"error": f"{container} audio is not decoded here; send WAV, MP3, FLAC or Ogg Vorbis"})
        client = httpx.AsyncClient(timeout=600.0, trust_env=False)
        # The request is built before admission is taken, so a request that
        # cannot be built never leaves a count behind.
        try:
            upstream = client.build_request(
                request.method, f"http://127.0.0.1:{process.port}/v1/{path}", content=body,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json"), **process.headers()},
            )
        except (UnicodeError, ValueError) as exc:
            await client.aclose()
            return JSONResponse(status_code=400, content={"error": f"request could not be forwarded: {exc}"})
        admission.enter()
        try:
            response = await client.send(upstream, stream=True)
        except httpx.HTTPError:
            admission.leave()
            await client.aclose()
            return JSONResponse(status_code=503, content={"error": "deployment engine unavailable"})
        except BaseException:
            admission.leave()
            await client.aclose()
            raise

        async def relay():
            async for chunk in response.aiter_raw():
                yield chunk

        async def cleanup():
            # Runs shielded from the client's disconnect, and releases
            # admission even when closing the upstream fails: a leaked count
            # would make the next replacement wait the whole drain timeout.
            try:
                await response.aclose()
                await client.aclose()
            finally:
                admission.leave()

        from lazarus.agent.server import RelayedResponse

        return RelayedResponse(
            relay(), cleanup=cleanup, status_code=response.status_code,
            media_type=response.headers.get("content-type"),
        )
