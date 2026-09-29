"""The OpenAI videos API over stable-diffusion.cpp's video jobs.

A video takes minutes to make, so it is a job: a request starts it and
answers at once with the video's id, the caller asks after it by that id,
and fetches the finished file. sd-server runs such jobs itself
(`/sdcpp/v1/vid_gen`, `/sdcpp/v1/jobs/{id}`); this module translates the
OpenAI shape onto them, with the deployment's pinned sampling.

A video's id carries the served model name beside the engine's job id, so a
gateway can route a later request about the video to the deployment that
made it without keeping any state of its own. The engine numbers its jobs in
order, so the id is also signed with the deployment's secret: only whoever
was given the id can ask after the video, fetch it or cancel it.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import time

MAX_PROMPT = 4000
MAX_SECONDS = 10
# Every side a whole number of 32 pixels, which every video model the
# engine serves accepts.
SIZE = re.compile(r"^([1-9][0-9]{2,3})x([1-9][0-9]{2,3})$")
JOB = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
STATUSES = {"queued": "queued", "generating": "in_progress", "completed": "completed"}


class VideoError(ValueError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def signature(secret: str, served_model_name: str, job: str) -> str:
    return hmac.new(secret.encode(), f"{served_model_name}/{job}".encode(), hashlib.sha256).hexdigest()[:32]


def video_id(secret: str, served_model_name: str, job: str) -> str:
    raw = f"{served_model_name}/{job}/{signature(secret, served_model_name, job)}".encode()
    return "video_" + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def job_of(secret: str, served_model_name: str, identifier: str) -> str:
    """The engine's job id inside a video id this deployment made and
    signed; the served name may itself hold a slash, the job and the
    signature never do."""
    encoded = identifier.removeprefix("video_")
    if encoded == identifier:
        raise VideoError(404, "no such video")
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise VideoError(404, "no such video")
    parts = raw.rsplit("/", 2)
    if len(parts) != 3:
        raise VideoError(404, "no such video")
    name, job, signed = parts
    if name != served_model_name or not JOB.match(job) or not hmac.compare_digest(signed, signature(secret, name, job)):
        raise VideoError(404, "no such video")
    return job


def seconds_of(value, reviewed: int) -> int:
    """OpenAI sends the length as a string of seconds; a number is taken too.
    A clip is no longer than the model was reviewed making, as it is no
    larger."""
    if value is None:
        return reviewed
    refused = VideoError(400, f"seconds is a whole number from 1 to {reviewed}")
    whole = isinstance(value, int) and not isinstance(value, bool)
    text = value.strip() if isinstance(value, str) else ""
    if not whole and not (text.isascii() and text.isdigit() and len(text) <= 3):
        raise refused
    seconds = int(value)
    if not 1 <= seconds <= min(reviewed, MAX_SECONDS):
        raise refused
    return seconds


def size_of(value, default: str, multiple: int = 32) -> tuple[int, int]:
    """A size no larger in area than the one the model was reviewed at, each
    side a multiple of 32, or of 64 for a clip made at half its size."""
    text = default if value is None else value
    match = SIZE.match(text) if isinstance(text, str) else None
    if match is None:
        raise VideoError(400, "size is WIDTHxHEIGHT")
    width, height = int(match.group(1)), int(match.group(2))
    if width % multiple or height % multiple:
        raise VideoError(400, f"each side of size is a multiple of {multiple}")
    reviewed = SIZE.match(default)
    if width * height > int(reviewed.group(1)) * int(reviewed.group(2)):
        raise VideoError(400, f"size is at most {default} in area")
    return width, height


def frames_of(seconds: int, fps: int) -> int:
    """The engine makes 4n + 1 frames; the nearest count not over the length."""
    frames = seconds * fps
    return max(frames - (frames - 1) % 4, 5)


def job_request(body: bytes, deployment) -> tuple[dict, dict]:
    """The engine's job for an OpenAI create request, and the video as the
    caller will see it."""
    try:
        request = json.loads(body or b"{}")
    except ValueError:
        raise VideoError(400, "the request is not JSON")
    if not isinstance(request, dict):
        raise VideoError(400, "the request is a JSON object")
    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise VideoError(400, "prompt is required")
    if len(prompt) > MAX_PROMPT:
        raise VideoError(400, f"prompt is at most {MAX_PROMPT} characters")
    seconds = seconds_of(request.get("seconds"), deployment.seconds)
    upscaler = deployment.components.get("spatial_upscaler")
    width, height = size_of(request.get("size"), deployment.size, 64 if upscaler else 32)
    job = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "video_frames": frames_of(seconds, deployment.fps),
        "fps": deployment.fps,
        "seed": -1,
        "sample_params": {
            "sample_method": deployment.sampler,
            "sample_steps": deployment.steps,
            "guidance": {"txt_cfg": deployment.cfg_scale},
        },
        "output_format": "webm",
    }
    if deployment.flow_shift is not None:
        job["sample_params"]["flow_shift"] = deployment.flow_shift
    # A model with a spatial upscaler makes the clip at half its size, then
    # doubles it and refines it: the upscaler is named as the engine knows
    # it, by its file's name without the extension.
    if upscaler is not None:
        hires = {"enabled": True, "upscaler": os.path.splitext(os.path.basename(upscaler.path))[0]}
        if deployment.upscale_sigmas:
            hires["custom_sigmas"] = deployment.upscale_sigmas
        job.update(width=width // 2, height=height // 2, hires=hires)
    video = {
        "object": "video",
        "model": deployment.served_model_name,
        "status": "queued",
        "progress": 0,
        # The engine's own time replaces this once it has taken the job.
        "created_at": int(time.time()),
        "size": f"{width}x{height}",
        "seconds": str(seconds),
    }
    return job, video


def video_of(secret: str, served_model_name: str, job: dict) -> dict:
    """The OpenAI video for an engine job; a cancelled job is a failed one."""
    status = STATUSES.get(job.get("status"), "failed")
    video = {
        "id": video_id(secret, served_model_name, job.get("id", "")),
        "object": "video",
        "model": served_model_name,
        "status": status,
        "progress": 100 if status == "completed" else 0,
        "created_at": job.get("created"),
    }
    if job.get("completed") and status == "completed":
        video["completed_at"] = job["completed"]
    if status == "failed":
        error = job.get("error") or {}
        video["error"] = {"code": error.get("code", "generation_failed"), "message": error.get("message", "the video was not made")}
    return video


def content_of(job: dict) -> tuple[bytes, str]:
    """The finished file and its media type."""
    if job.get("status") != "completed":
        raise VideoError(409, "the video is not ready")
    result = job.get("result") or {}
    try:
        data = base64.b64decode(result.get("b64_json") or "", validate=True)
    except (binascii.Error, ValueError):
        raise VideoError(502, "the engine returned an unreadable video")
    if not data:
        raise VideoError(502, "the engine returned no video")
    return data, result.get("mime_type") or "video/webm"
