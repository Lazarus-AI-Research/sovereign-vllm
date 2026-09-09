"""Bind the isolated SlimServe installation to its packaged source provenance.

The manifest is a first-party installation artifact, never a request input.
Importing an extension is not evidence that any of its kernels have run.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import inspect
import json
import re
import sys
import sysconfig
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import ModuleType

SOURCE_COMMIT = "44a7d1a21851c7164c098c93fbc1baa12ab99847"
PROVENANCE_PATH = Path("/opt/sovereign-slimserve/provenance.json")
KERNEL_MODULE = "vllm._quixicore_C"
FORBIDDEN_OPTIONAL_MODULES = ("flashinfer", "flashinfer_cubin", "flashinfer_jit_cache")
KERNEL_SOURCES = {
    "cuda": ("QuixiCore-CUDA", "08780aaa22cdc2d144b6beacb24df953134b34be"),
    "metal": ("QuixiCore-Metal", "71a08cd4cbcdc622ce31b3fc91e1f505e144b516"),
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
# Explicit computational entrypoints audited at the source pin. Do not infer
# evidence from all exports: _set_library, *_init, *_layer, *_debug, census,
# availability, residency and query functions do not prove native inference.
KERNEL_ENTRYPOINTS = frozenset({
    "dsv4_router_gemm", "dsv4_hash_router", "dsv4_projection_gemv",
    "dsv4_mhc_pre", "dsv4_mhc_fused_post_pre", "dsv4_mhc_post", "dsv4_hc_head",
    "mla_decode_fp8_sparse", "paged_attention", "paged_attention_partitioned",
    "paged_attention_verify", "muse_q38_run", "muse_step_run", "muse_step_run_aux",
    "dflash_step_run", "dflash_sample_greedy", "add_rms_norm", "gemma_rms_norm",
    "qgemv", "qgemm", "qflux_gelu", "qgemv_w8a8", "qgemv_w2a8",
    "quantize_per_token", "quantize_per_tensor", "lm_head_sample",
    "lm_head_sample_topk", "lm_head_sample_topp", "qgemm_actorder", "qgemm_blockscale",
})


class ProvenanceError(RuntimeError):
    """The installed implementation cannot be attributed to the reviewed pin."""


def reject_optional_flashinfer() -> None:
    """Check real importability without importing/probing the optional packages."""
    for name in FORBIDDEN_OPTIONAL_MODULES:
        if sys.modules.get(name) is not None or importlib.util.find_spec(name) is not None:
            raise ProvenanceError("unreviewed optional native package is importable")


def _object_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError("duplicate provenance field")
        result[key] = value
    return result


def bounded_json(value: str, limit: int) -> dict:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > limit:
        raise ProvenanceError("missing or oversized identity document")
    result = json.loads(value, object_pairs_hook=_object_pairs)
    if not isinstance(result, dict):
        raise ProvenanceError("identity document must be an object")
    return result


class InstalledProvenance:
    def __init__(self, path: Path, root: Path):
        reject_optional_flashinfer()
        self.root = root.resolve(strict=True)
        with path.open("rb") as source:
            raw = source.read(8 * 1024 * 1024 + 1)
        document = bounded_json(raw.decode("utf-8"), 8 * 1024 * 1024)
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ProvenanceError("unsupported installation provenance")
        slimserve = document.get("slimserve", {})
        if (
            slimserve.get("source_repository") != "https://github.com/QuixiAI/SlimServe"
            or slimserve.get("source_commit") != SOURCE_COMMIT
            or not isinstance(slimserve.get("package_version"), str)
            or not slimserve["package_version"]
        ):
            raise ProvenanceError("unreviewed SlimServe source")
        self.files = self._entries(slimserve.get("files"))
        self.kernels = document.get("kernels", {})
        self.artifacts = self._entries(self.kernels.get("artifacts"))
        self._verified = {}
        self.engine = {
            "name": "slimserve",
            "version": SOURCE_COMMIT,
            "adapter": "slimserve-runtime",
        }
        self.verify_file(self.root / "slimserve/profiles.json", self.files)

    @staticmethod
    def _entries(entries) -> dict[str, str]:
        if not isinstance(entries, list) or not entries:
            raise ProvenanceError("missing installed file provenance")
        result = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProvenanceError("invalid installed file provenance")
            name, digest = entry.get("path"), entry.get("sha256")
            if not isinstance(name, str) or not isinstance(digest, str):
                raise ProvenanceError("invalid installed file digest")
            relative = PurePosixPath(name)
            if (
                not name
                or len(name) > 1024
                or "\\" in name
                or relative.is_absolute()
                or any(part in ("", ".", "..") for part in name.split("/"))
                or not _DIGEST.fullmatch(digest)
                or name in result
            ):
                raise ProvenanceError("invalid installed file path or digest")
            result[name] = digest
        return result

    def verify_file(self, path: Path, entries: dict[str, str] | None = None) -> str:
        entries = self.files if entries is None else entries
        absolute = path.absolute()
        resolved = absolute.resolve(strict=True)
        if resolved != absolute or not resolved.is_relative_to(self.root):
            raise ProvenanceError("loaded implementation is outside its installation")
        relative = resolved.relative_to(self.root).as_posix()
        expected = entries.get(relative)
        if expected is None:
            raise ProvenanceError("loaded implementation has no source digest")
        stat = resolved.stat()
        identity = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        cached = self._verified.get(relative)
        if cached != (identity, expected):
            with resolved.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            if digest != expected:
                raise ProvenanceError("installed implementation digest mismatch")
            self._verified[relative] = (identity, expected)
        return expected

    def verify_module(self, module: ModuleType) -> None:
        name = getattr(module, "__name__", "")
        if not (name == "vllm" or name == "slimserve" or name.startswith(("vllm.", "slimserve."))):
            # python -m executes api_server under __main__.
            if name != "__main__":
                raise ProvenanceError("API is not from the isolated SlimServe package")
        filename = getattr(module, "__file__", None)
        if not isinstance(filename, str):
            raise ProvenanceError("loaded API has no source file")
        path = Path(filename)
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        if origin is None or Path(origin).resolve(strict=True) != path.resolve(strict=True):
            raise ProvenanceError("loaded API origin mismatch")
        if path.suffix == ".pyc":
            path = Path(importlib.util.source_from_cache(str(path)))
        if path.suffix != ".py":
            raise ProvenanceError("loaded API is not a provenance-bound Python source")
        self.verify_file(path)

    def verify_api(self, api) -> None:
        api = inspect.unwrap(api)
        module = inspect.getmodule(api)
        if module is None:
            raise ProvenanceError("loaded API has no module")
        self.verify_module(module)
        filename = inspect.getsourcefile(api)
        if filename is None:
            raise ProvenanceError("loaded API has no executable source")
        self.verify_file(Path(filename))
        module_source = inspect.getsourcefile(module)
        if module_source is None or Path(filename).resolve() != Path(module_source).resolve():
            raise ProvenanceError("loaded callable and module sources disagree")

    def verify_loaded_modules(self) -> None:
        reject_optional_flashinfer()
        for name, module in tuple(sys.modules.items()):
            if module is None or not name.startswith(("vllm.", "slimserve.")):
                continue
            filename = getattr(module, "__file__", None)
            if filename and Path(filename).suffix in (".py", ".pyc"):
                self.verify_module(module)

    def verify_extension(self, module: ModuleType, backend: str) -> dict:
        reject_optional_flashinfer()
        source = KERNEL_SOURCES.get(backend)
        if source is None:
            raise ProvenanceError("unsupported native kernel backend")
        repository, commit = source
        expected_version = f"{commit}+slimserve.{SOURCE_COMMIT}"
        if (
            self.kernels.get("library") != f"quixicore-{backend}"
            or self.kernels.get("source_repository") != f"https://github.com/QuixiAI/{repository}"
            or self.kernels.get("source_commit") != commit
            or self.kernels.get("modifications_commit") != SOURCE_COMMIT
            or self.kernels.get("version") != expected_version
            or self.kernels.get("module_name") != KERNEL_MODULE
            or not self.kernels.get("source_files")
            or getattr(module, "__name__", None) != KERNEL_MODULE
            or sys.modules.get(KERNEL_MODULE) is not module
        ):
            raise ProvenanceError("unreviewed native kernel provenance")
        filename = getattr(module, "__file__", "")
        if not isinstance(filename, str) or not any(
            filename.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES
        ):
            raise ProvenanceError("kernel module is not a compiled extension")
        spec = getattr(module, "__spec__", None)
        if (
            not isinstance(getattr(spec, "loader", None), importlib.machinery.ExtensionFileLoader)
            or getattr(spec, "origin", None) != filename
        ):
            raise ProvenanceError("kernel module has no native extension loader")
        self.verify_file(Path(filename), self.artifacts)
        for relative in self.artifacts:
            self.verify_file(self.root / relative, self.artifacts)
        if backend == "metal":
            library = Path(filename).with_name("quixicore_metal.metallib")
            self.verify_file(library, self.artifacts)
        return {"library": f"quixicore-{backend}", "version": expected_version, "backend": backend}


@lru_cache(maxsize=1)
def installed_provenance() -> InstalledProvenance:
    return InstalledProvenance(PROVENANCE_PATH, Path(sysconfig.get_path("purelib")))
