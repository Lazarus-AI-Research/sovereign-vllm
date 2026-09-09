#!/usr/bin/env python3
"""Build pinned SlimServe in its own interpreter; never invoke its installer/CLI."""

import argparse
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tomllib
import venv


PREFIX = Path("/opt/sovereign-slimserve")
SOURCE_LOCK = Path(__file__).with_name("provenance.json")


def run(*args, cwd=None, env=None):
    subprocess.run([str(arg) for arg in args], cwd=cwd, env=env, check=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path, root):
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError(f"Artifact escapes installed package root: {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path)}


def fetch_source(destination, source, environment):
    destination.mkdir(parents=True, exist_ok=True)
    run("git", "init", "--quiet", destination, env=environment)
    run("git", "-C", destination, "fetch", "--quiet", "--depth=1",
        "--no-tags", "--no-recurse-submodules", source["source_repository"],
        source["source_commit"], env=environment)
    run("git", "-C", destination, "-c", "submodule.recurse=false",
        "checkout", "--quiet", "--detach", "FETCH_HEAD", env=environment)
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != source["source_commit"]:
        raise RuntimeError(f"Wrong source revision for {destination}: {actual}")


def reject_flashinfer():
    """The reviewed native profiles have no pinned FlashInfer artifact closure."""
    for module in ("flashinfer", "flashinfer_cubin", "flashinfer_jit_cache"):
        if importlib.util.find_spec(module) is not None:
            raise RuntimeError(f"Unreviewed optional FlashInfer module is importable: {module}")
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name", "").lower().replace("_", "-").replace(".", "-")
        if name == "flashinfer" or name.startswith("flashinfer-"):
            raise RuntimeError(f"Unreviewed optional FlashInfer distribution is installed: {name}")


def require_build_dependencies(slimserve, runtime_source, target):
    from packaging.requirements import Requirement

    requirements = ["cmake>=3.26.1,<3.30"]
    for source in (slimserve, runtime_source):
        with (source / "pyproject.toml").open("rb") as stream:
            requirements.extend(tomllib.load(stream)["build-system"]["requires"])
    if target == "cuda":
        requirements.extend(
            line.split("#", 1)[0].strip()
            for line in (slimserve / "requirements/build/cuda.txt").read_text().splitlines()
            if line.split("#", 1)[0].strip()
        )
    for text in requirements:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        version = importlib.metadata.version(requirement.name)
        if version not in requirement.specifier:
            raise RuntimeError(f"Dependency lock has {requirement.name} {version}; requires {text}")


def preserve_licenses(source_root, destination):
    copied = []
    notice_names = ("license", "notice", "copying", "copyright")
    for directory, dirs, files in os.walk(source_root):
        dirs[:] = sorted(name for name in dirs if name != ".git")
        for name in sorted(files):
            relative = (Path(directory) / name).relative_to(source_root)
            if not any(part.lower().startswith(notice_names) for part in relative.parts):
                continue
            original = Path(directory) / name
            if not original.is_file():
                continue
            target = destination / original.relative_to(source_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            copied.append(file_record(target, PREFIX))
    if not copied:
        raise RuntimeError(f"No upstream license files found in {source_root}")
    return copied


def prepare_kernels(slimserve, upstream, kernel):
    """Keep the exact fork snapshot, attributing rather than erasing its changes."""
    records = []
    for local_dir, upstream_dir in kernel["source_mappings"].items():
        source_dir = slimserve / "csrc/quixicore" / local_dir
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or path.suffix not in {".cu", ".cuh", ".h", ".hpp", ".mm", ".metal"}:
                continue
            record = file_record(path, slimserve)
            base = upstream / upstream_dir / path.relative_to(source_dir) if upstream_dir else None
            record["origin"] = "slimserve-added"
            if base is not None and base.is_file():
                record["upstream_path"] = base.relative_to(upstream).as_posix()
                record["upstream_sha256"] = digest(base)
                if record["sha256"] == record["upstream_sha256"]:
                    # Supply byte-identical inputs from the pinned QuixiCore tree.
                    # Modified bindings/headers must remain the audited SlimServe ones.
                    shutil.copyfile(base, path)
                    record["origin"] = "quixicore"
                else:
                    record["origin"] = "slimserve-modified"
            records.append(record)
    if not records or not any(record["origin"] == "quixicore" for record in records):
        raise RuntimeError("Pinned QuixiCore source mapping produced no compiled inputs")
    return records


def build(args):
    if sys.version_info < (3, 11) or sys.version_info >= (3, 15):
        raise RuntimeError("Runtime plus SlimServe requires Python 3.11 through 3.14")
    if args.target == "metal" and (sys.platform != "darwin" or platform.machine() != "arm64"):
        raise RuntimeError("The native Metal build requires Apple Silicon macOS")
    if args.target == "cuda":
        if sys.platform != "linux" or platform.machine() != "x86_64":
            raise RuntimeError("This CUDA image recipe requires Linux x86_64")
        architectures = (args.cuda_arch_list or "").replace(";", " ").split()
        if not architectures:
            raise RuntimeError("Specify --cuda-arch-list for the actual target GPUs (SM80+)")
        if any(arch not in {"8.0", "8.6", "8.9"} for arch in architectures):
            raise RuntimeError(
                "This pinned source recipe supports SM80/86/89 only: upstream omits "
                "tools/build_deepgemm_C.py required by its SM90+ build path"
            )
        args.cuda_arch_list = ";".join(architectures)
        if not args.base_image or "@sha256:" not in args.base_image:
            raise RuntimeError("Supply an immutable, toolchain-equipped Runtime CUDA base image")
    if Path(sys.prefix) != PREFIX:
        if PREFIX.is_symlink() or (PREFIX.exists() and any(PREFIX.iterdir())):
            raise RuntimeError(f"Refusing to replace an existing installation at {PREFIX}")
        venv.EnvBuilder(with_pip=True).create(PREFIX)
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PATH"] = f"{PREFIX}/bin:{environment.get('PATH', '')}"
        python = str(PREFIX / "bin/python")
        os.execve(python, [python, "-I", str(Path(__file__).resolve()), *sys.argv[1:]], environment)

    dependencies = args.dependencies.resolve()
    dependency_lock = dependencies / "requirements.lock"
    if not dependency_lock.is_file():
        raise RuntimeError("The dependency wheelhouse must contain requirements.lock with SHA256 hashes")
    runtime_source = args.runtime_source.resolve()
    source_lock = json.loads(SOURCE_LOCK.read_text())
    sources = PREFIX / "sources"
    wheels = PREFIX / "wheels"
    wheels.mkdir()
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith(("VLLM_", "PIP_", "SETUPTOOLS_SCM_")) or name in {
            "CMAKE_ARGS", "TORCH_NIGHTLY", "DEEPGEMM_SRC_DIR", "FMHA_SM100_SRC_DIR",
            "FLASH_MLA_SRC_DIR", "QUTLASS_SRC_DIR", "TML_FA4_SRC_DIR", "TRITON_KERNELS_SRC_DIR",
            "DEEPGEMM_PYTHON_INTERPRETERS", "FETCHCONTENT_BASE_DIR",
        }:
            environment.pop(name)
    environment.update({
        "GIT_TERMINAL_PROMPT": "0",
        "VLLM_TARGET_DEVICE": args.target,
        "VLLM_USE_PRECOMPILED": "0",
        "TORCH_NIGHTLY": "0",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_NO_INDEX": "1",
        "PYTHONNOUSERSITE": "1",
        "CMAKE_BUILD_TYPE": "Release",
        "MAX_JOBS": str(args.max_jobs),
    })
    run(sys.executable, "-m", "pip", "install", "--no-index", "--only-binary=:all:",
        "--find-links", dependencies, "--require-hashes", "-r", dependency_lock, env=environment)
    reject_flashinfer()
    for name in ("vllm", "sovereign-runtime"):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Dependency lock must not include {name}; it is built from source")

    components = {"slimserve": source_lock["slimserve"], args.target + "-kernels": source_lock["kernels"][args.target]}
    if args.target == "cuda":
        components.update(source_lock["cuda_sources"])
    for name, source in components.items():
        fetch_source(sources / name, source, environment)
    slimserve = sources / "slimserve"
    require_build_dependencies(slimserve, runtime_source, args.target)
    kernel = source_lock["kernels"][args.target]
    source_files = prepare_kernels(slimserve, sources / (args.target + "-kernels"), kernel)
    environment["SOURCE_DATE_EPOCH"] = subprocess.check_output(
        ["git", "-C", str(slimserve), "show", "-s", "--format=%ct", "HEAD"], text=True
    ).strip()
    environment["FETCHCONTENT_BASE_DIR"] = str(PREFIX / "cmake-dependencies")
    environment["CMAKE_ARGS"] = "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    if args.target == "cuda":
        environment["TORCH_CUDA_ARCH_LIST"] = args.cuda_arch_list
        for name, source in source_lock["cuda_sources"].items():
            if "environment" in source:
                environment[source["environment"]] = str(sources / name / source.get("environment_subdirectory", ""))
        # CMake's standard source override avoids the development-only FA4 symlink
        # branch; upstream then copies Python files and rewrites package imports.
        environment["CMAKE_ARGS"] += (
            f" -DFETCHCONTENT_SOURCE_DIR_VLLM-FLASH-ATTN={sources / 'flash-attention'} "
            f"-DCUTLASS_INCLUDE_DIR={sources / 'cutlass/include'} "
            f"-DCUTLASS_TOOLS_UTIL_INCLUDE_DIR={sources / 'cutlass/tools/util/include'}"
        )

    for project in (slimserve, runtime_source):
        run(sys.executable, "-m", "pip", "wheel", "--no-index", "--no-deps",
            "--no-build-isolation", "--wheel-dir", wheels, project, env=environment)
    built_wheels = sorted(wheels.glob("*.whl"))
    if len(built_wheels) != 2:
        raise RuntimeError("Expected exactly the SlimServe and first-party Runtime wheels")
    run(sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", *built_wheels, env=environment)
    reject_flashinfer()
    # This runs when the recipe is invoked, not as a substitute for GPU qualification.
    run(sys.executable, "-m", "pip", "check", env=environment)
    compiler_commands = {
        "cmake": ["cmake", "--version"],
        "cxx": [os.environ.get("CXX", "c++"), "--version"],
    }
    if args.target == "cuda":
        nvcc = str(Path(environment["CUDA_HOME"]) / "bin/nvcc") if environment.get("CUDA_HOME") else "nvcc"
        compiler_commands["cuda"] = [nvcc, "--version"]
    else:
        compiler_commands["xcode"] = ["xcodebuild", "-version"]
        compiler_commands["metal"] = ["xcrun", "metal", "--version"]
    compilers = {
        name: subprocess.check_output(command, env=environment, text=True).strip()
        for name, command in compiler_commands.items()
    }

    site_packages = Path(sysconfig.get_path("purelib"))
    engine_package = site_packages / "vllm"
    extensions = [engine_package / ("_quixicore_C" + suffix)
                  for suffix in importlib.machinery.EXTENSION_SUFFIXES
                  if (engine_package / ("_quixicore_C" + suffix)).is_file()]
    if len(extensions) != 1:
        raise RuntimeError("The build must install exactly one native vllm._quixicore_C extension")
    kernel_artifacts = extensions
    if args.target == "metal":
        kernel_artifacts.append(engine_package / "quixicore_metal.metallib")
    profiles = site_packages / "slimserve/profiles.json"
    python_files = sorted(path for package in ("vllm", "slimserve")
                          for path in (site_packages / package).rglob("*.py"))
    if not python_files or not profiles.is_file():
        raise RuntimeError("The installed wheel lacks SlimServe Python sources or profiles.json")
    licenses = []
    for name in components:
        if "/" not in name:
            # Parent source notices already include the independently pinned
            # nested source trees; do not duplicate their files or records.
            licenses.extend(preserve_licenses(sources / name, PREFIX / "licenses" / name))
    for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"):
        target = PREFIX / "licenses/sovereign-runtime" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runtime_source / name, target)
        licenses.append(file_record(target, PREFIX))
    modification = source_lock["slimserve"]["source_commit"]
    provenance = {
        "schema_version": 1,
        "slimserve": {
            **source_lock["slimserve"],
            "package_version": importlib.metadata.version("vllm"),
            "files": [file_record(path, site_packages) for path in [*python_files, profiles]],
        },
        "kernels": {
            "library": kernel["library"],
            "source_repository": kernel["source_repository"],
            "source_commit": kernel["source_commit"],
            "modifications_commit": modification,
            "version": kernel["source_commit"] + "+slimserve." + modification,
            "module_name": "vllm._quixicore_C",
            "source_files": source_files,
            "artifacts": [file_record(path, site_packages) for path in kernel_artifacts],
        },
        "build": {
            "python_version": platform.python_version(),
            "builder_sha256": digest(Path(__file__)),
            "compilers": compilers,
            "target": args.target,
            "base_image": args.base_image,
            "cuda_arch_list": args.cuda_arch_list,
            "source_date_epoch": environment["SOURCE_DATE_EPOCH"],
            "source_lock_sha256": digest(SOURCE_LOCK),
            "dependency_lock_sha256": digest(dependency_lock),
            "sources": components,
            "wheels": [file_record(path, PREFIX) for path in built_wheels],
            "packages": sorted(
                ({"name": dist.metadata["Name"], "version": dist.version}
                 for dist in importlib.metadata.distributions()), key=lambda item: item["name"].lower()
            ),
        },
        "compiled_artifacts": [file_record(path, site_packages)
                               for path in sorted(engine_package.rglob("*"))
                               if path.is_file() and path.suffix in {".so", ".dylib", ".metallib"}],
        "licenses": licenses,
    }
    (PREFIX / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    shutil.copyfile(SOURCE_LOCK, PREFIX / "source-lock.json")
    shutil.copyfile(dependency_lock, PREFIX / "requirements.lock")
    shutil.rmtree(sources)
    cmake_dependencies = PREFIX / "cmake-dependencies"
    if cmake_dependencies.exists():
        shutil.rmtree(cmake_dependencies)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("cuda", "metal"), required=True)
    parser.add_argument("--dependencies", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--base-image")
    parser.add_argument("--cuda-arch-list")
    parser.add_argument("--max-jobs", type=int, default=4)
    build(parser.parse_args())
