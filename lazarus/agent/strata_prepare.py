"""Builds what Strata serves a deployment from out of its verified files, and
writes the run settings its server starts with. It runs in Strata's own
Python, beside Strata's tools, and reaches no network: the GGUF files and
the draft layer's tensors are the ones the agent was given and checked.

    <strata>/python/bin/python3 -I strata_prepare.py --strata DIR --request FILE --out DIR [--settings FILE]

The request file is the deployment as the agent records it: its snapshot
directory, its files with their tensors, and for the settings its window,
its requests and its thinking. The pack and the draft layer depend only on
the files, so they are built once per snapshot under --out and kept; the
settings are written for each start, and without --settings nothing but the
build is done.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# A size's first shard holds the model and its experts, its second the
# n-gram table the engine reads a few rows at a time.
SHARD = re.compile(r"-GSQ-RCO-(?P<size>[A-Z0-9_]+)-0000(?P<index>[12])-of-00002\.gguf$")
DONE = ".done"
# Strata's prompt kernels multiply on Metal 4's matrix units, which macOS
# 26 brought: before it they do not compile, and the engine exits at its
# first prompt. There prompts go through the decode window instead, eight
# tokens at a time, and the probe of the M5's matrix layout is skipped
# (measured on an M3 Max with macOS 15.7.9).
BEFORE_METAL_4 = {"STRATA_METAL_PROMPT": "window", "STRATA_METAL_PROMPT_GEMM": "0", "COOP_K4": "0"}


def metal_4() -> bool:
    release = platform.mac_ver()[0]
    return bool(release) and int(release.split(".")[0]) >= 26


def shards(files: dict) -> tuple[str, Path, Path]:
    found: dict[int, str] = {}
    size = None
    for name in files:
        match = SHARD.search(name)
        if match:
            size = size or match["size"]
            if match["size"] != size:
                raise SystemExit("the files hold more than one size of the model")
            found[int(match["index"])] = name
    if set(found) != {1, 2}:
        raise SystemExit("the files hold the model's two GGUF shards")
    return size, Path(found[1]), Path(found[2])


def run(command: list, environment: dict) -> None:
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.run([sys.executable, "-I", *(str(part) for part in command)], env=environment, check=True)


def build_pack(strata: Path, shard: Path, pack: Path, environment: dict) -> None:
    if (pack / DONE).is_file():
        return
    shutil.rmtree(pack, ignore_errors=True)
    run([strata / "tools" / "iq_pack.py", "--gguf", shard, "--out", pack], environment)
    (pack / DONE).write_text("")


# The draft layer is the checkpoint's MTP block, quantized as Strata's setup
# does it; the manifest names each tensor where the agent placed it.
def build_draft(strata: Path, snapshot: Path, tensors: dict, files: dict, draft: Path, environment: dict) -> Path:
    model = draft / "mtp-q2_0.gguf"
    if (draft / DONE).is_file():
        return model
    shutil.rmtree(draft, ignore_errors=True)
    draft.mkdir(parents=True)
    manifest = [
        {"name": tensor["name"], "dtype": tensor["dtype"], "shape": tensor["shape"],
         "file": str(snapshot / path), "sha256": files[path], "bytes": (snapshot / path).stat().st_size}
        for path, tensor in sorted(tensors.items(), key=lambda item: item[1]["name"])
    ]
    (draft / "mtp-manifest.json").write_text(json.dumps(manifest, indent=1))
    run([strata / "tools" / "mtp_pack.py", "--src", draft, "--experts", "q2_0", "--out", model], environment)
    # The Metal engine reads the draft model beside its token subset.
    (draft / "rt").mkdir()
    shutil.copyfile(strata / "data" / "draft_vocab.bin", draft / "rt" / "draft_vocab.bin")
    (draft / DONE).write_text("")
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--settings")
    parser.add_argument("--memory-gb", type=float, required=True)
    arguments = parser.parse_args()
    strata = Path(arguments.strata).resolve()
    sys.path.insert(0, str(strata))
    from strata_control import mac_fit

    request = json.loads(Path(arguments.request).read_text())
    snapshot = Path(request["model_path"])
    files, tensors = request["strata"]["files"], request["strata"]["tensors"]
    size, first, second = shards(files)
    if size not in mac_fit.MAC_MODELS:
        raise SystemExit(f"{size} is not a size Strata serves on a Mac")
    out = Path(arguments.out)
    environment = dict(os.environ, STRATA_GGUF_PY=str(strata / "third_party" / "llama.cpp" / "gguf-py"))
    build_pack(strata, snapshot / first, out / "pack", environment)
    streams = mac_fit.streams(size, arguments.memory_gb)
    drafts = mac_fit.stream_mtp(streams, arguments.memory_gb)
    if drafts:
        build_draft(strata, snapshot, tensors, files, out / "draft", environment)
    if arguments.settings is None:
        return 0
    window = request["context_length"]
    engine_arguments = mac_fit.engine_args(
        out / "pack", [snapshot / first, snapshot / second], out / "draft" / "rt", window, streams, arguments.memory_gb,
        strata / "data" / "expert-profile.bin",
    )
    settings = {
        "exe": str(strata / "engine" / "strata"), "args": engine_arguments, "cwd": str(strata),
        "tokenizer": str(out / "pack" / "tokenizer"), "model_name": request["served_model_name"],
        "log": str(Path(arguments.settings).with_suffix(".log")), "lib_dirs": [], "backend": "metal",
        "env": {"STRATA_METAL_STREAM": "1" if streams else "0", **({} if metal_4() else BEFORE_METAL_4)},
        "parallel": request["requests"],
    }
    if request.get("thinking") == "off":
        settings["reasoning_effort"] = "none"
    Path(arguments.settings).write_text(json.dumps(settings, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
