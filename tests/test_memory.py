import os
import subprocess
import sys

import pytest

from lazarus.agent import memory
from lazarus.agent.memory import memory_bytes, memory_held


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="the host counts memory only on macOS and Linux")
def test_a_running_process_holds_memory_and_a_gone_one_holds_none():
    assert memory_bytes(os.getpid()) > 0
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert memory_bytes(child.pid) is None


@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="the host counts memory only on macOS and Linux")
def test_a_process_is_counted_with_the_processes_it_started():
    # The parent holds little; the child it waits on holds 512 MiB.
    child_code = "b = bytearray(512 * 1024 * 1024); import time; time.sleep(30)"
    parent = subprocess.Popen([sys.executable, "-c", f"import subprocess, sys; subprocess.run([sys.executable, '-c', {child_code!r}])"])
    try:
        import time

        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and (memory_bytes(parent.pid) or 0) < 512 * 1024 * 1024:
            time.sleep(0.2)
        assert memory_bytes(parent.pid) >= 512 * 1024 * 1024
    finally:
        parent.kill()
        subprocess.run(["pkill", "-f", "bytearray\\(512"], check=False)


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS leaves mapped weights out of a process's footprint")
def test_the_weights_a_process_maps_from_the_models_directory_count_as_its_own(tmp_path):
    weights = tmp_path / "models" / "weights.gguf"
    weights.parent.mkdir()
    weights.write_bytes(os.urandom(64 * 1024 * 1024))
    # Like llama.cpp: the file mapped, every page read, and kept mapped.
    code = (
        "import mmap, sys, time; f = open(sys.argv[1], 'rb'); m = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ); "
        "sum(m[i] for i in range(0, len(m), 4096)); print('ready', flush=True); time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(weights)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        alone = memory_bytes(child.pid)
        with_weights = memory_bytes(child.pid, tmp_path / "models")
        elsewhere = memory_bytes(child.pid, tmp_path / "other")
        assert with_weights - alone >= 60 * 1024 * 1024
        assert elsewhere - alone < 8 * 1024 * 1024
    finally:
        child.kill()


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS leaves mapped weights out of a process's footprint")
def test_mapped_weights_count_at_the_size_mapped_not_the_pages_in_memory_now(tmp_path):
    # A sparse file mapped and never read: none of its pages is in memory,
    # as for experts not yet used, yet the model holds all of it when warm.
    whole = tmp_path / "models" / "whole.gguf"
    part = tmp_path / "models" / "part.gguf"
    whole.parent.mkdir()
    for file in (whole, part):
        with open(file, "wb") as handle:
            handle.truncate(64 * 1024 * 1024)
    code = (
        "import mmap, sys, time; w = open(sys.argv[1], 'rb'); p = open(sys.argv[2], 'rb'); "
        "a = mmap.mmap(w.fileno(), 0, prot=mmap.PROT_READ); b = mmap.mmap(w.fileno(), 0, prot=mmap.PROT_READ); "
        "c = mmap.mmap(p.fileno(), 16 * 1024 * 1024, prot=mmap.PROT_READ, offset=16 * 1024 * 1024); "
        "print('ready', flush=True); time.sleep(30)"
    )
    child = subprocess.Popen([sys.executable, "-c", code, str(whole), str(part)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
        _, weights = memory_held(child.pid, tmp_path / "models")
        # The whole file mapped twice counts once; of the other, the 16 MiB mapped.
        assert weights == 80 * 1024 * 1024
    finally:
        child.kill()


@pytest.mark.skipif(sys.platform != "darwin", reason="only macOS walks a process's mapped weights")
def test_a_process_that_maps_no_weights_is_not_walked_again_on_every_look(tmp_path, monkeypatch):
    memory._mapped_weights.clear()
    memory_held(os.getpid(), tmp_path)
    assert [weights for (pid, _, _), (_, weights) in memory._mapped_weights.items() if pid == os.getpid()] == [{}]
    library = memory._libproc()
    monkeypatch.setattr(memory, "_libproc", lambda: library)
    monkeypatch.setattr(library, "proc_pidinfo", lambda *arguments: pytest.fail("walked again"))
    memory_held(os.getpid(), tmp_path)
    # One still loading maps its weights later, so it is walked again in time.
    monkeypatch.setattr(memory, "_UNMAPPED_RECHECK_SECONDS", 0.0)
    with pytest.raises(pytest.fail.Exception):
        memory_held(os.getpid(), tmp_path)

def test_stretches_of_a_file_mapped_twice_count_once():
    assert memory._covered_bytes([(0, 10), (5, 20), (30, 40), (32, 35)]) == 30
