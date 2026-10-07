import os
import subprocess
import sys

import pytest

from lazarus.agent.memory import memory_bytes


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
