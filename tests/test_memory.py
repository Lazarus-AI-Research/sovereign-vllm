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
