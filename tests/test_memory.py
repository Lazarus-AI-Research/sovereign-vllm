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
