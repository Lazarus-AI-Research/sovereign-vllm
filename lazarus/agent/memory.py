"""How much memory a process holds, as the host counts it, with every
process it started: SlimServe's server holds its model in an engine process
of its own.

On macOS this is each process's physical footprint, the figure Activity
Monitor shows: a Metal engine's weights sit in unified memory the resident
set does not fully count. On Linux it is the resident set. None where the
host cannot say.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from pathlib import Path

_RUSAGE_INFO_V2 = 2


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def memory_bytes(pid: int) -> int | None:
    own = _own_bytes(pid)
    if own is None:
        return None
    return own + sum(_own_bytes(child) or 0 for child in _descendants(pid))


def _own_bytes(pid: int) -> int | None:
    if sys.platform == "darwin":
        return _darwin_footprint(pid)
    if sys.platform.startswith("linux"):
        return _linux_resident(pid)
    return None


def _descendants(pid: int) -> list[int]:
    found, pending = [], [pid]
    while pending:
        children = _children(pending.pop())
        found += children
        pending += children
    return found


def _children(pid: int) -> list[int]:
    if sys.platform == "darwin":
        return _darwin_children(pid)
    if sys.platform.startswith("linux"):
        try:
            return [int(child) for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
        except OSError:
            return []
    return []


def _libproc():
    path = ctypes.util.find_library("proc") or "/usr/lib/libSystem.dylib"
    try:
        return ctypes.CDLL(path, use_errno=True)
    except OSError:
        return None


def _darwin_children(pid: int) -> list[int]:
    library = _libproc()
    if library is None:
        return []
    buffer = (ctypes.c_int * 256)()
    count = library.proc_listchildpids(ctypes.c_int(pid), buffer, ctypes.sizeof(buffer))
    return [child for child in buffer[:max(count, 0)] if child > 0]


def _darwin_footprint(pid: int) -> int | None:
    library = _libproc()
    if library is None:
        return None
    info = _RusageInfoV2()
    if library.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V2), ctypes.byref(info)) != 0:
        return None
    return int(info.ri_phys_footprint)


def _linux_resident(pid: int) -> int | None:
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None
