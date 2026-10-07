"""How much memory a process holds, as the host counts it, with every
process it started: SlimServe's server holds its model in an engine process
of its own.

On macOS this is each process's physical footprint, the figure Activity
Monitor shows, and the weights it maps from the models directory: llama.cpp
maps its model file rather than copying it, and a mapped file's pages are
the file's, not the process's, so the footprint leaves them out. On Linux it
is the resident set, which counts mapped pages already. None where the host
cannot say.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import mmap
import os
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


def memory_bytes(pid: int, weights_root: Path | None = None) -> int | None:
    held = memory_held(pid, weights_root)
    return None if held is None else held[0]


def memory_held(pid: int, weights_root: Path | None = None) -> tuple[int, int] | None:
    """All the process and its children hold, and of that the weights they
    map, which macOS counts as files in memory rather than memory in use: a
    reader of the machine's memory in use adds them to it."""
    own = _own_bytes(pid)
    if own is None:
        return None
    family = [pid, *_descendants(pid)]
    held = own + sum(_own_bytes(child) or 0 for child in family[1:])
    weights = 0
    if sys.platform == "darwin" and weights_root is not None:
        # A file two of the family map is in memory once.
        mapped = set().union(*(_darwin_mapped_weights(member, weights_root) for member in family))
        weights = sum(_resident_bytes(path) for path in mapped)
    return held + weights, weights


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


class _RegionInfo(ctypes.Structure):
    _fields_ = [
        *((name, ctypes.c_uint32) for name in ("pri_protection", "pri_max_protection", "pri_inheritance", "pri_flags")),
        ("pri_offset", ctypes.c_uint64),
        *((name, ctypes.c_uint32) for name in (
            "pri_behavior", "pri_user_wired_count", "pri_user_tag", "pri_pages_resident", "pri_pages_shared_now_private",
            "pri_pages_swapped_out", "pri_pages_dirtied", "pri_ref_count", "pri_shadow_depth", "pri_share_mode",
            "pri_private_pages_resident", "pri_shared_pages_resident", "pri_obj_id", "pri_depth",
        )),
        ("pri_address", ctypes.c_uint64),
        ("pri_size", ctypes.c_uint64),
    ]


_PROC_PIDREGIONINFO = 7
# A process maps its weights once it has loaded them and keeps them mapped,
# so the walk over its regions, the slow part, is made once per process and
# directory: a process known by its id and when it started, since an id is
# used again.
_mapped_weights: dict[tuple[int, int, str], frozenset[str]] = {}


def _darwin_mapped_weights(pid: int, root: Path) -> frozenset[str]:
    library = _libproc()
    if library is None:
        return frozenset()
    usage = _RusageInfoV2()
    if library.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V2), ctypes.byref(usage)) != 0:
        return frozenset()
    prefix = str(root).rstrip("/") + "/"
    process = (pid, int(usage.ri_proc_start_abstime), prefix)
    if process in _mapped_weights:
        return _mapped_weights[process]
    found: set[str] = set()
    path = ctypes.create_string_buffer(4096)
    address = 0
    while True:
        info = _RegionInfo()
        if library.proc_pidinfo(ctypes.c_int(pid), ctypes.c_int(_PROC_PIDREGIONINFO), ctypes.c_uint64(address), ctypes.byref(info), ctypes.sizeof(info)) < ctypes.sizeof(info):
            break
        if library.proc_regionfilename(ctypes.c_int(pid), ctypes.c_uint64(info.pri_address), path, ctypes.sizeof(path)) > 0:
            name = os.fsdecode(path.value)
            if name.startswith(prefix):
                found.add(name)
        address = info.pri_address + info.pri_size
    weights = frozenset(found)
    # Until the model is loaded nothing is mapped yet; look again then.
    if weights:
        if len(_mapped_weights) > 256:
            _mapped_weights.clear()
        _mapped_weights[process] = weights
    return weights


# Each page's flags, as mincore writes them, to 1 when the page is in memory.
_IN_MEMORY = bytes(flags & 1 for flags in range(256))


def _libc():
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    return libc


def _resident_bytes(path: str) -> int:
    """How much of the file is in memory, whoever mapped it."""
    libc = _libc()
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return 0
    try:
        size = os.fstat(descriptor).st_size
        if size == 0:
            return 0
        address = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, descriptor, 0)
    finally:
        os.close(descriptor)
    if address in (None, ctypes.c_void_p(-1).value):
        return 0
    try:
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        vector = ctypes.create_string_buffer(pages)
        if libc.mincore(address, size, vector) != 0:
            return 0
        return min(size, vector.raw[:pages].translate(_IN_MEMORY).count(1) * mmap.PAGESIZE)
    finally:
        libc.munmap(address, size)
