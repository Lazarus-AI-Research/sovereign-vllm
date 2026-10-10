"""How much memory a process holds, as the host counts it, with every
process it started: SlimServe's server holds its model in an engine process
of its own.

On macOS this is each process's physical footprint, the figure Activity
Monitor shows, and the weights it maps from the models directory: llama.cpp
maps its model file rather than copying it, and a mapped file's pages are
the file's, not the process's, so the footprint leaves them out. They count
at the size mapped, what the model holds once warm, not at the pages in
memory now, which fall under memory pressure and for experts not yet used.
On Linux it is the resident set, which counts mapped pages already. None
where the host cannot say.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time
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
        # A stretch of a file two of the family map counts once.
        mapped: dict[str, list[tuple[int, int]]] = {}
        for member in family:
            for path, extents in _darwin_mapped_weights(member, weights_root).items():
                mapped.setdefault(path, []).extend(extents)
        weights = sum(_covered_bytes(extents) for extents in mapped.values())
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


class _RegionWithPath(ctypes.Structure):
    # The region and the file it maps, read together: naming the file of a
    # region in a second call can name the next region's file instead.
    _fields_ = [
        ("region", _RegionInfo),
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_times", ctypes.c_int64 * 8),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        *((name, ctypes.c_uint32) for name in ("vst_blksize", "vst_flags", "vst_gen", "vst_rdev")),
        ("vst_qspare", ctypes.c_int64 * 2),
        ("vi_type", ctypes.c_int32),
        ("vi_pad", ctypes.c_int32),
        ("vi_fsid", ctypes.c_int32 * 2),
        ("vip_path", ctypes.c_char * 1024),
    ]


_PROC_PIDREGIONPATHINFO = 8
# A process maps its weights once it has loaded them and keeps them mapped,
# so the walk over its regions, the slow part, is made once per process and
# directory: a process known by its id and when it started, since an id is
# used again. One that maps nothing yet may still be loading, so it is
# walked again, but not on every look.
_mapped_weights: dict[tuple[int, int, str], tuple[float, dict[str, tuple[tuple[int, int], ...]]]] = {}
_UNMAPPED_RECHECK_SECONDS = 30.0


def _darwin_mapped_weights(pid: int, root: Path) -> dict[str, tuple[tuple[int, int], ...]]:
    """Each file the process maps from under the root, with the stretches
    of it mapped, as start and end offsets."""
    library = _libproc()
    if library is None:
        return {}
    usage = _RusageInfoV2()
    if library.proc_pid_rusage(ctypes.c_int(pid), ctypes.c_int(_RUSAGE_INFO_V2), ctypes.byref(usage)) != 0:
        return {}
    prefix = str(root).rstrip("/") + "/"
    process = (pid, int(usage.ri_proc_start_abstime), prefix)
    cached = _mapped_weights.get(process)
    if cached is not None and (cached[1] or time.monotonic() - cached[0] < _UNMAPPED_RECHECK_SECONDS):
        return cached[1]
    found: dict[str, list[tuple[int, int]]] = {}
    address = 0
    while True:
        info = _RegionWithPath()
        if library.proc_pidinfo(ctypes.c_int(pid), ctypes.c_int(_PROC_PIDREGIONPATHINFO), ctypes.c_uint64(address), ctypes.byref(info), ctypes.sizeof(info)) < ctypes.sizeof(info):
            break
        region = info.region
        name = os.fsdecode(info.vip_path)
        if name.startswith(prefix):
            # The last page mapped runs past the end of the file.
            end = min(region.pri_offset + region.pri_size, info.vst_size)
            found.setdefault(name, []).append((region.pri_offset, end))
        address = region.pri_address + region.pri_size
    weights = {name: tuple(extents) for name, extents in found.items()}
    if len(_mapped_weights) > 256:
        _mapped_weights.clear()
    _mapped_weights[process] = (time.monotonic(), weights)
    return weights


def _covered_bytes(extents: list[tuple[int, int]]) -> int:
    """The bytes of a file its mappings cover, a stretch two of them map
    counted once."""
    covered, reached = 0, 0
    for start, end in sorted(extents):
        start = max(start, reached)
        if end > start:
            covered += end - start
            reached = end
    return covered
