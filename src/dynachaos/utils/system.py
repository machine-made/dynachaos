"""System-level utilities (memory, resource usage)."""

import ctypes
import sys

try:
    import resource
except ImportError:  # pragma: no cover - exercised on Windows CI
    resource = None


def get_rss_mb() -> float:
    """Return the process peak RSS in megabytes (MB = 2**20 bytes).

    Every platform reports the peak, not the current value: ru_maxrss on
    POSIX, PeakWorkingSetSize on Windows.
    """
    if sys.platform == "win32":
        return _get_rss_mb_windows()
    if resource is None:
        return 0.0

    rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS: ru_maxrss is in bytes; Linux: in KB
    return rss_raw / (1024 * 1024) if sys.platform == "darwin" else rss_raw / 1024


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _get_rss_mb_windows() -> float:
    """Return the process peak working set in MB via psapi GetProcessMemoryInfo."""
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    # Declare the handle as pointer-sized. The ctypes default is a 32-bit int,
    # which can corrupt the pseudo-handle on 64-bit Windows.
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory_info.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
        ctypes.c_ulong,
    ]
    get_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / 2**20
