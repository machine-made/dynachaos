"""Coverage tests for dynachaos.utils.system.

The `resource` module is POSIX-only; the None fallback is normally only
reached on Windows, so it is forced here directly. The Windows branch is
exercised by faking the ctypes psapi call.
"""

from dynachaos.utils import system


def test_get_rss_mb_returns_zero_when_resource_module_unavailable(monkeypatch):
    monkeypatch.setattr(system.sys, "platform", "linux")
    monkeypatch.setattr(system, "resource", None)

    assert system.get_rss_mb() == 0.0


def test_get_rss_mb_returns_a_positive_finite_value_on_this_platform():
    rss = system.get_rss_mb()

    assert rss > 0.0


def test_get_rss_mb_uses_peak_working_set_on_windows(monkeypatch):
    """Drive the win32 branch on Linux by faking the ctypes psapi call."""
    peak_bytes = 512 * 2**20

    class _FakePsapi:
        @staticmethod
        def GetProcessMemoryInfo(_handle, counters_ptr, _size):
            counters_ptr._obj.PeakWorkingSetSize = peak_bytes
            return 1

    class _FakeKernel32:
        @staticmethod
        def GetCurrentProcess():
            return -1

    class _FakeWindll:
        psapi = _FakePsapi
        kernel32 = _FakeKernel32

    monkeypatch.setattr(system.sys, "platform", "win32")
    monkeypatch.setattr(system.ctypes, "windll", _FakeWindll, raising=False)

    assert system.get_rss_mb() == 512.0
