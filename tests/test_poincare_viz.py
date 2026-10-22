import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from dynachaos.diagnostics.poincare import poincare_section  # noqa: E402
from dynachaos.viz import poincare_section_plot  # noqa: E402


def _periodic_signal(n: int = 4096, dt: float = 0.01) -> tuple[np.ndarray, float]:
    t = np.arange(n, dtype=np.float64) * dt
    x = np.sin(2.0 * np.pi * 2.0 * t)
    return x, 1.0 / dt


def test_poincare_section_plot_auto_plane_with_metrics_box():
    signal, fs = _periodic_signal()
    section = poincare_section(signal, fs)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    out_ax = poincare_section_plot(section, ax=ax, show_metrics=True, kind="single")

    assert out_ax is ax
    assert ax.get_xlabel() == "x(t - delay)"
    assert ax.get_ylabel() == "x(t + delay)"
    assert len(ax.collections) == 1

    text_blob = "\n".join(text.get_text() for text in ax.texts)
    assert "Crossings:" in text_blob
    assert "Section: signal_delay_pair" in text_blob
    plt.close(fig)


def test_poincare_section_plot_allows_plane_override():
    signal, fs = _periodic_signal()
    section = poincare_section(signal, fs)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    poincare_section_plot(section, ax=ax, plane="signal_derivative", show_metrics=False)

    assert ax.get_xlabel() == "x(t)"
    assert ax.get_ylabel() == "dx/dt(t)"
    plt.close(fig)


def test_poincare_section_plot_handles_empty_section_data():
    section = {
        "planes": {},
        "section_points": np.empty((0, 2), dtype=np.float64),
        "metrics": {
            "num_crossings": 0,
            "mean_period": np.nan,
            "coefficient_of_variation": np.nan,
            "spectral_peak_ratio": np.nan,
        },
        "section_plane_type": "unavailable",
    }

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    poincare_section_plot(section, ax=ax, show_metrics=True)

    texts = [text.get_text() for text in ax.texts]
    assert any("Insufficient crossings for section" in txt for txt in texts)
    plt.close(fig)
