"""Coverage tests for dynachaos.utils.animation.

make_attractor_gif renders through matplotlib's PillowWriter (pure-Python
GIF encoder, no ffmpeg binary), so unlike the paper-scale animation sweeps
in the maps/ modules that call it, it is cheap enough to exercise directly
with a handful of frames.
"""

import numpy as np

from dynachaos.utils.animation import compute_animation_sweep, make_attractor_gif


def test_compute_animation_sweep_fills_frames_from_iterate_fn(tmp_path):
    output_path = tmp_path / "sweep.npz"
    param_values = np.array([1.0, 1.1, 1.2])

    def iterate_fn(p):
        return np.column_stack([np.full(4, p), np.full(4, -p)])

    result = compute_animation_sweep(iterate_fn, param_values, output_path, n_plot=4)

    assert result["all_x"].shape == (3, 4)
    np.testing.assert_allclose(result["all_x"][1], 1.1)
    np.testing.assert_allclose(result["all_y"][1], -1.1)
    with np.load(output_path) as saved:
        np.testing.assert_allclose(saved["param_values"], param_values)


def test_compute_animation_sweep_fills_nan_when_diverged(tmp_path):
    output_path = tmp_path / "sweep.npz"
    param_values = np.array([1.0, 2.0])

    def iterate_fn(p):
        return None if p > 1.5 else np.zeros((4, 2))

    result = compute_animation_sweep(iterate_fn, param_values, output_path, n_plot=4)

    assert np.all(np.isfinite(result["all_x"][0]))
    assert np.all(np.isnan(result["all_x"][1]))
    assert np.all(np.isnan(result["all_y"][1]))


def test_compute_animation_sweep_pads_short_trajectories_with_nan(tmp_path):
    output_path = tmp_path / "sweep.npz"
    param_values = np.array([1.0])

    def iterate_fn(_p):
        # Fewer points than n_plot requests.
        return np.ones((2, 2))

    result = compute_animation_sweep(iterate_fn, param_values, output_path, n_plot=5)

    np.testing.assert_allclose(result["all_x"][0, :2], 1.0)
    assert np.all(np.isnan(result["all_x"][0, 2:]))


def test_compute_animation_sweep_writes_progress_checkpoints(tmp_path):
    output_path = tmp_path / "sweep.npz"
    param_values = np.linspace(0.0, 1.0, 4)

    def iterate_fn(p):
        return np.column_stack([np.full(3, p), np.full(3, p)])

    compute_animation_sweep(iterate_fn, param_values, output_path, n_plot=3, progress_interval=1)

    with np.load(output_path) as saved:
        assert saved["param_values"].shape == (4,)


def test_make_attractor_gif_renders_frames_with_finite_extent(tmp_path):
    output_path = tmp_path / "attractor.gif"
    param_values = np.array([1.0, 1.1, 1.2])
    rng = np.random.default_rng(0)
    all_x = rng.uniform(-1, 1, (3, 20))
    all_y = rng.uniform(-1, 1, (3, 20))

    result = make_attractor_gif(
        param_values,
        all_x,
        all_y,
        output_path,
        title_template=r"$D = {param_value}$",
        param_name="D",
        param_fmt=".2f",
        fps=5,
        dpi=50,
    )

    assert result == output_path
    assert output_path.stat().st_size > 0


def test_make_attractor_gif_ignores_nan_and_inf_when_computing_axis_limits(tmp_path):
    output_path = tmp_path / "attractor.gif"
    param_values = np.array([1.0, 1.1])
    all_x = np.array([[0.0, 1.0, np.nan], [np.inf, -1.0, 0.5]])
    all_y = np.array([[0.0, -1.0, np.nan], [-np.inf, 1.0, 0.0]])

    result = make_attractor_gif(param_values, all_x, all_y, output_path, fps=5, dpi=50)

    assert result == output_path
    assert output_path.stat().st_size > 0
