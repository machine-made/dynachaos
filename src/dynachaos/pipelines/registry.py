"""Section registry for dynachaos figure/data pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dynachaos.io.paths import section_dir


@dataclass(frozen=True)
class NpzContract:
    """Required keys for one generated ``.npz`` artifact."""

    file_name: str
    required_keys: tuple[str, ...]


@dataclass(frozen=True)
class FigureSpec:
    """One page figure: its artifact, the module that produces it, and its demo snippet.

    ``snippet`` is a repo-relative path under ``examples/snippets/`` (or ``None``
    where a demo has not been written yet).
    """

    png: str
    npz: str | None
    module: str
    snippet: str | None = None


@dataclass(frozen=True)
class SectionSpec:
    """Pipeline specification for one paper section."""

    section_id: str
    modules: tuple[str, ...]
    cache_files: tuple[str, ...]
    output_files: tuple[str, ...]
    npz_contracts: tuple[NpzContract, ...] = ()
    figures: tuple[FigureSpec, ...] = ()

    def cache_paths(self) -> tuple[Path, ...]:
        base = section_dir(self.section_id)
        return tuple(base / name for name in self.cache_files)

    def output_paths(self) -> tuple[Path, ...]:
        base = section_dir(self.section_id)
        return tuple(base / name for name in self.output_files)

    def required_npz_keys(self, file_name: str) -> tuple[str, ...]:
        """Return required keys for an artifact, or an empty tuple if unconstrained."""
        for contract in self.npz_contracts:
            if contract.file_name == file_name:
                return contract.required_keys
        return ()


SECTION_ORDER = (
    "sec02_circle_map",
    "sec03_transition",
    "sec04_doubling",
    "sec05_oscillation",
    "sec06_three_torus",
    "sec07_fractalization",
    "sec08_sti",
    "sec09_pattern",
    "sec10_gcm",
    "sec11_diagnostics",
    "sec12_intermittency",
)


SECTION_SPECS = {
    "sec02_circle_map": SectionSpec(
        section_id="sec02_circle_map",
        modules=("dynachaos.maps.circle_map", "dynachaos.maps.arnold_tongues"),
        cache_files=("devils_staircase.npz", "arnold_tongues.npz", "staircase_zoom.npz"),
        output_files=(
            "devils_staircase.npz",
            "devils_staircase.png",
            "arnold_tongues.npz",
            "arnold_tongues.png",
            "staircase_zoom.npz",
            "staircase_zoom.png",
        ),
        npz_contracts=(
            NpzContract("devils_staircase.npz", ("A", "rho", "lam")),
            NpzContract("arnold_tongues.npz", ("Omega", "K", "rho")),
            NpzContract("staircase_zoom.npz", ("A", "rho")),
        ),
        figures=(
            FigureSpec(
                "devils_staircase.png",
                "devils_staircase.npz",
                "dynachaos.maps.circle_map",
                "examples/snippets/sec02_circle_map/devils_staircase.py",
            ),
            FigureSpec(
                "arnold_tongues.png",
                "arnold_tongues.npz",
                "dynachaos.maps.arnold_tongues",
                "examples/snippets/sec02_circle_map/arnold_tongues.py",
            ),
            FigureSpec(
                "staircase_zoom.png",
                "staircase_zoom.npz",
                "dynachaos.maps.circle_map",
                "examples/snippets/sec02_circle_map/staircase_zoom.py",
            ),
        ),
    ),
    "sec03_transition": SectionSpec(
        section_id="sec03_transition",
        modules=("dynachaos.maps.coupled_logistic",),
        cache_files=("phase_diagram.npz", "attractors.npz", "basins.npz"),
        output_files=(
            "phase_diagram.npz",
            "attractors.npz",
            "basins.npz",
            "phase_diagram.png",
            "attractors.png",
            "basins.png",
        ),
        npz_contracts=(
            NpzContract("phase_diagram.npz", ("A", "D", "asym", "lyap", "schema_version")),
            NpzContract(
                "attractors.npz",
                (
                    "A_values",
                    "labels",
                    "initial_states",
                    "x_limits",
                    "y_limits",
                    "D",
                    "schema_version",
                ),
            ),
            NpzContract("basins.npz", ("x", "y", "basin", "A", "D")),
        ),
        figures=(
            FigureSpec(
                "phase_diagram.png",
                "phase_diagram.npz",
                "dynachaos.maps.coupled_logistic",
                "examples/snippets/sec03_transition/phase_diagram.py",
            ),
            FigureSpec(
                "attractors.png",
                "attractors.npz",
                "dynachaos.maps.coupled_logistic",
                "examples/snippets/sec03_transition/attractors.py",
            ),
            FigureSpec(
                "basins.png",
                "basins.npz",
                "dynachaos.maps.coupled_logistic",
                "examples/snippets/sec03_transition/basins.py",
            ),
        ),
    ),
    "sec04_doubling": SectionSpec(
        section_id="sec04_doubling",
        modules=("dynachaos.maps.torus_doubling",),
        cache_files=("map_I_attractors.npz", "map_IV_attractors.npz", "map_IV_lyapunov.npz"),
        output_files=(
            "map_I_attractors.npz",
            "map_IV_attractors.npz",
            "map_IV_lyapunov.npz",
            "map_I_attractors.png",
            "map_IV_attractors.png",
            "map_IV_lyapunov.png",
        ),
        npz_contracts=(
            NpzContract("map_I_attractors.npz", ("D_values",)),
            NpzContract("map_IV_attractors.npz", ("D_values",)),
            NpzContract("map_IV_lyapunov.npz", ("D", "spectra")),
        ),
        figures=(
            FigureSpec(
                "map_I_attractors.png",
                "map_I_attractors.npz",
                "dynachaos.maps.torus_doubling",
                "examples/snippets/sec04_doubling/map_I_attractors.py",
            ),
            FigureSpec(
                "map_IV_attractors.png", "map_IV_attractors.npz", "dynachaos.maps.torus_doubling"
            ),
            FigureSpec(
                "map_IV_lyapunov.png",
                "map_IV_lyapunov.npz",
                "dynachaos.maps.torus_doubling",
                "examples/snippets/sec04_doubling/map_IV_lyapunov.py",
            ),
        ),
    ),
    "sec05_oscillation": SectionSpec(
        section_id="sec05_oscillation",
        modules=("dynachaos.maps.delayed_logistic",),
        cache_files=("attractors.npz", "lyapunov_vs_D.npz", "locking_sequence.npz"),
        output_files=(
            "attractors.npz",
            "lyapunov_vs_D.npz",
            "locking_sequence.npz",
            "attractors.png",
            "lyapunov_vs_D.png",
            "locking_sequence.png",
        ),
        npz_contracts=(
            NpzContract("attractors.npz", ("D_values", "A")),
            NpzContract("lyapunov_vs_D.npz", ("D", "spectra")),
            NpzContract("locking_sequence.npz", ("D_values", "A")),
        ),
        figures=(
            FigureSpec(
                "attractors.png",
                "attractors.npz",
                "dynachaos.maps.delayed_logistic",
                "examples/snippets/sec05_oscillation/attractors.py",
            ),
            FigureSpec(
                "lyapunov_vs_D.png",
                "lyapunov_vs_D.npz",
                "dynachaos.maps.delayed_logistic",
                "examples/snippets/sec05_oscillation/lyapunov_vs_D.py",
            ),
            FigureSpec(
                "locking_sequence.png",
                "locking_sequence.npz",
                "dynachaos.maps.delayed_logistic",
                "examples/snippets/sec05_oscillation/locking_sequence.py",
            ),
        ),
    ),
    "sec06_three_torus": SectionSpec(
        section_id="sec06_three_torus",
        modules=("dynachaos.maps.coupled_delayed", "dynachaos.maps.modulated_circle"),
        cache_files=("lyapunov_vs_DB.npz", "xz_projections.npz", "double_staircase.npz"),
        output_files=(
            "lyapunov_vs_DB.npz",
            "xz_projections.npz",
            "double_staircase.npz",
            "lyapunov_vs_DB.png",
            "xz_projections.png",
            "double_staircase.png",
            "double_staircase_zoom.png",
        ),
        npz_contracts=(
            NpzContract(
                "lyapunov_vs_DB.npz",
                (
                    "DB",
                    "eps_values",
                    "eps_0.001_spectra",
                    "eps_0.005_spectra",
                    "eps_0.01_spectra",
                ),
            ),
            NpzContract(
                "xz_projections.npz",
                ("DB_values", "labels", "render_modes", "schema_version"),
            ),
            NpzContract("double_staircase.npz", ("D", "rho_theta", "rho_phi", "A", "C", "eps")),
        ),
        figures=(
            FigureSpec(
                "lyapunov_vs_DB.png",
                "lyapunov_vs_DB.npz",
                "dynachaos.maps.coupled_delayed",
                "examples/snippets/sec06_three_torus/lyapunov_vs_DB.py",
            ),
            FigureSpec(
                "xz_projections.png",
                "xz_projections.npz",
                "dynachaos.maps.coupled_delayed",
                "examples/snippets/sec06_three_torus/xz_projections.py",
            ),
            FigureSpec(
                "double_staircase.png",
                "double_staircase.npz",
                "dynachaos.maps.modulated_circle",
                "examples/snippets/sec06_three_torus/double_staircase.py",
            ),
            FigureSpec(
                "double_staircase_zoom.png",
                None,
                "dynachaos.maps.modulated_circle",
                "examples/snippets/sec06_three_torus/double_staircase_zoom.py",
            ),
        ),
    ),
    "sec07_fractalization": SectionSpec(
        section_id="sec07_fractalization",
        modules=("dynachaos.maps.fractalization",),
        cache_files=("fractal_attractors.npz", "correlation_dimension.npz"),
        output_files=(
            "fractal_attractors.npz",
            "correlation_dimension.npz",
            "fractal_attractors.png",
            "correlation_dimension.png",
        ),
        npz_contracts=(
            NpzContract("fractal_attractors.npz", ("D_values", "A")),
            NpzContract("correlation_dimension.npz", ("D", "D2", "A")),
        ),
        figures=(
            FigureSpec(
                "fractal_attractors.png",
                "fractal_attractors.npz",
                "dynachaos.maps.fractalization",
                "examples/snippets/sec07_fractalization/fractal_attractors.py",
            ),
            FigureSpec(
                "correlation_dimension.png",
                "correlation_dimension.npz",
                "dynachaos.maps.fractalization",
                "examples/snippets/sec07_fractalization/correlation_dimension.py",
            ),
        ),
    ),
    "sec08_sti": SectionSpec(
        section_id="sec08_sti",
        modules=(
            "dynachaos.cml.spatiotemporal",
            "dynachaos.cml.comoving_figure",
            "dynachaos.cml.correlation_figure",
        ),
        cache_files=(
            "spacetime_diagrams.npz",
            "comoving_lyapunov.npz",
            "correlation_decay.npz",
        ),
        output_files=(
            "spacetime_diagrams.npz",
            "spacetime_diagrams.png",
            "comoving_lyapunov.npz",
            "comoving_lyapunov.png",
            "correlation_decay.npz",
            "correlation_decay.png",
        ),
        npz_contracts=(
            NpzContract("spacetime_diagrams.npz", ("A_eps_0.06", "B_eps_0.02", "C_eps_0.16")),
            NpzContract("comoving_lyapunov.npz", ("v_values", "a_values", "eps", "N")),
            NpzContract("correlation_decay.npz", ("a_corr", "r_vals", "all_corr", "xi_values")),
        ),
        figures=(
            FigureSpec(
                "spacetime_diagrams.png",
                "spacetime_diagrams.npz",
                "dynachaos.cml.spatiotemporal",
                "examples/snippets/sec08_sti/spacetime_diagrams.py",
            ),
            FigureSpec(
                "comoving_lyapunov.png",
                "comoving_lyapunov.npz",
                "dynachaos.cml.comoving_figure",
                "examples/snippets/sec08_sti/comoving_lyapunov.py",
            ),
            FigureSpec(
                "correlation_decay.png",
                "correlation_decay.npz",
                "dynachaos.cml.correlation_figure",
                "examples/snippets/sec08_sti/correlation_decay.py",
            ),
        ),
    ),
    "sec09_pattern": SectionSpec(
        section_id="sec09_pattern",
        modules=("dynachaos.cml.pattern_dynamics",),
        cache_files=("phase_diagram.npz", "space_amplitude.npz"),
        output_files=(
            "phase_diagram.npz",
            "space_amplitude.npz",
            "phase_diagram.png",
            "space_amplitude.png",
        ),
        npz_contracts=(
            NpzContract("phase_diagram.npz", ("a", "eps", "lam", "spatial_activity")),
            NpzContract("space_amplitude.npz", ("params", "schema_version")),
        ),
        figures=(
            FigureSpec(
                "phase_diagram.png",
                "phase_diagram.npz",
                "dynachaos.cml.pattern_dynamics",
                "examples/snippets/sec09_pattern/phase_diagram.py",
            ),
            FigureSpec(
                "space_amplitude.png",
                "space_amplitude.npz",
                "dynachaos.cml.pattern_dynamics",
                "examples/snippets/sec09_pattern/space_amplitude.py",
            ),
        ),
    ),
    "sec10_gcm": SectionSpec(
        section_id="sec10_gcm",
        modules=(
            "dynachaos.cml.globally_coupled",
            "dynachaos.cml.gcm_clusters",
        ),
        cache_files=(
            "gcm_results.npz",
            "gcm_clusters.npz",
            "collective_lyapunov.npz",
        ),
        output_files=(
            "gcm_results.npz",
            "gcm_msd.png",
            "gcm_distribution.png",
            "gcm_clusters.npz",
            "gcm_clusters.png",
            "collective_lyapunov.npz",
            "collective_lyapunov.png",
        ),
        npz_contracts=(
            NpzContract("gcm_results.npz", ("N_values", "a", "eps", "N_grid", "msd_grid")),
            NpzContract("gcm_clusters.npz", ("cluster_labels", "x_record", "a", "eps", "N")),
            NpzContract("collective_lyapunov.npz", ("a_values", "lyap_c", "eps", "N")),
        ),
        figures=(
            FigureSpec(
                "gcm_msd.png",
                "gcm_results.npz",
                "dynachaos.cml.globally_coupled",
                "examples/snippets/sec10_gcm/gcm_msd.py",
            ),
            FigureSpec(
                "gcm_distribution.png",
                "gcm_results.npz",
                "dynachaos.cml.globally_coupled",
                "examples/snippets/sec10_gcm/gcm_distribution.py",
            ),
            FigureSpec(
                "gcm_clusters.png",
                "gcm_clusters.npz",
                "dynachaos.cml.gcm_clusters",
                "examples/snippets/sec10_gcm/gcm_clusters.py",
            ),
            FigureSpec(
                "collective_lyapunov.png",
                "collective_lyapunov.npz",
                "dynachaos.cml.gcm_clusters",
                "examples/snippets/sec10_gcm/collective_lyapunov.py",
            ),
        ),
    ),
    "sec11_diagnostics": SectionSpec(
        section_id="sec11_diagnostics",
        modules=("dynachaos.diagnostics.compare_all",),
        cache_files=(
            "test01_sweep.npz",
            "sali_comparison.npz",
            "permutation_entropy.npz",
            "complexity_entropy_plane.npz",
            "rqa_measures.npz",
        ),
        output_files=(
            "test01_sweep.npz",
            "sali_comparison.npz",
            "permutation_entropy.npz",
            "complexity_entropy_plane.npz",
            "rqa_measures.npz",
            "test01_sweep.png",
            "sali_comparison.png",
            "permutation_entropy.png",
            "complexity_entropy_plane.png",
            "rqa_measures.png",
        ),
        npz_contracts=(
            NpzContract("test01_sweep.npz", ("a", "K")),
            NpzContract("sali_comparison.npz", ("DB_values", "lambda1_values")),
            NpzContract("permutation_entropy.npz", ("a", "H_logistic", "D", "H_delayed")),
            NpzContract(
                "complexity_entropy_plane.npz",
                ("a", "H_logistic", "C_logistic", "D", "H_delayed", "C_delayed"),
            ),
            NpzContract("rqa_measures.npz", ("D", "RR", "DET", "LAM", "ENTR")),
        ),
        figures=(
            FigureSpec(
                "test01_sweep.png",
                "test01_sweep.npz",
                "dynachaos.diagnostics.compare_all",
                "examples/snippets/sec11_diagnostics/test01_sweep.py",
            ),
            FigureSpec(
                "sali_comparison.png",
                "sali_comparison.npz",
                "dynachaos.diagnostics.compare_all",
                "examples/snippets/sec11_diagnostics/sali_comparison.py",
            ),
            FigureSpec(
                "permutation_entropy.png",
                "permutation_entropy.npz",
                "dynachaos.diagnostics.compare_all",
                "examples/snippets/sec11_diagnostics/permutation_entropy.py",
            ),
            FigureSpec(
                "complexity_entropy_plane.png",
                "complexity_entropy_plane.npz",
                "dynachaos.diagnostics.compare_all",
                "examples/snippets/sec11_diagnostics/complexity_entropy_plane.py",
            ),
            FigureSpec(
                "rqa_measures.png",
                "rqa_measures.npz",
                "dynachaos.diagnostics.compare_all",
                "examples/snippets/sec11_diagnostics/rqa_measures.py",
            ),
        ),
    ),
    "sec12_intermittency": SectionSpec(
        section_id="sec12_intermittency",
        modules=(
            "dynachaos.diagnostics.intermittency_figure",
            "dynachaos.diagnostics.on_off_intermittency_figure",
            "dynachaos.diagnostics.type_ii_intermittency_figure",
            "dynachaos.diagnostics.type_iii_intermittency_figure",
            "dynachaos.cml.sti_spine_figure",
        ),
        cache_files=(
            "type_i_intermittency.npz",
            "on_off_intermittency.npz",
            "type_ii_intermittency.npz",
            "type_iii_intermittency.npz",
            "sti_spine.npz",
        ),
        output_files=(
            "type_i_intermittency.npz",
            "type_i_intermittency.png",
            "on_off_intermittency.npz",
            "on_off_intermittency.png",
            "type_ii_intermittency.npz",
            "type_ii_intermittency.png",
            "type_iii_intermittency.npz",
            "type_iii_intermittency.png",
            "sti_spine.npz",
            "sti_spine.png",
        ),
        npz_contracts=(
            NpzContract(
                "type_i_intermittency.npz",
                (
                    "schema_version",
                    "source_file",
                    "seed",
                    "logistic_mechanism_r",
                    "logistic_tail_r",
                    "logistic_laminar_lengths",
                    "type_i_tail_alpha",
                    "type_i_vuong_z",
                    "normal_form_beta",
                    "normal_form_eps",
                    "normal_form_mean_lengths",
                    "logistic_f3_return_points",
                    "logistic_f3_channel_slope",
                    "lorenz_return_points",
                    "lorenz_channel_slope",
                    "type_i_tail_alpha_ci",
                    "type_i_tail_gof_p",
                ),
            ),
            NpzContract(
                "on_off_intermittency.npz",
                (
                    "schema_version",
                    "source_file",
                    "seed",
                    "benchmark_eps",
                    "benchmark_lambda_perp",
                    "benchmark_series",
                    "benchmark_laminar_mask",
                    "benchmark_laminar_lengths",
                    "benchmark_burst_lengths",
                    "benchmark_burst_amplitudes",
                    "benchmark_threshold_percentile",
                    "off_time_alpha",
                    "off_time_alpha_ci",
                    "off_time_gof_p",
                    "burst_amplitude_alpha",
                    "burst_amplitude_alpha_ci",
                    "burst_amplitude_gof_p",
                    "scaling_eps_values",
                    "lambda_abs_values",
                    "mean_off_lengths",
                    "mean_off_beta",
                    "skew_driver_series",
                    "skew_transverse_series",
                ),
            ),
            NpzContract(
                "type_iii_intermittency.npz",
                (
                    "schema_version",
                    "source_file",
                    "seed",
                    "eps",
                    "a",
                    "escape_threshold",
                    "return_grid",
                    "f2_return_points",
                    "f2_linear_slope",
                    "f2_cubic_coefficient",
                    "reinjection_points",
                    "series",
                    "laminar_mask",
                    "laminar_lengths",
                    "laminar_tail_alpha",
                    "laminar_tail_alpha_ci",
                    "laminar_tail_gof_p",
                    "rpd_thresholds",
                    "rpd_conditional_means",
                    "rpd_slope",
                    "rpd_intercept",
                    "rpd_alpha",
                    "rpd_rvalue",
                ),
            ),
            NpzContract(
                "type_ii_intermittency.npz",
                (
                    "schema_version",
                    "source_file",
                    "seed",
                    "eps",
                    "a",
                    "theta",
                    "escape_threshold",
                    "spiral_orbit",
                    "spiral_radius",
                    "spiral_escape_index",
                    "reinjection_radii",
                    "laminar_lengths",
                    "laminar_histogram_edges",
                    "laminar_histogram_density",
                    "laminar_tail_alpha",
                    "laminar_tail_alpha_ci",
                    "laminar_tail_gof_p",
                    "exponential_rate",
                    "exponential_intercept",
                    "exponential_rvalue",
                ),
            ),
            NpzContract(
                "sti_spine.npz",
                (
                    "schema_version",
                    "source_file",
                    "seed",
                    "model_a_parameter",
                    "display_eps",
                    "sweep_eps",
                    "spacetime",
                    "turbulent_mask",
                    "turbulent_fraction",
                    "laminar_cluster_sizes",
                    "cluster_size_values",
                    "cluster_size_probabilities",
                    "cluster_decay_rate",
                ),
            ),
        ),
        figures=(
            FigureSpec(
                "type_i_intermittency.png",
                "type_i_intermittency.npz",
                "dynachaos.diagnostics.intermittency_figure",
                "examples/snippets/sec12_intermittency/type_i.py",
            ),
            FigureSpec(
                "on_off_intermittency.png",
                "on_off_intermittency.npz",
                "dynachaos.diagnostics.on_off_intermittency_figure",
                "examples/snippets/sec12_intermittency/on_off.py",
            ),
            FigureSpec(
                "type_ii_intermittency.png",
                "type_ii_intermittency.npz",
                "dynachaos.diagnostics.type_ii_intermittency_figure",
                "examples/snippets/sec12_intermittency/type_ii.py",
            ),
            FigureSpec(
                "type_iii_intermittency.png",
                "type_iii_intermittency.npz",
                "dynachaos.diagnostics.type_iii_intermittency_figure",
                "examples/snippets/sec12_intermittency/type_iii.py",
            ),
            FigureSpec(
                "sti_spine.png",
                "sti_spine.npz",
                "dynachaos.cml.sti_spine_figure",
                "examples/snippets/sec12_intermittency/sti_spine.py",
            ),
        ),
    ),
}


def list_sections() -> tuple[str, ...]:
    """Return section IDs in paper order."""
    return SECTION_ORDER


def get_section(section_id: str) -> SectionSpec:
    """Return a section spec or raise a KeyError."""
    return SECTION_SPECS[section_id]


def get_figure(section_id: str, png: str) -> FigureSpec | None:
    """Return the FigureSpec for one page figure, or None if unregistered."""
    for figure in SECTION_SPECS[section_id].figures:
        if figure.png == png:
            return figure
    return None
