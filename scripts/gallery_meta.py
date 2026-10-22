"""Gallery metadata: section titles and figure captions."""

from __future__ import annotations

SECTION_TITLES: dict[str, str] = {
    "sec02_circle_map": "Circle Map: Devil's Staircase and Arnold Tongues",
    "sec03_transition": "Transition to Chaos in Coupled Maps",
    "sec04_doubling": "Torus Doubling",
    "sec05_oscillation": "Quasiperiodic Oscillation and Phase Locking",
    "sec06_three_torus": "Three-Torus and the Double Staircase",
    "sec07_fractalization": "Torus Fractalization",
    "sec08_sti": "Spatiotemporal Intermittency",
    "sec09_pattern": "Pattern Dynamics in Coupled Map Lattices",
    "sec10_gcm": "Globally Coupled Maps",
    "sec11_diagnostics": "Chaos Diagnostics",
    "sec12_intermittency": "Routes to Intermittency",
}

CAPTIONS: dict[str, str] = {
    "sec02_circle_map/devils_staircase.png": (
        "Devil's staircase of the circle map: rotation number rises in flat mode-locking steps."
    ),
    "sec02_circle_map/arnold_tongues.png": (
        "Rotation number over the circle-map parameter plane; the uniform wedges are "
        "frequency-locked Arnold tongues."
    ),
    "sec02_circle_map/staircase_zoom.png": (
        "Zoomed view of the devil's staircase, showing self-similar mode-locking steps at "
        "finer scale."
    ),
    "sec03_transition/phase_diagram.png": (
        "Parameter-plane survey of the coupled logistic map showing symmetry breaking and "
        "regions of chaos."
    ),
    "sec03_transition/attractors.png": (
        "Attractor portraits of the coupled logistic map along its symmetry-breaking route."
    ),
    "sec03_transition/basins.png": (
        "Basin of attraction of the coupled logistic map, showing a striped boundary between "
        "two mirror-image cycles."
    ),
    "sec04_doubling/map_I_attractors.png": (
        "Torus doubling in Map (I): a simple torus, a doubled torus, and the collapse to chaos."
    ),
    "sec04_doubling/map_IV_attractors.png": (
        "Torus-doubling cascade of Map (IV): a fourfold torus, an eightfold torus, and chaos."
    ),
    "sec04_doubling/map_IV_lyapunov.png": (
        "Lyapunov spectrum of Map (IV), showing near-zero exponents through the doubling "
        "cascade until chaos sets in."
    ),
    "sec05_oscillation/attractors.png": (
        "Attractor portraits of the delayed logistic map, tracing a torus growing, locking, "
        "and breaking down into chaos."
    ),
    "sec05_oscillation/lyapunov_vs_D.png": (
        "Lyapunov exponents of the delayed logistic map, marking the shift from quasiperiodic "
        "motion to chaos."
    ),
    "sec05_oscillation/locking_sequence.png": (
        "Close-up sequence of attractors resolving the delayed logistic map's transition "
        "from locking to chaos."
    ),
    "sec06_three_torus/lyapunov_vs_DB.png": (
        "Lyapunov spectrum of the coupled delayed logistic map, distinguishing quasiperiodic "
        "motion, locking, and chaos."
    ),
    "sec06_three_torus/xz_projections.png": (
        "Attractor projections of the coupled delayed logistic map moving through a "
        "resonance web, locking, and chaos."
    ),
    "sec06_three_torus/double_staircase.png": (
        "Double devil's staircase of the modulated circle map, showing locking plateaus in "
        "both rotation numbers."
    ),
    "sec06_three_torus/double_staircase_zoom.png": (
        "Zoomed view of the double devil's staircase, showing locking plateaus in finer detail."
    ),
    "sec07_fractalization/fractal_attractors.png": (
        "A smooth torus in the delayed logistic map develops wrinkles at finer and finer "
        "scales as it fractalizes."
    ),
    "sec07_fractalization/correlation_dimension.png": (
        "Correlation dimension of the attractor rising from a smooth torus toward a "
        "fractalized one."
    ),
    "sec08_sti/spacetime_diagrams.png": (
        "Spacetime diagrams of spatiotemporal intermittency in three coupled map lattice models."
    ),
    "sec08_sti/comoving_lyapunov.png": (
        "Co-moving Lyapunov exponent of the logistic coupled map lattice, whose zero crossings "
        "mark propagation speeds."
    ),
    "sec08_sti/correlation_decay.png": (
        "Spatial correlation decay and finite-size convergence in the logistic coupled map lattice."
    ),
    "sec09_pattern/phase_diagram.png": (
        "Activity map of the logistic coupled map lattice across its nonlinearity and "
        "coupling parameters."
    ),
    "sec09_pattern/space_amplitude.png": (
        "Space-amplitude snapshots of the logistic coupled map lattice in five representative "
        "pattern regimes."
    ),
    "sec10_gcm/gcm_msd.png": (
        "Mean-square deviation of the mean field in a globally coupled map, failing to shrink "
        "with system size."
    ),
    "sec10_gcm/gcm_distribution.png": (
        "Distribution of the mean field in a globally coupled map, with variance that does not "
        "narrow as system size grows."
    ),
    "sec10_gcm/gcm_clusters.png": (
        "Cluster states in a globally coupled map, showing a partially ordered regime."
    ),
    "sec10_gcm/collective_lyapunov.png": (
        "Collective Lyapunov exponent of the mean field in a globally coupled map, marking "
        "collective chaos."
    ),
    "sec11_diagnostics/test01_sweep.png": (
        "The 0-1 test statistic for the logistic map, near 0 in periodic windows and near 1 "
        "in chaotic bands."
    ),
    "sec11_diagnostics/sali_comparison.png": (
        "SALI time series for the coupled delayed logistic map across regimes from "
        "quasiperiodic motion to chaos."
    ),
    "sec11_diagnostics/permutation_entropy.png": (
        "Permutation entropy for the logistic and delayed logistic maps, low in regular "
        "windows and high in irregular ones."
    ),
    "sec11_diagnostics/complexity_entropy_plane.png": (
        "Complexity-entropy plane locations for the logistic and delayed logistic maps."
    ),
    "sec11_diagnostics/rqa_measures.png": (
        "Recurrence quantification measures tracking the delayed logistic map's torus-to-chaos "
        "transition."
    ),
    "sec12_intermittency/type_i_intermittency.png": (
        "Type-I intermittency: a tangent-bifurcation channel, laminar-length statistics, "
        "and a Lorenz reinjection channel."
    ),
    "sec12_intermittency/on_off_intermittency.png": (
        "On-off intermittency near a blowout onset, showing laminar epochs, bursts, and their "
        "statistics."
    ),
    "sec12_intermittency/type_ii_intermittency.png": (
        "Type-II intermittency, illustrated with a normal-form spiral orbit and its "
        "laminar-length statistics."
    ),
    "sec12_intermittency/type_iii_intermittency.png": (
        "Type-III intermittency: a period-doubling return map, escape episodes, and "
        "reinjection statistics."
    ),
    "sec12_intermittency/sti_spine.png": (
        "Spatiotemporal intermittency in a coupled map lattice: turbulent-fraction onset and "
        "an exponential laminar cluster-size tail."
    ),
}

HERO: tuple[str, ...] = (
    "sec02_circle_map/arnold_tongues.png",
    "sec04_doubling/map_IV_attractors.png",
    "sec06_three_torus/double_staircase.png",
    "sec07_fractalization/fractal_attractors.png",
    "sec08_sti/spacetime_diagrams.png",
    "sec10_gcm/gcm_clusters.png",
    "sec11_diagnostics/complexity_entropy_plane.png",
    "sec12_intermittency/type_i_intermittency.png",
)
