#!/usr/bin/env python3
"""
benchmark_lorenz: Validate diagnostics against the Lorenz system.

At (sigma=10, rho=28, beta=8/3):
- lambda_1 = 0.9056 (Sprott)
- lambda_2 ~ 0
- lambda_3 = -14.57
- D2 = 2.05 (Grassberger & Procaccia 1983)
- D_H = 2.0627160 (Viswanath 2004, Hausdorff dimension)

References
----------
Grassberger, P. & Procaccia, I. (1983). "Measuring the strangeness of strange
  attractors." Physica D, 9(1-2), 189-208. DOI: 10.1016/0167-2789(83)90298-1
Viswanath, D. (2004). "The fractal property of the Lorenz attractor."
  Physica D, 190(1-2), 115-128. DOI: 10.1016/j.physd.2003.10.006
Sprott, J.C. (2003). "Chaos and Time-Series Analysis." Oxford University Press.

OUTPUTS: benchmark_lorenz.npz, benchmark_lorenz.png
USAGE:   python examples/benchmark_lorenz.py
         rm examples/benchmark_lorenz.npz && python examples/benchmark_lorenz.py
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))
OUTPUT_NPZ = SCRIPT.with_suffix(".npz")
OUTPUT_PNG = SCRIPT.with_suffix(".png")

from _pipeline import load_jsonc  # noqa: E402

CONFIG = load_jsonc(SCRIPT.with_suffix(".jsonc"))
REF_LAMBDA1 = CONFIG["ref_lambda1"]
REF_SPECTRUM = CONFIG["ref_spectrum"]
REF_D2 = CONFIG["ref_D2"]


def compute():
    from dynachaos.diagnostics.lyapunov import flow_lyapunov_spectrum
    from dynachaos.maps.flows import lorenz_jac, lorenz_rhs, lorenz_trajectory

    # Generate trajectory (long for good D2 estimate)
    traj = lorenz_trajectory(
        t_span=tuple(CONFIG["t_span"]), dt=CONFIG["dt"],
        t_transient=CONFIG["t_transient"],
    )

    # Lyapunov spectrum via variational equations
    spectrum = flow_lyapunov_spectrum(
        lorenz_rhs, lorenz_jac,
        x0=np.array(CONFIG["lyap_x0"]),
        t_total=CONFIG["lyap_t_total"], dt=CONFIG["lyap_dt"],
        t_transient=CONFIG["lyap_t_transient"],
        reorth_dt=CONFIG["reorth_dt"],
    )

    # Embedding analysis on x-component (subsample to reduce computation)
    from _pipeline import run_embedding_analysis
    sub = CONFIG["subsample"]
    x_series = traj[::sub, 0]
    results = run_embedding_analysis(
        x_series, tau_max=CONFIG["tau_max"], d_max=CONFIG["d_max"],
        n_r=CONFIG["n_r"], verbose=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        traj=traj, spectrum=spectrum,
        tau_opt=results["tau_opt"], d_opt=results["d_opt"],
        taus=results["taus"], mi_values=results["mi_values"],
        dims=results["dims"], E1=results["E1"], E2=results["E2"],
        fnn_dims=results["fnn_dims"],
        f1=results["f1"], f2=results["f2"], f3=results["f3"],
        r_values=results["r_values"], C_values=results["C_values"],
        D2=results["D2"],
        local_slopes=results["local_slopes"],
        scaling_mask=results["scaling_mask"],
    )
    print(f"Saved {OUTPUT_NPZ}")
    print(f"  spectrum = [{spectrum[0]:.4f}, {spectrum[1]:.4f}, {spectrum[2]:.4f}]")
    print(f"  D2       = {results['D2']:.3f} (ref: {REF_D2})")


def plot(data):
    from _pipeline import plot_benchmark, plot_multifractal, plot_zero_one_test

    traj = data["traj"]
    spectrum = data["spectrum"]
    results = {
        "tau_opt": int(data["tau_opt"]),
        "d_opt": int(data["d_opt"]),
        "taus": data["taus"],
        "mi_values": data["mi_values"],
        "dims": data["dims"],
        "E1": data["E1"],
        "E2": data["E2"],
        "fnn_dims": data["fnn_dims"],
        "f1": data["f1"],
        "f2": data["f2"],
        "f3": data["f3"],
        "r_values": data["r_values"],
        "C_values": data["C_values"],
        "D2": float(data["D2"]),
        "local_slopes": data["local_slopes"],
        "scaling_mask": data["scaling_mask"].astype(bool),
    }

    # (x, z) projection
    attractor_xy = (traj[:, 0], traj[:, 2])
    system_name = r"Lorenz system ($\sigma$=10, $\rho$=28, $\beta$=8/3)"

    plot_benchmark(
        results, attractor_xy, OUTPUT_PNG,
        system_name=system_name,
        ref_D2=REF_D2,
        ref_lambda1=REF_LAMBDA1,
        computed_lambda1=float(spectrum[0]),
        computed_spectrum=spectrum,
        ref_spectrum=REF_SPECTRUM,
        attractor_xlabel="$x$",
        attractor_ylabel="$z$",
    )

    # 0-1 test (on subsampled x-component — raw dt=0.01 is too correlated)
    sub = CONFIG["subsample"]
    plot_zero_one_test(
        traj[::sub, 0], OUTPUT_PNG.with_name("benchmark_lorenz_01test.png"),
        system_name=system_name,
    )

    # Multifractal
    plot_multifractal(
        attractor_xy, OUTPUT_PNG.with_name("benchmark_lorenz_multifractal.png"),
        system_name=system_name,
    )


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing Lorenz benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
