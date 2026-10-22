#!/usr/bin/env python3
"""
benchmark_rossler: Validate diagnostics against the Rossler system.

At (a=0.2, b=0.2, c=5.7):
- lambda_1 = 0.0714 (Sprott)
- lambda_2 ~ 0
- lambda_3 = -5.39
- D2 = 2.01 (Sprott & Rowlands 2001)

References
----------
Sprott, J.C. & Rowlands, G. (2001). "Improved correlation dimension calculation."
  Int. J. Bifurcation Chaos, 11(7), 1865-1880. DOI: 10.1142/S021812740100305X
Sprott, J.C. (2003). "Chaos and Time-Series Analysis." Oxford University Press.

OUTPUTS: benchmark_rossler.npz, benchmark_rossler.png
USAGE:   python examples/benchmark_rossler.py
         rm examples/benchmark_rossler.npz && python examples/benchmark_rossler.py
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
    from dynachaos.maps.flows import rossler_jac, rossler_rhs, rossler_trajectory

    # Rossler has slower dynamics -> longer integration
    traj = rossler_trajectory(
        t_span=tuple(CONFIG["t_span"]), dt=CONFIG["dt"],
        t_transient=CONFIG["t_transient"],
    )

    # Lyapunov spectrum
    spectrum = flow_lyapunov_spectrum(
        rossler_rhs, rossler_jac,
        x0=np.array(CONFIG["lyap_x0"]),
        t_total=CONFIG["lyap_t_total"], dt=CONFIG["lyap_dt"],
        t_transient=CONFIG["lyap_t_transient"],
        reorth_dt=CONFIG["reorth_dt"],
    )

    # Embedding analysis on x-component (subsample)
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

    # (x, y) projection
    attractor_xy = (traj[:, 0], traj[:, 1])
    system_name = "Rossler system (a=0.2, b=0.2, c=5.7)"

    plot_benchmark(
        results, attractor_xy, OUTPUT_PNG,
        system_name=system_name,
        ref_D2=REF_D2,
        ref_lambda1=REF_LAMBDA1,
        computed_lambda1=float(spectrum[0]),
        computed_spectrum=spectrum,
        ref_spectrum=REF_SPECTRUM,
    )

    # 0-1 test: subsample by tau_opt on the embedding-rate series to
    # decorrelate.  Rossler's slow orbit needs aggressive subsampling.
    sub = CONFIG["subsample"]
    tau_01 = results["tau_opt"]
    plot_zero_one_test(
        traj[::sub * tau_01, 0], OUTPUT_PNG.with_name("benchmark_rossler_01test.png"),
        system_name=system_name,
    )

    # Multifractal
    plot_multifractal(
        attractor_xy, OUTPUT_PNG.with_name("benchmark_rossler_multifractal.png"),
        system_name=system_name,
    )


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing Rossler benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
