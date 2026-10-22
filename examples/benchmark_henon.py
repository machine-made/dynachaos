#!/usr/bin/env python3
"""
benchmark_henon: Validate diagnostics against the Henon map.

At (a=1.4, b=0.3):
- lambda_1 = 0.4192 (Sprott)
- lambda_2 = -1.623
- D2 = 1.21 ± 0.01 (Grassberger & Procaccia 1983, Table I)

References
----------
Grassberger, P. & Procaccia, I. (1983). "Measuring the strangeness of strange
  attractors." Physica D, 9(1-2), 189-208. DOI: 10.1016/0167-2789(83)90298-1
Sprott, J.C. (2003). "Chaos and Time-Series Analysis." Oxford University Press.

OUTPUTS: benchmark_henon.npz, benchmark_henon.png
USAGE:   python examples/benchmark_henon.py
         rm examples/benchmark_henon.npz && python examples/benchmark_henon.py
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
    from dynachaos.diagnostics.lyapunov import lyapunov_spectrum
    from dynachaos.maps.base import HenonMap
    from dynachaos.maps.henon import henon, henon_jac

    a, b = CONFIG["a"], CONFIG["b"]
    hm = HenonMap(a=a, b=b)

    N = CONFIG["N"]
    n_transient = CONFIG["n_transient"]
    x0 = np.array(CONFIG["x0"])
    traj = hm.trajectory(x0, n_iter=N, n_transient=n_transient)

    # Lyapunov spectrum
    def f(state):
        return henon(state, a, b)

    def jac(state):
        return henon_jac(state, a, b)

    spectrum = lyapunov_spectrum(f, jac, x0,
                                n_iter=CONFIG["lyap_n_iter"],
                                n_transient=CONFIG["lyap_n_transient"])

    # Embedding analysis on x-component
    from _pipeline import run_embedding_analysis
    x_series = traj[:, 0]
    results = run_embedding_analysis(
        x_series, tau_max=CONFIG["tau_max"], d_max=CONFIG["d_max"],
        n_r=CONFIG["n_r"], verbose=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        traj=traj, a=a, b=b, spectrum=spectrum,
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
    print(f"  spectrum = [{spectrum[0]:.4f}, {spectrum[1]:.4f}]")
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

    attractor_xy = (traj[:, 0], traj[:, 1])
    system_name = f"Henon map, a={float(data['a'])}, b={float(data['b'])}"

    plot_benchmark(
        results, attractor_xy, OUTPUT_PNG,
        system_name=system_name,
        ref_D2=REF_D2,
        ref_lambda1=REF_LAMBDA1,
        computed_lambda1=float(spectrum[0]),
        computed_spectrum=spectrum,
        ref_spectrum=REF_SPECTRUM,
    )

    # 0-1 test (on x-component)
    plot_zero_one_test(
        traj[:, 0], OUTPUT_PNG.with_name("benchmark_henon_01test.png"),
        system_name=system_name,
    )

    # Multifractal
    plot_multifractal(
        attractor_xy, OUTPUT_PNG.with_name("benchmark_henon_multifractal.png"),
        system_name=system_name,
    )


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing Henon benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
