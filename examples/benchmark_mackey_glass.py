#!/usr/bin/env python3
"""
benchmark_mackey_glass: Validate diagnostics against the Mackey-Glass DDE.

At tau=17:
- D_KY ~ 2.10 (Kaplan-Yorke dimension; Farmer 1982)
- D_F ~ 2.13 (fractal dimension; Farmer 1982)
- lambda_1 ~ 0.006 (literature; not computed here -- DDE requires
  specialized methods beyond our scope)

Note: REF_D2 here is the Kaplan-Yorke dimension D_KY, not a GP-measured D2.
The GP correlation dimension D2 is expected to be close but not identical.

References
----------
Farmer, J.D. (1982). "Chaotic attractors of an infinite-dimensional dynamical
  system." Physica D, 4(3), 366-393. DOI: 10.1016/0167-2789(82)90042-2

OUTPUTS: benchmark_mackey_glass.npz, benchmark_mackey_glass.png
USAGE:   python examples/benchmark_mackey_glass.py
         rm examples/benchmark_mackey_glass.npz && python examples/benchmark_mackey_glass.py
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
REF_D2 = CONFIG["ref_D2"]
REF_LAMBDA1 = CONFIG["ref_lambda1"]


def compute():
    from dynachaos.maps.flows import mackey_glass_series

    # Generate MG series
    x = mackey_glass_series(
        n_points=CONFIG["n_points"], tau=CONFIG["tau_mg"],
        t_transient=CONFIG["t_transient"],
    )

    # Embedding analysis
    from _pipeline import run_embedding_analysis
    results = run_embedding_analysis(
        x, tau_max=CONFIG["tau_max"], d_max=CONFIG["d_max"],
        n_r=CONFIG["n_r"], verbose=True,
    )

    np.savez_compressed(
        OUTPUT_NPZ,
        x=x, tau_mg=17,
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
    print(f"  D2 = {results['D2']:.3f} (ref: {REF_D2})")


def plot(data):
    from _pipeline import plot_benchmark, plot_multifractal, plot_zero_one_test

    x = data["x"]
    tau_opt = int(data["tau_opt"])
    results = {
        "tau_opt": tau_opt,
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

    # Time-delay embedding for attractor visualization
    n = len(x) - tau_opt
    attractor_xy = (x[:n], x[tau_opt:tau_opt + n])
    system_name = r"Mackey-Glass DDE ($\tau$=17)"

    plot_benchmark(
        results, attractor_xy, OUTPUT_PNG,
        system_name=system_name,
        ref_D2=REF_D2,
        ref_lambda1=REF_LAMBDA1,
        computed_lambda1=None,  # not computed for DDE
        attractor_xlabel=r"$x(t)$",
        attractor_ylabel=rf"$x(t-{tau_opt})$",
    )

    # 0-1 test (subsample by tau_opt to decorrelate the DDE output)
    plot_zero_one_test(
        x[::tau_opt], OUTPUT_PNG.with_name("benchmark_mackey_glass_01test.png"),
        system_name=system_name,
    )

    # Multifractal
    plot_multifractal(
        attractor_xy, OUTPUT_PNG.with_name("benchmark_mackey_glass_multifractal.png"),
        system_name=system_name,
    )


def main():
    try:
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
        print(f"Loaded {OUTPUT_NPZ}")
    except FileNotFoundError:
        print("Computing Mackey-Glass benchmark...")
        compute()
        data = np.load(OUTPUT_NPZ, allow_pickle = False)
    plot(data)


if __name__ == "__main__":
    main()
