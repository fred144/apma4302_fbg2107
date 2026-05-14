"""
Compare the FVM solver output with the exact Riemann solution.

Usage (single snapshot):
    python plot.py --numeric results_t0.2000.csv --analytic analytic_sod.csv

Convergence study (L1 error vs N):
    python plot.py --convergence --problem sod --t 0.2 --grids 100 200 400 800
    (expects results_t0.2000.csv from each run, renamed as results_N<N>.csv)
"""

import argparse
import glob
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analytic import ExactRiemann, PROBLEMS

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.size": 13,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)


# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #
def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]  # x, rho, vx, P


def l1_error(x_num, q_num, x_ref, q_ref):
    """L1 error: interpolate ref onto numeric grid."""
    q_ref_interp = np.interp(x_num, x_ref, q_ref)
    dx = x_num[1] - x_num[0]
    return np.sum(np.abs(q_num - q_ref_interp)) * dx


# ------------------------------------------------------------------ #
# Comparison plot                                                      #
# ------------------------------------------------------------------ #
def plot_comparison(numeric_csv, analytic_csv, outfile="comparison.png"):
    x_n, rho_n, vx_n, P_n = load_csv(numeric_csv)
    x_a, rho_a, vx_a, P_a = load_csv(analytic_csv)

    fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True, dpi=300)
    fig.suptitle(f"FVM vs Exact: {os.path.basename(numeric_csv)}")

    for ax, (yn, ya, lbl) in zip(
        axes, [(rho_n, rho_a, r"$\rho$"), (vx_n, vx_a, r"$v_x$"), (P_n, P_a, r"$P$")]
    ):
        ax.plot(x_a, ya, "k-", lw=1.5, label="Exact")
        ax.plot(x_n, yn, "r--", lw=1.0, label="FVM", alpha=0.85)
        ax.set_ylabel(lbl)
        ax.legend()

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


# ------------------------------------------------------------------ #
# Limiter comparison (multiple numeric CSVs on same axes)             #
# ------------------------------------------------------------------ #
def plot_limiter_comparison(
    csvs, labels, analytic_csv, outfile="limiter_comparison.png"
):
    x_a, rho_a, vx_a, P_a = load_csv(analytic_csv)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    ls = ["--", "-.", ":", (0, (3, 1, 1, 1))]

    fig, axes = plt.subplots(3, 1, figsize=(5, 7), sharex=True, dpi=300)
    # fig.suptitle("Limiter comparison")
    ylabels = [r"$\rho$", r"$v_x$", r"$P$"]

    for ax, (ya, lbl) in zip(axes, zip([rho_a, vx_a, P_a], ylabels)):
        ax.plot(x_a, ya, "k-", lw=1.8, label="Exact", zorder=0)
        ax.set_ylabel(lbl)

    for csv, lab, c, l in zip(csvs, labels, colors, ls):
        x_n, rho_n, vx_n, P_n = load_csv(csv)
        for ax, yn in zip(axes, [rho_n, vx_n, P_n]):
            ax.plot(x_n, yn, color=c, ls=l, lw=1.0, label=lab, alpha=0.9)

 
    axes[0].legend(ncol=2, bbox_to_anchor=(0.5, 1.01), loc="lower center")
    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


# ------------------------------------------------------------------ #
# Convergence study                                                    #
# ------------------------------------------------------------------ #
def plot_convergence(grids, problem, t, outfile="convergence.png"):
    cfg = PROBLEMS[problem]
    rs = ExactRiemann(**cfg)

    Ns, errs = [], []
    for N in grids:
        fname = f"results_N{N}.csv"
        if not os.path.exists(fname):
            print(f"  WARNING: {fname} not found, skipping")
            continue
        x_n, rho_n, _, _ = load_csv(fname)
        x_ref = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
        rho_ref, _, _ = rs.solve(x_ref, t)
        err = l1_error(x_n, rho_n, x_ref, rho_ref)
        Ns.append(N)
        errs.append(err)
        print(f"  N={N:4d}  L1(rho) = {err:.4e}")

    if len(Ns) < 2:
        print("Need at least 2 grids for convergence plot.")
        return

    Ns = np.array(Ns, dtype=float)
    errs = np.array(errs, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    ax.loglog(Ns, errs, "o-", label=r"$\|\rho_{FVM} - \rho_{exact}\|_{L1}$")

    # reference slopes
    for order, ls in [(1, "--"), (2, ":")]:
        ref = errs[0] * (Ns[0] / Ns) ** order
        ax.loglog(Ns, ref, ls, color="gray", label=f"$O(N^{{-{order}}})$")

    ax.set_xlabel("N cells")
    ax.set_ylabel(r"$L^1$ error in $\rho$")
    ax.set_title(f"Convergence — {problem} at t={t}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", default=None, help="FVM CSV to plot against analytic")
    ap.add_argument(
        "--analytic", default=None, help="Exact solution CSV (or generated on the fly)"
    )
    ap.add_argument("--problem", default="sod", choices=PROBLEMS.keys())
    ap.add_argument("--t", type=float, default=0.2)
    ap.add_argument("--convergence", action="store_true")
    ap.add_argument("--grids", type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.convergence:
        plot_convergence(
            args.grids, args.problem, args.t, outfile=args.out or "convergence.png"
        )
        return

    if args.numeric is None:
        ap.error("Provide --numeric <csv> for comparison plot")

    # generate analytic CSV if not provided
    analytic_csv = args.analytic
    if analytic_csv is None:
        import subprocess, sys

        analytic_csv = f"analytic_{args.problem}.csv"
        N = len(np.loadtxt(args.numeric, delimiter=",", skiprows=1))
        subprocess.run(
            [
                sys.executable,
                "analytic.py",
                "--problem",
                args.problem,
                "--t",
                str(args.t),
                "--N",
                str(N),
                "--out",
                analytic_csv,
            ],
            check=True,
        )

    plot_comparison(args.numeric, analytic_csv, outfile=args.out or "comparison.png")


if __name__ == "__main__":
    main()
