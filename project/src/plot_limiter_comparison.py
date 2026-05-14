"""
Compare multiple slope limiters on the same problem and snapshot.

Usage:
    python plot_limiter_comparison.py \
        --csvs results_lim_none.csv results_lim_minmod.csv \
               results_lim_vanleer.csv results_lim_superbee.csv \
        --analytic analytic_sod.csv --problem sod --t 0.2 --N 400 \
        --out limiter_comparison.png
"""
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analytic import PROBLEMS

_RC = {
    "text.usetex": True,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.size": 12,
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

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
_LS = ["--", "-.", ":", (0, (3, 1, 1, 1))]


def load_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def plot_limiter_comparison(
    csvs, labels, analytic_csv, problem, t, N, outfile="limiter_comparison.png"
):
    plt.rcParams.update(_RC)

    x_a, rho_a, vx_a, P_a = load_csv(analytic_csv)

    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True, dpi=300)
    # fig.suptitle("Limiter comparison", fontsize=13)
    ylabels = [r"$\rho$", r"$u$", r"$p$"]

    for ax, ya, lbl in zip(axes, [rho_a, vx_a, P_a], ylabels):
        ax.plot(x_a, ya, "k-", lw=1.8, label="Exact", alpha=0.5)
        ax.set_ylabel(lbl)

    for csv, lab, c, ls in zip(csvs, labels, _COLORS, _LS):
        x_n, rho_n, vx_n, P_n = load_csv(csv)
        for ax, yn in zip(axes, [rho_n, vx_n, P_n]):
            ax.plot(x_n, yn, color=c, ls=ls, lw=1.0, label=lab, alpha=0.9)

    axes[0].text(
        0.97,
        0.95,
        rf"$t = {t}$,\ $N = {N}$",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )

    axes[0].legend(
        loc="lower center",
        ncols=3,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=10,
        frameon=False,
    )
    axes[-1].set_xlabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True)
    ap.add_argument(
        "--labels",
        nargs="+",
        default=["none (1st order)", "MinMod", "van Leer", "Superbee"],
    )
    ap.add_argument("--analytic", required=True)
    ap.add_argument("--problem", default="sod", choices=list(PROBLEMS.keys()))
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--out", default="limiter_comparison.png")
    args = ap.parse_args()
    plot_limiter_comparison(
        args.csvs, args.labels, args.analytic, args.problem, args.t, args.N, args.out
    )


if __name__ == "__main__":
    main()
