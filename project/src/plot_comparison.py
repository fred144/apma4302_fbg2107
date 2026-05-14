"""

python plot_comparison.py --numeric results_t0.2000.csv --analytic analytic_sod.csv \
                              --problem sod --t 0.2 --out comparison_sod.png
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


def load_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]  # x, rho, vx, P


def plot_comparison(numeric_csv, analytic_csv, problem, t, outfile="comparison.png"):
    plt.rcParams.update(_RC)

    x_n, rho_n, vx_n, P_n = load_csv(numeric_csv)
    x_a, rho_a, vx_a, P_a = load_csv(analytic_csv)
    N = len(x_n)

    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=True)
    axes[0].set_title(f"{problem.capitalize()} shock tube, $t = {t}$, $N = {N}$", fontsize=13)

    for ax, yn, ya, lbl in zip(
        axes, [rho_n, vx_n, P_n], [rho_a, vx_a, P_a], [r"$\rho$", r"$u$", r"$p$"]
    ):
        ax.plot(x_a, ya, "k-", lw=1.5, label="exact")
        ax.plot(x_n, yn, color="tab:red", ls="--", lw=1.0, label="FVM", alpha=0.85)
        ax.set_ylabel(lbl)
    axes[1].legend()

    # axes[0].text(
    #     0.97,
    #     0.95,
    #     rf"$t = {t}$,\ $N = {N}$",
    #     transform=axes[0].transAxes,
    #     ha="right",
    #     va="top",
    #     fontsize=11,
    # )
    axes[-1].set_xlabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--numeric", required=True)
    ap.add_argument("--analytic", required=True)
    ap.add_argument("--problem", default="sod", choices=list(PROBLEMS.keys()))
    ap.add_argument("--t", type=float, required=True)
    ap.add_argument("--out", default="comparison.png")
    args = ap.parse_args()
    plot_comparison(args.numeric, args.analytic, args.problem, args.t, args.out)


if __name__ == "__main__":
    main()
