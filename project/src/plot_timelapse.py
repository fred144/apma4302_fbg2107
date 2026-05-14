"""
5-panel density timelapse for a shock-tube problem (numeric vs exact).

The first panel is always t=0 (step-function ICs from PROBLEMS dict).
Remaining panels are loaded from CSV files produced by the solver.

python plot_timelapse.py --problem sod --N 400 \
    --snapshots snap_t0.05.csv snap_t0.10.csv snap_t0.15.csv snap_t0.20.csv \
    --times 0.05 0.10 0.15 0.20 \
    --out timelapse_sod.png
"""
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analytic import ExactRiemann, PROBLEMS

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
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def _ic_step(cfg):
    """Return (x, rho) step function for the t=0 ICs."""
    eps = 1e-9
    x = np.array([0.0, 0.5 - eps, 0.5 + eps, 1.0])
    rho = np.array([cfg["rho_L"], cfg["rho_L"], cfg["rho_R"], cfg["rho_R"]])
    return x, rho


def plot_timelapse(problem, N, snapshot_csvs, times, outfile="timelapse.png"):
    plt.rcParams.update(_RC)

    cfg = PROBLEMS[problem]
    rs = ExactRiemann(**cfg)

    all_times = [0.0] + list(times)
    all_csvs = [None] + list(snapshot_csvs)
    nrows = len(all_times)

    fig, axes = plt.subplots(nrows, 1, figsize=(5, 1.8 * nrows), sharex=True, dpi=200)
    if nrows == 1:
        axes = [axes]

    x_ref = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)

    for ax, t, csv in zip(axes, all_times, all_csvs):
        if t == 0.0:
            x_ic, rho_ic = _ic_step(cfg)
            ax.step(x_ic, rho_ic, where="post", color="k", lw=1.5, label="Exact")
            ax.step(
                x_ic,
                rho_ic,
                where="post",
                color="tab:red",
               
                lw=2.0,
                label="FVM",
                alpha=0.8,
            )
        else:
            x_n, rho_n, _, _ = load_csv(csv)
            rho_exact, _, _ = rs.solve(x_ref, t)
            ax.plot(x_ref, rho_exact, "k-", lw=1.5, label="Exact")
            ax.plot(
                x_n, rho_n, color="tab:red", ls="--", lw=1.0, label="FVM", alpha=0.85
            )

        ax.set_ylabel(r"$\rho$")
        ax.text(
            0.97,
            0.90,
            rf"$t = {t:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
           
        )
    axes[0].legend(loc="lower left")

    axes[0].set_title(rf"{problem.capitalize()} shock tube --- density, $N = {N}$")
    axes[-1].set_xlabel(r"$x$")
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", default="sod", choices=list(PROBLEMS.keys()))
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument(
        "--snapshots",
        nargs="+",
        required=True,
        help="CSV files from the solver, one per time (excluding t=0)",
    )
    ap.add_argument(
        "--times",
        type=float,
        nargs="+",
        required=True,
        help="Times corresponding to each snapshot CSV",
    )
    ap.add_argument("--out", default="timelapse.png")
    args = ap.parse_args()

    if len(args.snapshots) != len(args.times):
        ap.error("--snapshots and --times must have the same number of entries")

    plot_timelapse(args.problem, args.N, args.snapshots, args.times, args.out)


if __name__ == "__main__":
    main()
