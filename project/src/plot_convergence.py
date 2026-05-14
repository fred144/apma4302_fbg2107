"""
L1 convergence study: error in rho vs N cells, one curve per limiter.

Usage:
    python plot_convergence.py --problem sod --t 0.2 --grids 100 200 400 800 \
        --limiters none minmod vanleer superbee --out convergence.png
    (expects results_N<N>_<limiter>.csv files in the current directory)
"""
import argparse
import os
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

_COLORS  = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
_MARKERS = ["o", "s", "^", "D"]
_LABELS  = {
    "none":     "none (1st order)",
    "minmod":   "MinMod",
    "vanleer":  "van Leer",
    "superbee": "Superbee",
}


def load_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3]


def l1_error(x_num, q_num, x_ref, q_ref):
    q_interp = np.interp(x_num, x_ref, q_ref)
    return np.sum(np.abs(q_num - q_interp)) * (x_num[1] - x_num[0])


def plot_convergence(grids, limiters, problem, t, outfile="convergence.png"):
    plt.rcParams.update(_RC)

    cfg = PROBLEMS[problem]
    rs  = ExactRiemann(**cfg)

    fig, ax = plt.subplots(figsize=(5, 4))

    all_errs = []  # collect first limiter's errors for reference slopes

    for lim, color, marker in zip(limiters, _COLORS, _MARKERS):
        Ns, errs = [], []
        for N in grids:
            fname = f"results_N{N}_{lim}.csv"
            if not os.path.exists(fname):
                print(f"  WARNING: {fname} not found, skipping")
                continue
            x_n, rho_n, _, _ = load_csv(fname)
            x_ref = np.linspace(0.5 / N, 1.0 - 0.5 / N, N)
            rho_ref, _, _ = rs.solve(x_ref, t)
            err = l1_error(x_n, rho_n, x_ref, rho_ref)
            Ns.append(N)
            errs.append(err)
            print(f"  {lim:10s}  N={N:4d}  L1(rho) = {err:.4e}")

        if not Ns:
            continue

        Ns_arr   = np.array(Ns,   dtype=float)
        errs_arr = np.array(errs, dtype=float)
        label    = _LABELS.get(lim, lim)
        ax.loglog(Ns_arr, errs_arr, f"{marker}-", color=color, lw=1.4,
                  markersize=5, label=label)

        if not all_errs:
            all_errs = (Ns_arr, errs_arr)

    # reference slopes anchored to the first limiter's first point
    if all_errs:
        Ns0, errs0 = all_errs
        for order, ls, lbl in [(1, "--", r"$O(N^{-1})$"), (2, ":", r"$O(N^{-2})$")]:
            ref = errs0[0] * (Ns0[0] / Ns0) ** order
            ax.loglog(Ns0, ref, ls, color="gray", lw=1.0, label=lbl)

    ax.set_xlabel(r"$N$ cells")
    ax.set_ylabel(r"$L^1$ error in $\rho$")
    ax.set_xlim(90, 1e3)
    ax.legend(loc="lower left", title=f"{problem.capitalize()}, $t={t}$",
              fontsize=9, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"Saved {outfile}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem",  default="sod", choices=list(PROBLEMS.keys()))
    ap.add_argument("--t",        type=float, default=0.2)
    ap.add_argument("--grids",    type=int, nargs="+", default=[100, 200, 400, 800])
    ap.add_argument("--limiters", nargs="+",
                    default=["none", "minmod", "vanleer", "superbee"],
                    choices=["none", "minmod", "vanleer", "superbee"])
    ap.add_argument("--out",      default="convergence.png")
    args = ap.parse_args()
    plot_convergence(args.grids, args.limiters, args.problem, args.t, args.out)


if __name__ == "__main__":
    main()
