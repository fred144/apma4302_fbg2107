# %%
# note su
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analytic import PROBLEMS

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

outfile = "ics.png"
problems = ("sod", "lax")
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey="col", dpi=300)

ylabels = [r"density $\rho$", r"velocity $u$", r"pressure $p$"]
keys = ["rho", "vx", "P"]
colors = {"sod": "tab:blue", "lax": "tab:red"}

for prob in problems:
    cfg = PROBLEMS[prob]
    x = np.array([0.0, 0.5, 0.5, 1.0])
    vals = {
        "rho": np.array([cfg["rho_L"], cfg["rho_L"], cfg["rho_R"], cfg["rho_R"]]),
        "vx": np.array([cfg["vx_L"], cfg["vx_L"], cfg["vx_R"], cfg["vx_R"]]),
        "P": np.array([cfg["P_L"], cfg["P_L"], cfg["P_R"], cfg["P_R"]]),
    }

    for col, (key, lbl) in enumerate(zip(keys, ylabels)):
        ax = axes[col]
        line_style = "-" if prob == "sod" else "--"
        ax.step(
            x,
            vals[key],
            where="post",
            color=colors[prob],
            lw=2,
            ls=line_style,
            label=prob.capitalize(),
        )
        ax.axvline(0.5, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(lbl)

        ax.text(
            0.25,
            0.82 if prob == "sod" else 0.68,
            rf"$L:\ {vals[key][0]:.3g}$",
            transform=ax.transAxes,
            ha="center",
            color=colors[prob],
        )
        ax.text(
            0.75,
            0.82 if prob == "sod" else 0.68,
            rf"$R:\ {vals[key][2]:.3g}$",
            transform=ax.transAxes,
            ha="center",
            color=colors[prob],
        )

axes[0].legend(loc="lower left")

# fig.suptitle("Initial conditions ($t = 0$)", fontsize=13)
axes[1].set_title("initial conditions ($t = 0$)")
plt.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
