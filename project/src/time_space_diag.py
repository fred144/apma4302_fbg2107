#%% 
import matplotlib

import matplotlib.pyplot as plt

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

outfile = "time_space_diag.png"
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

x0 = 0.5
t_top = 1.0

# Axes and layout.
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, t_top)
ax.set_xlabel(r"position $x$")
ax.set_ylabel(r"time $t$")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Regions in the self-similar x-t diagram.
left_wave = [(x0, 0.0), (0.18, t_top)]
contact = [(x0, 0.0), (0.50, t_top)]
right_wave = [(x0, 0.0), (0.82, t_top)]

# Shade the star region bounded by the two outer waves.
star_poly_x = [0.18, 0.82, 1.0, 0.0]
star_poly_y = [t_top, t_top, 0.0, 0.0]
ax.fill(star_poly_x, star_poly_y, color="tab:orange", alpha=0.12, zorder=0)

# Draw the initial discontinuity and wave paths.
ax.plot([x0, x0], [0.0, 0.02], color="black", lw=2.0)
ax.plot(*zip(*left_wave), color="tab:blue", lw=2.2, ls="--")
ax.plot(*zip(*contact), color="tab:purple", lw=2.2, ls="-")
ax.plot(*zip(*right_wave), color="tab:red", lw=2.2, ls="--")

# Arrow labels for the waves.
ax.annotate("left wave", xy=(0.23, 0.78), xytext=(0.08, 0.92),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:blue"),
            color="tab:blue", fontsize=10)
ax.annotate("contact", xy=(0.52, 0.72), xytext=(0.55, 0.90),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:purple"),
            color="tab:purple", fontsize=10, ha="left")
ax.annotate("right wave", xy=(0.77, 0.78), xytext=(0.83, 0.92),
            arrowprops=dict(arrowstyle="->", lw=1.0, color="tab:red"),
            color="tab:red", fontsize=10, ha="left")

# Region labels.
ax.text(0.08, 0.58, r"left state $L$", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, ec="none"))
# Place the star-state labels inside the fan: halfway between the contact
# and the respective outer wave, and vertically centered in the fan.
ax.text(0.34, 0.55, r"left star state $L^*$", fontsize=11, ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, ec="none"))
ax.text(0.66, 0.55, r"right star state $R^*$", fontsize=11, ha="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, ec="none"))
ax.text(0.86, 0.58, r"right state $R$", fontsize=11, ha="right",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, ec="none"))

# Explanatory panel for the unknowns.
explanation = (
    r"Solve for $P^*$ and $u^*$" "\n"
    r"Then obtain $\rho_{L}^*$ and $\rho_{R}^*$" "\n"
    r"using the wave type on each side"
)
ax.text(0.03, 0.1, explanation, transform=ax.transAxes, va="bottom", ha="left",
        fontsize=11, bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
        alpha=0.9, ec="0.8"))

# Small marker at the initial discontinuity.
ax.text(x0 + 0.01, 0.03, r"$x_0$", fontsize=10, ha="left", va="bottom")
ax.text(x0 - 0.03, 0.02, r"$t=0$", fontsize=10, ha="right", va="bottom")

ax.set_title("Riemann problem: star states in time-space", fontsize=13)
ax.grid(alpha=0.15)

plt.savefig(outfile, dpi=200, bbox_inches="tight", pad_inches=0.1)
plt.show()


