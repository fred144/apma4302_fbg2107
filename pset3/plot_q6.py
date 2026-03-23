import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("q6_results/convergence_data.txt")
iterations = data[:, 0]
residuals = data[:, 1]

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(
    iterations,
    residuals,
    "o-",
    linewidth=2.5,
    markersize=9,
    color="steelblue",
    markeredgecolor="darkblue",
    markeredgewidth=0.5,
)
ax[0].set(xlabel="Newton Iteration", ylabel="Residual Norm $||F(u)||_2$")
ax[0].set_xlim(left=-0.3)

ax[1].semilogy(
    iterations,
    residuals,
    "o-",
    linewidth=2.5,
    markersize=9,
    color="steelblue",
    markeredgecolor="darkblue",
    markeredgewidth=0.5,
)
ax[1].set(xlabel="Newton Iteration", ylabel="Residual Norm $||F(u)||_2$")



ax[1].axhline(
    y=1e-10,
    color="red",
    linestyle="--",
    linewidth=2,
    label="Target: $10^{-10}$",
    alpha=0.7,
)
ax[1].legend(fontsize=12, loc="upper right")
ax[1].set_xlim(left=-0.3)

plt.savefig("q6_results/convergence_plot.png", dpi=300, bbox_inches="tight")
