# %%
import numpy as np
import matplotlib.pyplot as plt

# Read data
data = np.loadtxt("q8_results/scaling_data.txt", dtype=str)

# Parse data
refine = data[:, 0].astype(int)
grid_size = data[:, 1]  # Keep as string (e.g., "33x33")
nprocs = data[:, 2].astype(int)
time_sec = data[:, 3].astype(float)
newton_iters = data[:, 4].astype(int)
rel_error = data[:, 5].astype(float)

# Extract grid sizes as integers for plotting
grid_nums = np.array([int(g.split("x")[0]) for g in grid_size])

# Unique values
unique_refine = np.unique(refine)
unique_nprocs = np.unique(nprocs)
unique_grids = np.unique(grid_nums)


fig, axes = plt.subplots(1, 5, figsize=(15, 3), dpi=300)
plt.subplots_adjust(wspace=0.35)
# Colors for different processor counts
colors = {1: "tab:blue", 2: "tab:orange", 4: "tab:green"}
markers = {1: "o", 2: "s", 4: "^"}

for np_val in unique_nprocs:
    mask = nprocs == np_val
    axes[0].plot(
        grid_nums[mask],
        time_sec[mask],
        marker=markers[np_val],
        markersize=8,
        linewidth=2,
        color=colors[np_val],
        label=f"{np_val} proc(s)",
    )
axes[0].set_xlabel("Grid Size (N)")
axes[0].set_ylabel("Runtime (seconds)")
axes[0].set_title("Runtime vs Grid Size")
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale("log")
axes[0].set_yscale("log")

# Get baseline (1 processor) times
baseline_times = {}
for r in unique_refine:
    mask = (refine == r) & (nprocs == 1)
    if np.any(mask):
        baseline_times[r] = time_sec[mask][0]

for np_val in [2, 4]:
    speedups = []
    grids_plot = []
    for r in unique_refine:
        mask = (refine == r) & (nprocs == np_val)
        if np.any(mask) and r in baseline_times:
            speedup = baseline_times[r] / time_sec[mask][0]
            speedups.append(speedup)
            grids_plot.append(unique_grids[list(unique_refine).index(r)])

    axes[1].plot(
        grids_plot,
        speedups,
        marker=markers[np_val],
        markersize=8,
        linewidth=2,
        color=colors[np_val],
        label=f"{np_val} procs",
    )

# Ideal speedup lines
axes[1].plot(
    unique_grids, [2] * len(unique_grids), "k--", alpha=0.5, label="Ideal (2x)"
)
axes[1].plot(unique_grids, [4] * len(unique_grids), "k:", alpha=0.5, label="Ideal (4x)")

axes[1].set_xlabel("Grid Size (N)")
axes[1].set_ylabel("Speedup")
axes[1].set_title("Parallel Speedup")
# axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale("log")


for np_val in unique_nprocs:
    mask = nprocs == np_val
    axes[2].plot(
        grid_nums[mask],
        newton_iters[mask],
        marker=markers[np_val],
        markersize=8,
        linewidth=2,
        color=colors[np_val],
        label=f"{np_val} proc(s)",
    )
axes[2].set_xlabel("Grid Size (N)")
axes[2].set_ylabel("Newton Iterations")
axes[2].set_title("Convergence: Newton Iterations")
# axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xscale("log")


for np_val in unique_nprocs:
    mask = nprocs == np_val
    axes[3].semilogy(
        grid_nums[mask],
        rel_error[mask],
        marker=markers[np_val],
        markersize=8,
        linewidth=2,
        color=colors[np_val],
        label=f"{np_val} proc(s)",
    )
axes[3].set_xlabel("Grid Size (N)")
axes[3].set_ylabel("Relative Error")
axes[3].set_title("Error vs Grid Size")
# axes[3].legend(fontsize=10)
axes[3].grid(True, alpha=0.3, which="both")
axes[3].set_xscale("log")

# Strong scaling for all grids
for np_val in unique_nprocs:
    mask = nprocs == np_val
    nprocs_strong = nprocs[mask]
    time_strong = time_sec[mask]
    grid_strong = grid_nums[mask]
    
    # Sort by grid size
    sort_idx = np.argsort(grid_strong)
    nprocs_strong = nprocs_strong[sort_idx]
    time_strong = time_strong[sort_idx]
    grid_strong = grid_strong[sort_idx]
    
    # Compute efficiency
    baseline_time = time_strong[0]
    efficiency = (baseline_time / time_strong) / nprocs_strong * 100
    
    axes[4].plot(
        grid_strong,
        efficiency,
        marker=markers[np_val],
        markersize=8,
        linewidth=2,
        color=colors[np_val],
        label=f"{np_val} proc(s)",
    )

axes[4].set_xlabel("Grid Size (N)")
axes[4].set_ylabel("Parallel Efficiency (%)")
axes[4].set_title("Scaling Efficiency")
# axes[4].legend(fontsize=10)
axes[4].grid(True, alpha=0.3)
axes[4].set_xscale("log")
axes[4].set_yscale("log")
axes[4].set_ylim([0, 110])

plt.savefig("q8_results/scaling_plots.png", dpi=300, bbox_inches="tight")


plt.show()
