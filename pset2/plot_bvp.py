import numpy as np
import sys
import matplotlib.pyplot as plt
from petsc4py import PETSc

plt.rcParams.update(
    {
        "text.usetex": True,
        # "font.family": "Helvetica",
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.size": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.right": True,
        "xtick.top": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
    }
)

def read_hdf5_vec(filename, vec_name):
    """
    Read PETSc HDF5 viewer output and convert to numpy arrays.

    Parameters:
    filename: str - path to the HDF5 file
    vec_name: str - name of the vector to read

    Returns:
    numpy array containing the data
    """
    # Create a viewer for reading HDF5 files
    viewer = PETSc.Viewer().createHDF5(filename, "r")

    # Create a Vec to load the data
    vec = PETSc.Vec().create(comm=PETSc.COMM_WORLD)
    vec.setName(vec_name)
    vec.load(viewer)

    # Convert to numpy array
    array = vec.getArray()

    # Clean up
    vec.destroy()
    viewer.destroy()

    return array.copy()


def plot_bvp_solution(x, u_numeric, u_exact):
    """
    Plot the numerical and exact solutions of the BVP.

    Parameters:
    x: numpy array - grid points
    u_numeric: numpy array - numerical solution
    u_exact: numpy array - exact solution
    """
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(6, 5), dpi=200)
    ax2 = ax1.twinx()

    ax1.plot(x, u_numeric, "b-", label="Numerical Solution", linewidth=2)
    ax1.plot(x, u_exact, "r--", label="Exact Solution", linewidth=2)
    ax1.set_xlabel("x", fontsize=14)
    ax1.set_ylabel("u(x)", fontsize=14)
    ax1.set_title("BVP Numerical vs Exact Solution")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x, u_numeric - u_exact, "g--", label="Error", linewidth=1)
    ax2.set_ylabel("Error")
    ax2.legend(loc="lower right")
    ax2.set_ylim(
        -np.max(np.abs(u_numeric - u_exact)) * 3.0,
        np.max(np.abs(u_numeric - u_exact)) * 3.0,
    )
    plt.savefig("bvp_solution.png", dpi=200, bbox_inches="tight")

    plt.show()


def plot_convergence(datafile):
    """
    read convergence data written by run_3c.sh and plot
    error vs h on a log-log scale for each k value.

    datafile format (one line per run):
        k   m   h   relative_error

    for Q3(c): gamma=0, k in {1,5,10}, m in {40,80,...,1280}
    expected result: slope = 2 on log-log plot (O(h^2) convergence)
    """
    # read data file
    data = {}  # data[k] = {'h': [...], 'err': [...]}

    with open(datafile, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            k = int(parts[0])
            # m = int(parts[1])   # not needed for plot
            h = float(parts[2])
            err = float(parts[3])
            if k not in data:
                data[k] = {"h": [], "err": []}
            data[k]["h"].append(h)
            data[k]["err"].append(err)

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))

    markers = {1: "o", 5: "s", 10: "^"}
    colors = {1: "tab:blue", 5: "tab:orange", 10: "tab:green"}

    for k in sorted(data.keys()):
        h_arr = np.array(data[k]["h"])
        err_arr = np.array(data[k]["err"])

        # sort by h (ascending) in case they came in any order
        idx = np.argsort(h_arr)
        h_arr = h_arr[idx]
        err_arr = err_arr[idx]

        # convergence rate
        # log(err) = p * log(h) + const
        # np.polyfit on logs gives slope p = order of convergence
        coeffs = np.polyfit(np.log(h_arr), np.log(err_arr), 1)
        order = coeffs[0]

        ax.loglog(
            h_arr,
            err_arr,
            marker=markers.get(k, "o"),
            color=colors.get(k, "black"),
            linewidth=2,
            markersize=7,
            label=f"k = {k},  measured rate = {order:.2f}",
        )

        # print table to console
        print(f"\nk = {k}  (fitted order = {order:.3f})")
        print(f"{'h':>12}  {'error':>14}  {'ratio':>10}")
        for i in range(len(h_arr)):
            if i == 0:
                ratio_str = "   —"
            else:
                ratio = err_arr[i - 1] / err_arr[i]
                ratio_str = f"{ratio:10.3f}"
            print(f"{h_arr[i]:12.6f}  {err_arr[i]:14.4e}  {ratio_str}")

    # reference O(h^2) line
    all_h = np.concatenate([data[k]["h"] for k in data])
    h_min, h_max = min(all_h), max(all_h)
    h_ref = np.array([h_min, h_max])
    # scale reference line to sit below all curves
    ref_scale = min(data[k]["err"][0] for k in data) * 0.3
    ax.loglog(
        h_ref,
        ref_scale * (h_ref / h_ref[0]) ** 2,
        "k--",
        linewidth=1.5,
        label=r"$O(h^2)$ reference",
    )

    # formatting
    ax.set_xlabel(r"$h$ (grid spacing)", fontsize=13)
    ax.set_ylabel(r"$\|u_h - u_{\rm exact}\|\,/\,\|u_{\rm exact}\|$", fontsize=13)
    ax.legend(fontsize=11)

    plt.savefig("q3c_convergence.png", dpi=150)
    print("\nConvergence plot saved to q3c_convergence.png")
    plt.show()


# MAIN
#   python3 plot_bvp.py                      single solution plot
#   python3 plot_bvp.py --convergence FILE    convergence study plot
if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--convergence":
        # Q3(c): convergence study
        datafile = sys.argv[2]
        plot_convergence(datafile)
    else:
        # original behaviour: single solution plot
        h5_filename = "bvp_solution.h5"
        u = read_hdf5_vec(h5_filename, "u")
        u_exact = read_hdf5_vec(h5_filename, "uexact")
        x = np.linspace(0, 1, len(u))
        plot_bvp_solution(x, u, u_exact)
