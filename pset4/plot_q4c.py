# plot_q4c.py
# reads Nu history files from q4c_result
# plots Nu vs mesh size for Ra=1e4 and compares to Blankenbach benchmark Nu=4.884

import numpy as np
import matplotlib.pyplot as plt
import os

Ns         = [16, 32, 64, 128]
Nu_blanken = 4.884
Ra_str     = "Ra1e4"

Nu_final = []
Ns_found = []

for N in Ns:
    fname = f"q4c_result/Nu_history_{Ra_str}_N{N}.txt"
    if not os.path.exists(fname):
        print(f"missing {fname}, skipping")
        continue
    data = np.loadtxt(fname, comments='#')
    Nu_final.append(data[-1, 1])
    Ns_found.append(N)
    print(f"  N={N}:  final Nu = {data[-1, 1]:.6f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(Ns_found, Nu_final, 'bo-', linewidth=1.5, markersize=8, label='computed Nu')
ax.axhline(y=Nu_blanken, color='r', linestyle='--', linewidth=1.5,
           label=f'Blankenbach benchmark Nu={Nu_blanken}')
ax.set_xlabel('mesh size N')
ax.set_ylabel('Nu (steady state)')
ax.set_title('Nusselt number convergence Ra=1e4')
ax.set_xticks(Ns_found)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Nu_vs_meshsize_Ra1e4.png", dpi=150)
plt.close()
print("convergence plot saved to Nu_vs_meshsize_Ra1e4.png")