# plot_q4ab.py
# reads Nu history files from q4a_result and q4b_result
# produces a combined Nu vs time plot for Ra=1e2, 1e4, 1e5, 1e6

import numpy as np
import matplotlib.pyplot as plt
import os

Ra_vals = [1e2,          1e4,          1e5,     1e6  ]
colors  = ['steelblue',  'darkorange', 'green',  'red']
dirs    = ['q4a_result', 'q4b_result', 'q4b_result', 'q4b_result']

fig, ax = plt.subplots(figsize=(9, 5))

for Ra, color, d in zip(Ra_vals, colors, dirs):
    Ra_str = f"Ra{Ra:.0e}".replace('+', '').replace('e0', 'e')
    fname  = f"{d}/Nu_history_{Ra_str}_N64.txt"
    if not os.path.exists(fname):
        print(f"missing {fname}, skipping")
        continue
    data = np.loadtxt(fname, comments='#')
    t, Nu = data[:, 0], data[:, 1]
    ax.plot(t, Nu, color=color, linewidth=1.5, label=f'Ra={Ra:.0e}')
    print(f"  Ra={Ra:.0e}:  final Nu = {Nu[-1]:.6f}")

ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Nu=1 (conduction)')
ax.set_xscale('log')
ax.set_xlabel('time')
ax.set_ylabel('Nu')
ax.set_title('Nusselt number vs time  —  64x64 mesh')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Nu_vs_time_combined_q4ab.png", dpi=150)
plt.close()
print("combined plot saved to Nu_vs_time_combined_q4ab.png")



Ra_vals = [1e2,          1e4,          1e5,     1e6  ]
colors  = ['steelblue',  'darkorange', 'green',  'red']
dirs    = ['q4a_result', 'q4b_result', 'q4b_result', 'q4b_result']

fig, ax = plt.subplots(figsize=(9, 5))

for Ra, color, d in zip(Ra_vals, colors, dirs):
    Ra_str = f"Ra{Ra:.0e}".replace('+', '').replace('e0', 'e')
    fname  = f"{d}/Nu_history_{Ra_str}_N64.txt"
    if not os.path.exists(fname):
        print(f"missing {fname}, skipping")
        continue
    data = np.loadtxt(fname, comments='#')
    t, Nu = data[:, 0], data[:, 1]
    ax.plot(t, Nu, color=color, linewidth=1.5, label=f'Ra={Ra:.0e}')
    print(f"  Ra={Ra:.0e}:  final Nu = {Nu[-1]:.6f}")

ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Nu=1 (conduction)')
ax.set_xscale('log')
ax.set_xlabel('time')
ax.set_ylabel('Nu')
ax.set_ylim(0, 2)
ax.set_title('Nusselt number vs time  —  64x64 mesh')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("Nu_vs_time_combined_q4ab_zoomed.png", dpi=150)
plt.close()
print("combined plot saved to Nu_vs_time_combined_q4ab_zoomed.png")
