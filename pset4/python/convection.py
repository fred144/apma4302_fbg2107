# convection.py
# Solves infinite Prandtl number thermal convection as a monolithic DAE.
# System:
#   dT/dt + v.grad(T) = (1/Ra) Lap T   [temperature]
#   -Lap omega = dT/dx                  [vorticity]
#   -Lap psi   = omega                  [streamfunction]
#   v = (dpsi/dy, -dpsi/dx)
#
# BCs: T=1 bottom, T=0 top, dT/dx=0 sides (natural), omega=psi=0 everywhere
# IC:  T = (1-y) + 0.1*cos(pi*x), omega=psi=0
#
# Usage: python convection.py --Ra 1e4 --N 64 --t_max 1e5 --outdir result

from firedrake import *
import firedrake_ts
import numpy as np
import matplotlib.pyplot as plt
import os, argparse, math

parser = argparse.ArgumentParser()
parser.add_argument('--Ra',    type=float, default=1e4)
parser.add_argument('--N',     type=int,   default=64)
parser.add_argument('--dt',    type=float, default=0.1)
parser.add_argument('--t_max', type=float, default=1e5)
parser.add_argument('--outdir',type=str,   default='result')
args, _ = parser.parse_known_args()

Ra_val  = args.Ra
Nmesh   = args.N
dt_init = args.dt
t_max   = args.t_max
outdir  = args.outdir

# mesh
levels = int(round(math.log2(Nmesh / 4)))
N      = Nmesh // (2**levels)
Nfine  = N * 2**levels
print(f"mesh: N={N}, levels={levels}, Nfine={Nfine}", flush=True)

base_mesh = UnitSquareMesh(N, N, quadrilateral=True)
Hierarchy = MeshHierarchy(base_mesh, levels)
mesh      = Hierarchy[-1]

# function spaces
V  = FunctionSpace(mesh, "Lagrange", 1)
ME = MixedFunctionSpace([V, V, V], name=["temperature","vorticity","streamfunction"])

T_t, omega_t, psi_t = TestFunctions(ME)

u = Function(ME)
u.subfunctions[0].rename("temperature")
u.subfunctions[1].rename("vorticity")
u.subfunctions[2].rename("streamfunction")

# u_dot: only T component has time derivative; omega,psi are algebraic
u_dot = Function(ME, name="u_dot")
T_dot, _, _ = split(u_dot)

T, omega, psi = split(u)
x, y = SpatialCoordinate(mesh)

Ra     = Constant(Ra_val)
v_flow = as_vector([psi.dx(1), -psi.dx(0)])

# weak forms
F_T     = ( inner(T_dot, T_t)
          + inner(dot(v_flow, grad(T)), T_t)
          + (1/Ra) * inner(grad(T), grad(T_t)) ) * dx
F_omega = inner(grad(omega_t), grad(omega)) * dx - omega_t * T.dx(0) * dx
F_psi   = inner(grad(psi_t),   grad(psi))   * dx - psi_t * omega * dx
F = F_T + F_omega + F_psi

# initial conditions
u.subfunctions[0].interpolate((1 - y) + Constant(0.1) * cos(pi * x))
u.subfunctions[1].interpolate(Constant(0.0))
u.subfunctions[2].interpolate(Constant(0.0))

# boundary conditions
bcs = [
    DirichletBC(ME.sub(0), Constant(1.0), 3),              # T=1 bottom
    DirichletBC(ME.sub(0), Constant(0.0), 4),              # T=0 top
    DirichletBC(ME.sub(1), Constant(0.0), "on_boundary"),  # omega=0
    DirichletBC(ME.sub(2), Constant(0.0), "on_boundary"),  # psi=0
]

# solver parameters
params = {
    'ts_type':             'bdf',
    'ts_bdf_order':        2,
    'ts_dt':               dt_init,
    'ts_monitor':          None,
    'ts_rtol':             1e-6,
    'ts_atol':             1e-10,
    'ts_max_time':         t_max,
    'ts_adapt_dt_min':     1e-6,
    'ts_adapt_dt_max':     t_max / 100,   # at most 1% of total time per step
    'ts_exact_final_time': 'matchstep',
    'ksp_type':            'preonly',
    'pc_type':             'lu',
    'pc_factor_mat_solver_type': 'mumps',
}

# output
Ra_str = f"Ra{Ra_val:.0e}".replace('+','').replace('e0','e')
os.makedirs(outdir, exist_ok=True)
outfile = VTKFile(f"{outdir}/convection_{Ra_str}_N{Nfine}.pvd")
outfile.write(*u.subfunctions, time=0.0)

rank       = COMM_WORLD.rank
Nu_history = []
t_history  = []

def monitor(ts, step, t, x):
    dTdy_top    = assemble(u.subfunctions[0].dx(1) * ds(4))
    dTdy_bottom = assemble(u.subfunctions[0].dx(1) * ds(3))
    Nu = dTdy_top / dTdy_bottom
    if rank == 0:
        Nu_history.append(Nu)
        t_history.append(t)
        print(f"  t={t:.4f}  dt={ts.getTimeStep():.4e}  Nu={Nu:.6f}", flush=True)
    if step % 10 == 0:
        outfile.write(*u.subfunctions, time=t)

problem = firedrake_ts.DAEProblem(F, u, u_dot, (0.0, t_max), bcs=bcs)
solver  = firedrake_ts.DAESolver(problem, solver_parameters=params,
                                  monitor_callback=monitor)
solver.solve()

print(f"\ndone on {Nfine}x{Nfine} mesh, Ra={Ra_val:.0e}, t_max={t_max:.0e}", flush=True)

if rank == 0:
    print(f"final Nu = {Nu_history[-1]:.6f}", flush=True)

    np.savetxt(f"{outdir}/Nu_history_{Ra_str}_N{Nfine}.txt",
               np.column_stack([t_history, Nu_history]),
               header="t  Nu")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(t_history, Nu_history, 'b-', linewidth=1.5)
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Nu=1 (conduction)')
    ax.set_xlabel('time')
    ax.set_ylabel('Nu')
    ax.set_title(f'Nusselt number vs time  —  Ra={Ra_val:.0e}, {Nfine}x{Nfine} mesh')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/Nu_vs_time_{Ra_str}_N{Nfine}.png", dpi=150)
    plt.close()
    print(f"plot saved to {outdir}/Nu_vs_time_{Ra_str}_N{Nfine}.png", flush=True)