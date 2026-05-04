# q3_biharm_temp.py
# Based directly on biharm.py.
# CHANGES from biharm.py:
#   1. T(x,y) = (1-y) + A*cos(pi*x) defined as a Firedrake function, new additons
#   2. f = dT/dx using symbolic UFL differentiation T.dx(0), which replaces the manufactured solution RHS from biharm.py
#   3. u_true and error norm computation removed-- removed no exact solution
#   4. T written to output file alongside omega, psi, v for visualization 
#   5. output file renamed to q3_biharm_temp.pvd 

import firedrake_ts
from firedrake import *
import numpy as np

N = 2
levels = 8
Nfine = N*2**levels
base_mesh = UnitSquareMesh(N, N, quadrilateral=True)
Hierarchy = MeshHierarchy(base_mesh, levels)
mesh = Hierarchy[-1]
V = FunctionSpace(mesh, "Lagrange", 1)
ME = MixedFunctionSpace([V, V], name=["vorticity", "streamfunction"])
Vv = VectorFunctionSpace(mesh, "Lagrange", 1)

omega_t, psi_t = TestFunctions(ME)

u = Function(ME)
u.subfunctions[0].rename("vorticity")
u.subfunctions[1].rename("streamfunction")

omega, psi = split(u)

x, z = SpatialCoordinate(mesh)

# we changed replace manufactured solution RHS with temperature-driven forcing
A = Constant(0.1)
T = Function(V, name="temperature")
T.interpolate((1.0 - z) + A * cos(pi * x))      # T(x,y) = (1-y) + A*cos(pi*x)

f = Function(V, name="dT_dx")
f.interpolate(T.dx(0))                           # f = dT/dx, UFL symbolic differentiation
###

# u_true and manufactured solution removed - no exact solution for this problem

# Weak statement of the equations - UNCHANGED
Fomega = inner(grad(omega_t), grad(omega)) * dx - omega_t * f * dx
Fpsi   = inner(grad(psi_t),   grad(psi))   * dx - psi_t * omega * dx
F = Fomega + Fpsi

u.subfunctions[0].interpolate(0.)
u.subfunctions[1].interpolate(0.)

# boundary conditions - UNCHANGED
bcs = [DirichletBC(ME.sub(0), 0., "on_boundary"),
       DirichletBC(ME.sub(1), 0., "on_boundary")]

# solver parameters - UNCHANGED
pc = "fieldsplit"
params_general = {
    "snes_type": "ksponly",
    "ksp_rtol": 1e-6,
    "ksp_atol": 1e-10,
    "snes_monitor": None,
    "ksp_monitor": None
}
params = {
    "ksp_type": "fgmres",
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "multiplicative",
    "fieldsplit_0": {
        "ksp_type": "preonly",
        "pc_type": "mg",
        "pc_factor_mat_solver_type": "mumps"
    },
    "fieldsplit_1": {
        "ksp_type": "preonly",
        "pc_type": "mg",
        "pc_factor_mat_solver_type": "mumps"
    }
}
params.update(params_general)

problem = NonlinearVariationalProblem(F, u, bcs=bcs)
solver  = NonlinearVariationalSolver(problem, solver_parameters=params)
solver.solve()

v = Function(Vv, name="Velocity")
v.interpolate(curl(psi))

# changed write T to output, renamed output file
import os
os.makedirs("result", exist_ok=True)
outfile = VTKFile("result/q3_biharm_temp.pvd")
outfile.write(T, f, u.subfunctions[0], u.subfunctions[1], v)

print(f"\nworking on {Nfine}x{Nfine} mesh with {pc} preconditioner and {params['ksp_type']} solver")
print("output written to result/q3_biharm_temp.pvd")