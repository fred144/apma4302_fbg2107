# monolithic LU (equivalent to options_file_direct)
import firedrake_ts
from firedrake import *
import numpy as np
import time

N = 2
levels = 8
Nfine = N*2**levels + 1  # 513 points, matching da_refine 8 from 3x3 in PETSc
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

f = Function(V, name="rhs")
x, z = SpatialCoordinate(mesh)
cx = x**3 * (1.0-x)**3
cz = z**3 * (1.0-z)**3
ddcx = 6.0 * x * (1.0-x) * (1.0 - 5.0 * x + 5.0 * x*x)
ddcz = 6.0 * z * (1.0-z) * (1.0 - 5.0 * z + 5.0 * z*z)
d4cx = - 72.0 * (1.0 - 5.0 * x + 5.0 * x*x)
d4cz = - 72.0 * (1.0 - 5.0 * z + 5.0 * z*z)
f.interpolate(d4cx * cz + 2.0 * ddcx * ddcz + cx * d4cz)

u_true = Function(ME, name="u_true")
u_true.subfunctions[0].interpolate(-ddcx * cz - cx * ddcz)
u_true.subfunctions[1].interpolate(cx * cz)

Fomega = inner(grad(omega_t), grad(omega)) * dx - omega_t * f * dx
Fpsi   = inner(grad(psi_t),   grad(psi))   * dx - psi_t * omega * dx
F = Fomega + Fpsi

u.subfunctions[0].interpolate(0.)
u.subfunctions[1].interpolate(0.)

bcs = [DirichletBC(ME.sub(0), 0., "on_boundary"),
       DirichletBC(ME.sub(1), 0., "on_boundary")]

pc = "monolithic"
params = {
    "snes_type": "ksponly",
    "ksp_type": "preonly",
    "ksp_monitor": None,
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps"
}

problem = NonlinearVariationalProblem(F, u, bcs=bcs)
solver  = NonlinearVariationalSolver(problem, solver_parameters=params)
t0 = time.time()
solver.solve()
t1 = time.time()
print(f"Solve time: {t1-t0:.4f} s")

v = Function(Vv, name="Velocity")
v.interpolate(curl(psi))
outfile = VTKFile("result/biharm_direct.pvd")
outfile.write(f, u.subfunctions[0], u.subfunctions[1], v)

abs_error_o = assemble(sqrt(inner(u.subfunctions[0] - u_true.subfunctions[0], u.subfunctions[0] - u_true.subfunctions[0])) * dx)
abs_error_p = assemble(sqrt(inner(u.subfunctions[1] - u_true.subfunctions[1], u.subfunctions[1] - u_true.subfunctions[1])) * dx)
rel_error_o = abs_error_o / assemble(sqrt(inner(u_true.subfunctions[0], u_true.subfunctions[0])) * dx)
rel_error_p = abs_error_p / assemble(sqrt(inner(u_true.subfunctions[1], u_true.subfunctions[1])) * dx)
print(f"\nworking on {Nfine}x{Nfine} mesh with {pc} preconditioner and {params['ksp_type']} solver")
print(f"L2 error (vorticity) -- abs: {abs_error_o:.3e}, rel: {rel_error_o:.3e}")
print(f"L2 error (streamfun) -- abs: {abs_error_p:.3e}, rel: {rel_error_p:.3e}")