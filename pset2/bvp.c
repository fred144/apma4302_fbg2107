static char help[] = "solve BVP: -u'' + gamma*u = f on [0,1]\n"
                     "manufactured solution: u(x) = sin(k*pi*x) + c*(x-0.5)^3\n"
                     "boundary conditions: u(0) = -c/8, u(1) = c/8\n"
                     "enforces BCs via MatZeroRowsColumns to preserve symmetry.\n"
                     "ex usage: $mpiexec -np numProcessors ./bvp -options_file options_file\n"
                     "Option prefix = bvp_.\n";

#include <petsc.h>
#include <petscviewerhdf5.h>

/*
exact solution:
    u(x) = sin(k*pi*x) + c*(x - 0.5)^3
boundary values:
    u(0) = c*(-0.5)^3 = -c/8
    u(1) = c*(0.5)^3  =  c/8
*/
static PetscReal uexact_func(PetscReal x, PetscReal k, PetscReal c)
{
    return PetscSinReal(k * PETSC_PI * x) + c * PetscPowReal(x - 0.5, 3);
}

/*
right hand side after substituting uexact into the PDE.
-u''(x) = k^2*pi^2 * sin(k*pi*x)  -  6c*(x - 0.5)
gamma*u =    gamma  * sin(k*pi*x)  +  gamma*c*(x-0.5)^3

so: f(x) = -u'' + gamma*u
       = (k^2*pi^2 + gamma)*sin(k*pi*x)  +  c*[ gamma*(x-0.5)^3  -  6*(x-0.5) ]
*/
static PetscReal f_func(PetscReal x, PetscReal k,
                        PetscReal gamma, PetscReal c)
{
    PetscReal sine_part = (k * k * PETSC_PI * PETSC_PI + gamma) * PetscSinReal(k * PETSC_PI * x);
    PetscReal cubic_part = c * (gamma * PetscPowReal(x - 0.5, 3) - 6.0 * (x - 0.5));
    return sine_part + cubic_part;
}

/* ═══════════════════════════════════════════════════════════ */
int main(int argc, char **args)
{
    Vec u, f, uexact; // u: numerical solution, f: RHS vector, uexact: exact solution
    Mat A;            // matrix for the linear system
    KSP ksp;          /*Krylov Subspace Solver context*/
    PetscViewer viewer;

    PetscInt Istart, Iend;

    // if you dont pass these options via command line, they will take default values as defined here
    PetscInt m = 40;         /* number of intervals */
    PetscReal gamma = 1.0;   /* reaction coefficien */
    PetscReal k_wave = 1.0;  /* wave number */
    PetscReal c_const = 1.0; /* cubic coefficient */

    /*
    PetscInitialize automatically handles -options_file so
   $ mpiexec -np P ./bvp -options_file options_file"
    */
    PetscCall(PetscInitialize(&argc, &args, NULL, help));

    // options parsing with prefix "bvp_"
    PetscOptionsBegin(PETSC_COMM_WORLD, "bvp_", "BVP options", NULL);
    PetscCall(PetscOptionsInt(
        "-m", "number of intervals (grid has m+1 points)",
        "bvp.c", m, &m, NULL));
    PetscCall(PetscOptionsReal(
        "-gamma", "reaction coefficient gamma",
        "bvp.c", gamma, &gamma, NULL));
    PetscCall(PetscOptionsReal(
        "-k", "wave number k in sin(k*pi*x)",
        "bvp.c", k_wave, &k_wave, NULL));
    PetscCall(PetscOptionsReal(
        "-c", "coefficient c in c*(x-0.5)^3",
        "bvp.c", c_const, &c_const, NULL));
    PetscOptionsEnd();

    /*
    determine the grid and system size,
    m intervals
    N = m + 1 grid points (including boundaries)
    the boundary rows will be handled later by MatZeroRowsColumns,
    so we include them in the system size N
    note this is differnt from tri.c where we had N=m and no BCs,
    here we have N=m+1 and will enforce BCs explicitly
    */
    PetscInt N = m + 1;
    PetscReal h = 1.0 / (PetscReal)m;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "BVP: m=%d  N=%d  h=%.6e  gamma=%.2f  k=%.1f  c=%.2f\n",
                          m, N, h, gamma, k_wave, c_const));

    // allocate vectors and matrix, using u instead of x in tri.c
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, N));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecDuplicate(u, &f));
    PetscCall(VecDuplicate(u, &uexact));

    PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
    PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, N, N));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

    /*
    make A, each process loops over its owned rows and sets values for A, f, uexact.
    row i maps to grid point x_i = i*h, and we can compute uexact[i] and f[i] from the formulas above

    stencil for interior rows: (-1/h^2, 2/h^2 + gamma, -1/h^2)
    boundary rows: placeholder diagonal=1, RHS=f[i]=uexact[i]
    (will be overwritten by MatZeroRowsColumns later,
    but we set it here for readability and
    to have correct BC values in the RHS after MatZeroRowsColumns does its adjustments)
    */
    for (PetscInt i = Istart; i < Iend; i++)
    {
        PetscReal xi = (PetscReal)i * h;
        PetscReal uval = uexact_func(xi, k_wave, c_const);
        PetscReal fval = f_func(xi, k_wave, gamma, c_const);

        /* uexact set at EVERY point (including boundaries) */
        PetscCall(VecSetValues(uexact, 1, &i, &uval, INSERT_VALUES));

        if (i == 0 || i == m)
        {
            /*
            this is a bouyndary row,
            we set a placeholder value for the diagonal and RHS
            row and col will be zeriod after MatZeroRowsColumns and diagonal
            the rhs becomes uexact[i] = u(x_i) which is the BC value at that point
            */

            PetscReal one = 1.0;
            PetscCall(MatSetValues(A, 1, &i, 1, &i, &one, INSERT_VALUES));
            PetscCall(VecSetValues(f, 1, &i, &uval, INSERT_VALUES));
        }
        else
        {
            /*
            this is an interior row, set the 3-point stencil for the PDE
            which looks like (-1/h^2)*u_{i-1} + (2/h^2 + gamma)*u_i + (-1/h^2)*u_{i+1} = f[i]
            */

            PetscReal diag = 2.0 / (h * h) + gamma;
            PetscReal offdiag = -1.0 / (h * h);
            PetscInt cols[3] = {i - 1, i, i + 1};
            PetscReal vals[3] = {offdiag, diag, offdiag};
            PetscCall(MatSetValues(A, 1, &i, 3, cols, vals, INSERT_VALUES));
            PetscCall(VecSetValues(f, 1, &i, &fval, INSERT_VALUES));
        }
    }

    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(uexact));
    PetscCall(VecAssemblyEnd(uexact));
    PetscCall(VecAssemblyBegin(f));
    PetscCall(VecAssemblyEnd(f));

    /*
      MatZeroRowsColumns
       for each boundary index i in {0, m}, PETSc does:
         (a) f[j] -= A[j][i] * uexact[i]  for all j != i
             (moves known BC contribution to RHS)
         (b) A[i][j] = A[j][i] = 0         for all j != i
             (zeros row AND column => preserves symmetry)
         (c) A[i][i] = 1.0,  f[i] = uexact[i]
             (equation becomes: u_i = BC value)
       matrix remains SYMMETRIC POSITIVE DEFINITE.
       CG and ICC work correctly on the modified system.
    */
    PetscInt bc_rows[2] = {0, m};
    PetscCall(MatZeroRowsColumns(A,
                                 2,       /* number of BC nodes   */
                                 bc_rows, /* global indices {0,m} */
                                 1.0,     /* new diagonal value   */
                                 uexact,  /* BC values vector     */
                                 f));     /* RHS modified in-place*/

    /*
    actually solve the linear system Au=f for u using KSP.
    it take  -ksp_type, -pc_type from options_file
    */
    PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp, f, u));

    /* print convergence reason */
    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "KSP converged reason: %s\n", KSPConvergedReasons[reason]));

    /* compute and print relative error in the solution*/
    Vec err_vec;
    PetscCall(VecDuplicate(u, &err_vec));
    PetscCall(VecCopy(u, err_vec));
    PetscCall(VecAXPY(err_vec, -1.0, uexact)); /* err = u - uexact */

    PetscReal errnorm, exactnorm;
    PetscCall(VecNorm(err_vec, NORM_2, &errnorm));
    PetscCall(VecNorm(uexact, NORM_2, &exactnorm));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "m=%d  h=%.6e  ||u-uexact||/||uexact|| = %.6e\n", m, h, errnorm / exactnorm));
    PetscCall(VecDestroy(&err_vec));

    /* output soluition, RHS, and exact solution to hdf5*/
    PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "bvp_solution.h5",
                                  FILE_MODE_WRITE, &viewer));
    PetscCall(PetscObjectSetName((PetscObject)uexact, "uexact"));
    PetscCall(PetscObjectSetName((PetscObject)f, "f"));
    PetscCall(PetscObjectSetName((PetscObject)u, "u"));
    PetscCall(VecView(f, viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(VecView(uexact, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    /*clean*/
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&uexact));
    PetscCall(PetscFinalize());
    return 0;
}