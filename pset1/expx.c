#include <petsc.h>

int main(int argc, char **argv)
{
    PetscMPIInt rank, total_ranks;
    PetscInt i, N;
    PetscReal x = 1.0, localval, globalsum, x_use;

    PetscCall(PetscInitialize(&argc, &argv, NULL,
                              "Compute exp(x) in parallel with PETSc.\n\n"));
    PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(MPI_Comm_size(PETSC_COMM_WORLD, &total_ranks));

    // read option
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "options for expx", "");
    PetscCall(PetscOptionsReal("-x", "input to exp(x) function", NULL, x, &x, NULL));
    PetscCall(PetscOptionsInt("-N", "N terms in the polynomial approximation", NULL, N, &N, NULL));
    PetscOptionsEnd();

    // old code
    // // each process computes its local contribution
    // // compute  x^n/n!  where n = (rank of process) + 1`
    // localval = 1.0;
    // for (i = 1; i < rank + 1; i++)
    //     localval *= x / i;

    // // sum the contributions over all processes
    // PetscCall(MPI_Allreduce(&localval, &globalsum, 1, MPIU_REAL, MPIU_SUM,
    //                         PETSC_COMM_WORLD));

    /*
    we need to know which block is being
    handled by each process, so we can
    compute the local contribution
    */

    PetscInt k0 = (rank * N) / total_ranks;       // first index in this rank
    PetscInt k1 = ((rank + 1) * N) / total_ranks; // one past last index, in case N is not divisible by total_ranks

    /* cheeky since e^(-x) = 1/e^x, not sure how this is gonna factor into the performance*/
    PetscBool invert = PETSC_FALSE;
    if (x < 0.0)
    {
        x_use = -x;
        invert = PETSC_TRUE;
    }
    else
        x_use = x;

    PetscReal a_k = 1.0; // last term in block = 1/(k1-1)!
    if (k1 > 1)
    {
        for (i = 1; i < k1; i++)
            a_k /= i; // compute 1/(k1-1)!
    }

    for (i = k1 - 1; i >= k0; i--)
    {
        localval = a_k + x_use * localval; // horner update
        if (i > 0)
            a_k *= i; // move to previous coefficient: a_{i-1} = a_i * i
    }

    PetscCall(MPI_Reduce(&localval, &globalsum, 1, MPIU_REAL, MPIU_SUM, 0, PETSC_COMM_WORLD));

    if (rank == 0)
    {
        if (invert)
            globalsum = 1.0 / globalsum; // handle negative x

        PetscReal exact = exp(x);
        PetscReal relerr = PetscAbsReal(globalsum - exact) / PetscAbsReal(exact);
        PetscReal eps_machine = relerr / PETSC_MACHINE_EPSILON;

        PetscPrintf(PETSC_COMM_SELF,
                    "using blocked Horner: exp(%g) = %.32e (N=%d)\n", x, globalsum, N);
        PetscPrintf(PETSC_COMM_SELF,
                    "using exact: exp(%g) = %.32e\n", x, exact);
        PetscPrintf(PETSC_COMM_SELF,
                    "relative error: %.3e = %.3e  [machine eps]\n",
                    relerr, eps_machine);
    }

    // output estimate and report on work from each process
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    //                       "exp(%17.15f) is about %17.15f\n", x, globalsum));
    // PetscCall(PetscPrintf(PETSC_COMM_SELF,
    //                       "rank %d did %d flops\n", rank, (rank > 0) ? 2 * rank : 0));

    PetscCall(PetscFinalize());
    return 0;
}
