static char help[] = "2D reaction-diffusion problem with DMDA and SNES.  Option prefix -rct_.\n\n";

#include <petsc.h>

// typedef struct {
//     PetscReal  rho, M, alpha, beta;
//     PetscBool  noRinJ;
// } AppCtx;

typedef struct
{
    PetscReal sigma;    // gaussian width
    PetscReal x0, y0;   // gaussian center
    PetscReal amp;      // gaussian amp
    PetscReal gamma;    // coefficient
    PetscInt p;         // exponent for u^p
    PetscBool linear_f; // flag: rct_linear_f, RHS to be f(x,y) = -nabla^2 u_exact,

} AppCtx;

// extern PetscReal f_source(PetscReal);
// extern PetscErrorCode InitialAndExact(DMDALocalInfo *, PetscReal *, PetscReal *, AppCtx *);

extern PetscErrorCode FormFunctionLocal(DMDALocalInfo *, PetscReal **, PetscReal **, AppCtx *);
extern PetscErrorCode FormJacobianLocal(DMDALocalInfo *, PetscReal **, Mat, Mat, AppCtx *);

// iniitialize
extern PetscReal uExact(PetscReal, PetscReal, AppCtx *);
extern PetscReal laplacianUExact(PetscReal, PetscReal, AppCtx *);

// similar to poisson2d.c
extern PetscErrorCode formExact(DM, Vec, AppCtx *);
extern PetscErrorCode formRHS(DM, Vec, AppCtx *);
extern PetscErrorCode formRankMap(DM, Vec, PetscInt);

// STARTMAIN
int main(int argc, char **args)
{
    DM da;
    SNES snes;
    AppCtx user;

    Vec u, uexact, f, rankmap; // added f and rankmap similar to poisson2d.c

    // PetscReal errnorm, *au, *auex;
    PetscReal errnorm, uexactnorm; // copied from poisson2d.c, to compute error norms
    DMDALocalInfo info;
    PetscViewer viewer;
    PetscMPIInt rank;

    PetscCall(PetscInitialize(&argc, &args, NULL, help));

    // get rank for viz
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    // user.rho = 10.0;
    // user.M = PetscSqr(user.rho / 12.0);
    // user.alpha = user.M;
    // user.beta = 16.0 * user.M;
    // user.noRinJ = PETSC_FALSE;

    user.gamma = 0.0;            // default: linear problem
    user.p = 3;                  // default exponent
    user.linear_f = PETSC_FALSE; // default: full nonlinear RHS
    user.sigma = 0.3;            // width, seems to be good value take from poisson2d.c
    user.x0 = 0.65;              // center x
    user.y0 = 0.65;              // center y
    user.amp = 1.0;              // amplitude

    // cmd line stuff 3(a)
    PetscOptionsBegin(PETSC_COMM_WORLD, "rct_", "options for reaction2d", "");
    PetscCall(PetscOptionsReal("-gamma", "coefficient",
                               "reaction2d.c", user.gamma, &(user.gamma), NULL));
    PetscCall(PetscOptionsInt("-p", "p in exponent for u^p term",
                              "reaction2d.c", user.p, &(user.p), NULL));
    PetscCall(PetscOptionsBool("-linear_f", "RHS to be f(x,y) = -nabla^2 u_exact",
                               "reaction2d.c", user.linear_f, &(user.linear_f), NULL));
    // PetscCall(PetscOptionsBool("-noRinJ", "do not include R(u) term in Jacobian",
    //                            "reaction.c", user.noRinJ, &(user.noRinJ), NULL));
    PetscOptionsEnd();

    /*
    3(c) 2D setup, defaults to 9x9 grid, can change with -da_grid_x M -da_grid_y N
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 9, 1, 1, NULL, &da));
    */
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD,
                           DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_STAR,
                           9, 9, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL, NULL, &da));

    // create linear system matrix A
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));
    PetscCall(DMSetApplicationContext(da, &user));

    // create vectors and matrix
    PetscCall(DMCreateGlobalVector(da, &u));
    PetscCall(VecDuplicate(u, &uexact));
    PetscCall(VecDuplicate(u, &f));
    PetscCall(VecDuplicate(u, &rankmap));
    // PetscCall(DMDAVecGetArray(da, u, &au));

    // 3(b) - boundary conditions using u_exact
    PetscCall(formExact(da, uexact, &user));

    /*
    initial guess to exact sol. want to converge in one step for linear case,
    and good initial guess for nonlinear case
    */
    PetscCall(VecCopy(uexact, u));

    // 3(c),  newton's method with user-provided residual and Jacobin
    PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
    PetscCall(SNESSetDM(snes, da)); // and 3(d) compatible with -snes_fd, -snes_fd_color, -snes_mf, -snes_mf_operator
    PetscCall(DMDASNESSetFunctionLocal(da, INSERT_VALUES,
                                       (DMDASNESFunctionFn *)FormFunctionLocal, &user));
    PetscCall(DMDASNESSetJacobianLocal(da,
                                       (DMDASNESJacobianFn *)FormJacobianLocal, &user));
    PetscCall(SNESSetFromOptions(snes));

    // solve F(u) = 0 for u, with initial guess u
    PetscCall(SNESSolve(snes, NULL, u));

    // 3e, for paraview vtk stuff
    PetscCall(formRHS(da, f, &user));
    PetscCall(formRankMap(da, rankmap, (PetscInt)rank));
    PetscCall(PetscViewerVTKOpen(PETSC_COMM_WORLD, "reaction2d.vtr",
                                 FILE_MODE_WRITE, &viewer));
    PetscCall(PetscObjectSetName((PetscObject)uexact, "u_exact"));
    PetscCall(PetscObjectSetName((PetscObject)u, "u"));
    PetscCall(PetscObjectSetName((PetscObject)f, "f"));
    PetscCall(PetscObjectSetName((PetscObject)rankmap, "rankmap"));
    PetscCall(VecView(uexact, viewer));
    PetscCall(VecView(u, viewer));
    PetscCall(VecView(f, viewer));
    PetscCall(VecView(rankmap, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    // 3(a), verify against exact solution, compute error norms
    PetscCall(VecNorm(uexact, NORM_2, &uexactnorm));
    PetscCall(VecAXPY(u, -1.0, uexact)); // u <- u - uexact
    PetscCall(VecNorm(u, NORM_2, &errnorm));
    PetscCall(DMDAGetLocalInfo(da, &info));

    // PetscCall(DMDAVecGetArray(da, uexact, &auex));
    // PetscCall(InitialAndExact(&info, au, auex, &user));
    // PetscCall(DMDAVecRestoreArray(da, u, &au));
    // PetscCall(DMDAVecRestoreArray(da, uexact, &auex));
    // PetscCall(VecAXPY(u, -1.0, uexact)); // u <- u + (-1.0) uexact
    // PetscCall(VecNorm(u, NORM_INFINITY, &errnorm));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD,
    //                       "on %d point grid:  |u-u_exact|_inf = %g\n", info.mx, errnorm));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "on %d x %d grid:  relative error |u-u_exact|_2/|u_exact|_2 = %.6e\n",
                          info.mx, info.my, errnorm / uexactnorm));

    // cleanup
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&uexact));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&rankmap));

    PetscCall(SNESDestroy(&snes));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}
// ENDMAIN

// PetscReal f_source(PetscReal x)
// {
//     return 0.0;
// }

// PetscErrorCode InitialAndExact(DMDALocalInfo *info, PetscReal *u0,
//                                PetscReal *uex, AppCtx *user)
// {
//     PetscInt i;
//     PetscReal h = 1.0 / (info->mx - 1), x;
//     for (i = info->xs; i < info->xs + info->xm; i++)
//     {
//         x = h * i;
//         u0[i] = user->alpha * (1.0 - x) + user->beta * x;
//         uex[i] = user->M * PetscPowReal(x + 1.0, 4.0);
//     }
//     return 0;
// }

// 3a and 3b, manufactured solution and RHS for 2D problem, exact, gaussian bump
// same as poisson2d.c , but we set the params 
// as in the user. struct above
PetscReal uExact(PetscReal x, PetscReal y, AppCtx *user)
{
    PetscReal r2 = (x - user->x0) * (x - user->x0) + (y - user->y0) * (y - user->y0);
    return user->amp * PetscExpReal(-r2 / (user->sigma * user->sigma));
}

// as well as laplacian of exact solution, for RHS
PetscReal laplacianUExact(PetscReal x, PetscReal y, AppCtx *user)
{
    PetscReal r2 = (x - user->x0) * (x - user->x0) + (y - user->y0) * (y - user->y0);
    PetscReal sigma2 = user->sigma * user->sigma;
    PetscReal expterm = PetscExpReal(-r2 / sigma2);
    return user->amp * expterm * 4.0 / sigma2 * (r2 / sigma2 - 1.0);
}
// and boundary conditions
PetscErrorCode formExact(DM da, Vec uexact, AppCtx *user)
{
    PetscInt i, j;
    PetscReal hx, hy, x, y, **auexact;
    DMDALocalInfo info;

    PetscCall(DMDAGetLocalInfo(da, &info));
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);
    PetscCall(DMDAVecGetArray(da, uexact, &auexact));
    for (j = info.ys; j < info.ys + info.ym; j++)
    {
        y = j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++)
        {
            x = i * hx;
            auexact[j][i] = uExact(x, y, user);
        }
    }
    PetscCall(DMDAVecRestoreArray(da, uexact, &auexact));
    return 0;
}
// and form RHS vector f, -linear_f flag controls RHS
PetscErrorCode formRHS(DM da, Vec frhs, AppCtx *user)
{
    PetscInt i, j;
    PetscReal hx, hy, x, y, **af, uex;
    DMDALocalInfo info;

    PetscCall(DMDAGetLocalInfo(da, &info));
    hx = 1.0 / (info.mx - 1);
    hy = 1.0 / (info.my - 1);
    PetscCall(DMDAVecGetArray(da, frhs, &af));
    for (j = info.ys; j < info.ys + info.ym; j++)
    {
        y = j * hy;
        for (i = info.xs; i < info.xs + info.xm; i++)
        {
            x = i * hx;
            // f = -laplacian(u_exact) for linear problem
            af[j][i] = -laplacianUExact(x, y, user);

            // For nonlinear problem, add gamma * u_exact^p
            if (!user->linear_f)
            {
                uex = uExact(x, y, user);
                af[j][i] += user->gamma * PetscPowReal(uex, user->p);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(da, frhs, &af));
    return 0;
}

// STARTFUNCTIONS

/*
form nonlinear residual F(u) = 0
nonlinear residual with gamma*u^p term
generalize to 2D
*/
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info, PetscReal **u,
                                 PetscReal **FF, AppCtx *user)
{
    PetscInt i, j;

    PetscReal hx, hy, hxhy, hyhx, x, y, uij, f_val, uex;
    // PetscReal h = 1.0 / (info->mx - 1), x, R;
    hx = 1.0 / (info->mx - 1);
    hy = 1.0 / (info->my - 1);
    hxhy = hx / hy;
    hyhx = hy / hx;

    // for (i = info->xs; i < info->xs + info->xm; i++)
    // {
    //     if (i == 0)
    //     {
    //         FF[i] = u[i] - user->alpha;
    //     }
    //     else if (i == info->mx - 1)
    //     {
    //         FF[i] = u[i] - user->beta;
    //     }
    //     else
    //     { // interior location
    //         if (i == 1)
    //         {
    //             FF[i] = -u[i + 1] + 2.0 * u[i] - user->alpha;
    //         }
    //         else if (i == info->mx - 2)
    //         {
    //             FF[i] = -user->beta + 2.0 * u[i] - u[i - 1];
    //         }
    //         else
    //         {
    //             FF[i] = -u[i + 1] + 2.0 * u[i] - u[i - 1];
    //         }
    //         R = -user->rho * PetscSqrtReal(u[i]);
    //         x = i * h;
    //         FF[i] -= h * h * (R + f_source(x));
    //     }
    // }

    // grid points owned by this processor, loop over y and x, compute F at each point
    for (j = info->ys; j < info->ys + info->ym; j++)
    {
        y = j * hy;
        for (i = info->xs; i < info->xs + info->xm; i++)
        {
            x = i * hx;

            // Dirichlet BC with u_exactF = u - u_exact
            if (i == 0 || i == info->mx - 1 || j == 0 || j == info->my - 1)
            {
                FF[j][i] = u[j][i] - uExact(x, y, user);
            }
            else // interior F = -laplacian(u) + gamma*u^p - f
            {
                uij = u[j][i];

                // -nabla^2 u = -(u_xx + u_yy), -laplacian(u)  using 5-point stencil
                FF[j][i] = -hyhx * (u[j][i - 1] + u[j][i + 1]) - hxhy * (u[j - 1][i] + u[j + 1][i]) + 2.0 * (hyhx + hxhy) * uij;

                // nonlinear reaction term: gamma * u^p
                FF[j][i] += hx * hy * user->gamma * PetscPowReal(uij, user->p);

                // -  RHS f(x,y)
                f_val = -laplacianUExact(x, y, user);
                if (!user->linear_f)
                {
                    uex = uExact(x, y, user);
                    f_val += user->gamma * PetscPowReal(uex, user->p);
                }
                FF[j][i] -= hx * hy * f_val;
            }
        }
    }
    return 0;
}

// 3c analytical Jacobian with derivative of gamma*u^p
// J is the Jacobian matrix, P is the preconditioning matrix, u is the current solution vector
PetscErrorCode FormJacobianLocal(DMDALocalInfo *info, PetscReal **u,
                                 Mat J, Mat P, AppCtx *user)
{
    PetscInt i, j, ncols; // col[3];
    PetscReal hx, hy, hxhy, hyhx, v[5], uij;
    MatStencil row, col[5];

    hx = 1.0 / (info->mx - 1);
    hy = 1.0 / (info->my - 1);
    hxhy = hx / hy;
    hyhx = hy / hx;

    // PetscReal h = 1.0 / (info->mx - 1), dRdu, v[3];

    // for (i = info->xs; i < info->xs + info->xm; i++)
    // {
    //     if ((i == 0) | (i == info->mx - 1))
    //     {
    //         v[0] = 1.0;
    //         PetscCall(MatSetValues(P, 1, &i, 1, &i, v, INSERT_VALUES));
    //     }
    //     else
    //     {
    //         col[0] = i;
    //         v[0] = 2.0;
    //         if (!user->noRinJ)
    //         {
    //             dRdu = -(user->rho / 2.0) / PetscSqrtReal(u[i]);
    //             v[0] -= h * h * dRdu;
    //         }
    //         col[1] = i - 1;
    //         v[1] = (i > 1) ? -1.0 : 0.0;
    //         col[2] = i + 1;
    //         v[2] = (i < info->mx - 2) ? -1.0 : 0.0;
    //         PetscCall(MatSetValues(P, 1, &i, 3, col, v, INSERT_VALUES));
    //     }
    // }

    // loop over grid points owned by this processor, compute Jacobian entries for each point
    for (j = info->ys; j < info->ys + info->ym; j++)
    {
        for (i = info->xs; i < info->xs + info->xm; i++)
        {
            row.j = j;
            row.i = i;

            // dF/du = 1 (diagonal only), bounday
            if (i == 0 || i == info->mx - 1 || j == 0 || j == info->my - 1)
            {
                v[0] = 1.0;
                col[0].j = j;
                col[0].i = i;
                PetscCall(MatSetValuesStencil(P, 1, &row, 1, col, v, INSERT_VALUES));
            }

            else // interior  5-point stencil
            {
                uij = u[j][i];
                ncols = 0;

                // center (diagonal), 2(hy/hx + hx/hy) + h^2 * gamma * p * u^(p-1)
                col[ncols].j = j;
                col[ncols].i = i;
                v[ncols] = 2.0 * (hyhx + hxhy);
                v[ncols] += hx * hy * user->gamma * user->p * PetscPowReal(uij, user->p - 1);
                ncols++;

                // left neighbor: -hy/hx
                if (i > 0)
                {
                    col[ncols].j = j;
                    col[ncols].i = i - 1;
                    v[ncols] = -hyhx;
                    ncols++;
                }

                // right neighbor: -hy/hx
                if (i < info->mx - 1)
                {
                    col[ncols].j = j;
                    col[ncols].i = i + 1;
                    v[ncols] = -hyhx;
                    ncols++;
                }

                //  -hx/hy, down neighbor
                if (j > 0)
                {
                    col[ncols].j = j - 1;
                    col[ncols].i = i;
                    v[ncols] = -hxhy;
                    ncols++;
                }

                // -hx/hy, up neighbor
                if (j < info->my - 1)
                {
                    col[ncols].j = j + 1;
                    col[ncols].i = i;
                    v[ncols] = -hxhy;
                    ncols++;
                }

                PetscCall(MatSetValuesStencil(P, 1, &row, ncols, col, v, INSERT_VALUES));
            }
        }
    }

    // assemble Jacobian matrix after setting all entries, for both J and P
    PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
    if (J != P)
    {
        PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
        PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
    }
    return 0;
}

// ENDFUNCTIONS

// STARTRANKMAP
//  form rankmap vector - shows which processor owns each grid point
//  using same as poisson2d.c, for paraview
PetscErrorCode formRankMap(DM da, Vec rankmap, PetscInt rank)
{
    PetscInt i, j;
    PetscReal **ab;
    DMDALocalInfo info;

    PetscCall(DMDAGetLocalInfo(da, &info));
    PetscCall(DMDAVecGetArray(da, rankmap, &ab));
    for (j = info.ys; j < info.ys + info.ym; j++)
    {
        for (i = info.xs; i < info.xs + info.xm; i++)
        {
            ab[j][i] = (PetscReal)rank;
        }
    }
    PetscCall(DMDAVecRestoreArray(da, rankmap, &ab));
    return 0;
}
// ENDRANKMAP