static char help[] =
    "1D finite-volume solver for the compressible Euler equations.\n"
    "Uniform Cartesian grid, explicit forward-Euler time integration.\n\n"
    "Options:\n"
    "  -da_grid_x N        number of cells (default 400)\n"
    "  -cfl C              CFL number (default 0.4)\n"
    "  -t_end T            end time (default 0.2)\n"
    "  -output_freq F      save every F steps (default 10)\n"
    "  -riemann_type TYPE  lf | hllc (default hllc)\n"
    "  -limiter_type TYPE  none | minmod | vanleer | superbee (default minmod)\n"
    "  -problem TYPE       sod | lax | vacuum (default sod)\n"
    "  -gamma G            adiabatic index (default 5/3)\n\n";

#include <petsc.h>
#include <math.h>
#include <string.h>
#include <stdio.h>

/*
Index macros for the 3-component conserved/primitive state vector
U = [rho, rho*vx, E],   W = [rho, vx, P]
*/
#define RHO 0
#define MOM 1
#define ENE 2

/*
EOS and related functions: pressure, sound speed, primitive<->conserved
*/
static inline PetscReal pressure(const PetscReal *U, PetscReal gam)
{
    PetscReal rho = U[RHO];
    PetscReal vx = U[MOM] / rho;
    PetscReal E = U[ENE];
    return (gam - 1.0) * (E - 0.5 * rho * vx * vx);
}

static inline PetscReal soundspeed(PetscReal rho, PetscReal P, PetscReal gam)
{
    return PetscSqrtReal(gam * P / rho);
}

/* convert conserved U -> primitive W = [rho, vx, P] */
static inline void cons_to_prim(const PetscReal *U, PetscReal *W, PetscReal gam)
{
    W[RHO] = U[RHO];
    W[MOM] = U[MOM] / U[RHO];  /* vx */
    W[ENE] = pressure(U, gam); /* P  */
}

/* convert primitive W = [rho, vx, P] -> conserved U */
static inline void prim_to_cons(const PetscReal *W, PetscReal *U, PetscReal gam)
{
    U[RHO] = W[RHO];
    U[MOM] = W[RHO] * W[MOM];
    U[ENE] = W[ENE] / (gam - 1.0) + 0.5 * W[RHO] * W[MOM] * W[MOM];
}

/*  flux F(U) */
static inline void flux(const PetscReal *U, PetscReal *F, PetscReal gam)
{
    PetscReal rho = U[RHO];
    PetscReal vx = U[MOM] / rho;
    PetscReal E = U[ENE];
    PetscReal P = pressure(U, gam);
    F[RHO] = rho * vx;
    F[MOM] = rho * vx * vx + P;
    F[ENE] = (E + P) * vx;
}

/*
slope limiter phi(r)
and the various types
*/
typedef enum
{
    LIM_NONE,
    LIM_MINMOD,
    LIM_VANLEER,
    LIM_SUPERBEE
} LimiterType;

static inline PetscReal limiter(PetscReal r, LimiterType ltype)
{
    switch (ltype)
    {
    case LIM_NONE:
        return 0.0;
    case LIM_MINMOD:
        return PetscMax(0.0, PetscMin(1.0, r));
    case LIM_VANLEER:
        return (r + PetscAbsReal(r)) / (1.0 + PetscAbsReal(r));
    case LIM_SUPERBEE:
        return PetscMax(0.0, PetscMax(PetscMin(2.0 * r, 1.0), PetscMin(r, 2.0)));
    }
    return 0.0;
}

/*
MUSCL reconstruction: interface states at face i+1/2
W_L from cell i,  W_R from cell i+1  (using ghosts i-1 and i+2)
 */
static void reconstruct(const PetscReal wL[3], const PetscReal wC[3],
                        const PetscReal wR[3], const PetscReal wRR[3],
                        PetscReal ifL[3], PetscReal ifR[3],
                        LimiterType ltype)
{
    for (PetscInt d = 0; d < 3; d++)
    {
        PetscReal dL = wC[d] - wL[d];
        PetscReal dR = wR[d] - wC[d];
        PetscReal dRL = wRR[d] - wR[d];

        /* slope ratio for left-biased cell (cell i) */
        PetscReal rL = (PetscAbsReal(dR) > 1e-14) ? dL / dR : 0.0;
        /* slope ratio for right-biased cell (cell i+1) */
        PetscReal rR = (PetscAbsReal(dR) > 1e-14) ? dRL / dR : 0.0;

        ifL[d] = wC[d] + 0.5 * limiter(rL, ltype) * dR;
        ifR[d] = wR[d] - 0.5 * limiter(rR, ltype) * dRL;
    }
}

/*  LF numerical flux  */
static void flux_lf(const PetscReal *UL, const PetscReal *UR,
                    PetscReal *Fface, PetscReal dx, PetscReal dt,
                    PetscReal gam)
{
    PetscReal FL[3], FR[3];
    flux(UL, FL, gam);
    flux(UR, FR, gam);
    PetscReal alpha = dx / dt;
    for (PetscInt d = 0; d < 3; d++)
        Fface[d] = 0.5 * (FL[d] + FR[d]) - 0.5 * alpha * (UR[d] - UL[d]);
}

/*
HLLC numerical flux  (Toro et al. 1994)   
https://edanya.uma.es/NSPDE/images/ef%20toro/2-HLLHLLC.pdf                         
*/
static void flux_hllc(const PetscReal *UL, const PetscReal *UR,
                      PetscReal *Fface, PetscReal gam)
{
    PetscReal rhoL = UL[RHO], vxL = UL[MOM] / rhoL, EL = UL[ENE];
    PetscReal rhoR = UR[RHO], vxR = UR[MOM] / rhoR, ER = UR[ENE];
    PetscReal PL = pressure(UL, gam);
    PetscReal PR = pressure(UR, gam);
    PetscReal csL = soundspeed(rhoL, PL, gam);
    PetscReal csR = soundspeed(rhoR, PR, gam);

    /* Roe-averaged sound speeds for wave-speed estimates (simple version) */
    PetscReal SL = PetscMin(vxL - csL, vxR - csR);
    PetscReal SR = PetscMax(vxL + csL, vxR + csR);

    /* contact wave speed S* */
    PetscReal num = PR - PL + rhoL * vxL * (SL - vxL) - rhoR * vxR * (SR - vxR);
    PetscReal denom = rhoL * (SL - vxL) - rhoR * (SR - vxR);
    PetscReal Sstar = (PetscAbsReal(denom) > 1e-14) ? num / denom : 0.0;

    if (SL >= 0.0)
    {
        /* supersonic right: use left flux */
        flux(UL, Fface, gam);
    }
    else if (SR <= 0.0)
    {
        /* supersonic left: use right flux */
        flux(UR, Fface, gam);
    }
    else
    {
        /* subsonic: HLLC flux */
        PetscReal FL[3], FR[3];
        flux(UL, FL, gam);
        flux(UR, FR, gam);

        if (Sstar >= 0.0)
        {
            /* left star state */
            PetscReal coef = rhoL * (SL - vxL) / (SL - Sstar);
            PetscReal UstarL[3];
            UstarL[RHO] = coef;
            UstarL[MOM] = coef * Sstar;
            UstarL[ENE] = coef * (EL / rhoL + (Sstar - vxL) *
                                                  (Sstar + PL / (rhoL * (SL - vxL))));
            for (PetscInt d = 0; d < 3; d++)
                Fface[d] = FL[d] + SL * (UstarL[d] - UL[d]);
        }
        else
        {
            /* right star state */
            PetscReal coef = rhoR * (SR - vxR) / (SR - Sstar);
            PetscReal UstarR[3];
            UstarR[RHO] = coef;
            UstarR[MOM] = coef * Sstar;
            UstarR[ENE] = coef * (ER / rhoR + (Sstar - vxR) *
                                                  (Sstar + PR / (rhoR * (SR - vxR))));
            for (PetscInt d = 0; d < 3; d++)
                Fface[d] = FR[d] + SR * (UstarR[d] - UR[d]);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Write a snapshot CSV: gather the distributed Vec to rank 0, write  */
/* ------------------------------------------------------------------ */
static PetscErrorCode write_csv(DM da, Vec Uglobal, PetscReal t,
                                PetscReal gam, PetscMPIInt rank)
{
    DMDALocalInfo info;
    Vec U0; /* whole-domain copy on rank 0 */
    VecScatter ctx;
    char fname[128];

    PetscCall(DMDAGetLocalInfo(da, &info));

    /* Gather the full global vector to rank 0 in PETSc's internal order.
       For a 1D DMDA the internal layout is already contiguous in x, so
       this gives the correct spatial ordering. */
    PetscCall(VecScatterCreateToZero(Uglobal, &ctx, &U0));
    PetscCall(VecScatterBegin(ctx, Uglobal, U0, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx, Uglobal, U0, INSERT_VALUES, SCATTER_FORWARD));

    if (rank == 0)
    {
        const PetscScalar *arr;
        PetscCall(VecGetArrayRead(U0, &arr));
        PetscInt N = info.mx;
        PetscReal dx = 1.0 / N;

        PetscSNPrintf(fname, sizeof(fname), "results_t%.4f.csv", (double)t);
        FILE *fp = fopen(fname, "w");
        if (!fp)
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "cannot open output csv");
        fprintf(fp, "x,rho,vx,P,E\n");
        for (PetscInt i = 0; i < N; i++)
        {
            /* Each cell has 3 scalars: [RHO, MOM, ENE] */
            PetscReal Ui[3];
            Ui[RHO] = (PetscReal)arr[3 * i + RHO];
            Ui[MOM] = (PetscReal)arr[3 * i + MOM];
            Ui[ENE] = (PetscReal)arr[3 * i + ENE];
            PetscReal rho = Ui[RHO];
            PetscReal vx = Ui[MOM] / rho;
            PetscReal P = pressure(Ui, gam);
            PetscReal E = Ui[ENE];
            PetscReal x = (i + 0.5) * dx;
            fprintf(fp, "%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    (double)x, (double)rho, (double)vx, (double)P, (double)E);
        }
        fclose(fp);
        PetscCall(VecRestoreArrayRead(U0, &arr));
    }

    PetscCall(VecScatterDestroy(&ctx));
    PetscCall(VecDestroy(&U0));
    return 0;
}

/*Set ICs     */
typedef enum
{
    PROB_SOD,
    PROB_LAX,
    PROB_VACUUM
} ProblemType;

static PetscErrorCode set_initial_conditions(DM da, Vec Uglobal,
                                             ProblemType ptype, PetscReal gam)
{
    DMDALocalInfo info;
    PetscScalar *arr;

    PetscCall(DMDAGetLocalInfo(da, &info));
    /* Use flat VecGetArray to stay within the local allocation ([0, xm*3)).
       Cell i maps to local offset (i - xs)*3 + d.  No pointer shifting needed. */
    PetscCall(VecGetArray(Uglobal, &arr));

    PetscReal dx = 1.0 / info.mx;
    PetscReal xdisc = 0.5;

    // Left and right primitive states [rho, vx, P] 
    // phil mocz inspired https://github.com/pmocz/riemann-solver
    PetscReal WL[3], WR[3];
    switch (ptype)
    {
    case PROB_SOD:
        WL[RHO] = 1.0;
        WL[MOM] = 0.0;
        WL[ENE] = 1.0;
        WR[RHO] = 0.125;
        WR[MOM] = 0.0;
        WR[ENE] = 0.1;
        break;
    case PROB_LAX:
        WL[RHO] = 0.445;
        WL[MOM] = 0.698;
        WL[ENE] = 3.528;
        WR[RHO] = 0.5;
        WR[MOM] = 0.0;
        WR[ENE] = 0.571;
        break;
    case PROB_VACUUM:
        WL[RHO] = 1.0;
        WL[MOM] = -2.0;
        WL[ENE] = 0.4;
        WR[RHO] = 1.0;
        WR[MOM] = 2.0;
        WR[ENE] = 0.4;
        break;
    }

    for (PetscInt i = info.xs; i < info.xs + info.xm; i++)
    {
        PetscReal x = (i + 0.5) * dx;
        PetscReal *W = (x < xdisc) ? WL : WR;
        PetscReal Uc[3];
        prim_to_cons(W, Uc, gam);
        PetscInt li = (i - info.xs) * 3; /* local flat offset for cell i */
        arr[li + RHO] = Uc[RHO];
        arr[li + MOM] = Uc[MOM];
        arr[li + ENE] = Uc[ENE];
    }

    PetscCall(VecRestoreArray(Uglobal, &arr));
    return 0;
}

/*
 Apply outflow (transmissive) BCs to ghost cells
*/
static void apply_bcs(PetscScalar **u, DMDALocalInfo *info)
{
    /* Two left ghost cells if we own the left physical boundary */
    if (info->xs == 0)
    {
        for (PetscInt d = 0; d < 3; d++)
        {
            u[info->xs - 1][d] = u[info->xs][d];
            u[info->xs - 2][d] = u[info->xs][d];
        }
    }
    /* Two right ghost cells if we own the right physical boundary */
    if (info->xs + info->xm == info->mx)
    {
        PetscInt last = info->xs + info->xm - 1;
        for (PetscInt d = 0; d < 3; d++)
        {
            u[last + 1][d] = u[last][d];
            u[last + 2][d] = u[last][d];
        }
    }
}

/*
CFL timestep: local max wave speed, then global MPI_Allreduce
*/
static PetscErrorCode compute_dt(DM da, Vec Ulocal, PetscReal gam,
                                 PetscReal dx, PetscReal cfl,
                                 PetscReal t_end, PetscReal t,
                                 PetscReal *dt_out)
{
    DMDALocalInfo info;
    PetscScalar **u;
    PetscReal local_max = 0.0, global_max;

    PetscCall(DMDAGetLocalInfo(da, &info));
    PetscCall(DMDAVecGetArrayDOFRead(da, Ulocal, &u));
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++)
    {
        PetscReal rho = (PetscReal)u[i][RHO];
        PetscReal vx = (PetscReal)u[i][MOM] / rho;
        PetscReal E = (PetscReal)u[i][ENE];
        PetscReal P = (gam - 1.0) * (E - 0.5 * rho * vx * vx);
        PetscReal cs = soundspeed(rho, P, gam);
        PetscReal lam = PetscAbsReal(vx) + cs;
        if (lam > local_max)
            local_max = lam;
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(da, Ulocal, &u));
    /* global max wave speed across all ranks */
    PetscCallMPI(MPI_Allreduce(&local_max, &global_max, 1,
                               MPI_DOUBLE, MPI_MAX, PETSC_COMM_WORLD));
    /*we decide what dt is by
    the maximum wave speed across the whole domain, so that the fastest signal doesn't travel more than one cell per step.
    */
    PetscReal dt = (global_max > 1e-14) ? cfl * dx / global_max : 1e-6; // avoid zero division
    if (t + dt > t_end)
        dt = t_end - t;
    *dt_out = dt;
    return 0;
}

/*
tsetp loop
 ghost exchange > prim vars > reconstruct > flux >update
*/
typedef enum
{
    RIE_LF,
    RIE_HLLC
} RiemannType;

static PetscErrorCode step(DM da, Vec Uglobal, Vec Unew,
                           PetscReal dt, PetscReal dx, PetscReal gam,
                           RiemannType rtype, LimiterType ltype)
{
    DMDALocalInfo info;
    Vec Ulocal;
    PetscScalar **u;

    PetscCall(DMDAGetLocalInfo(da, &info));
    PetscCall(DMGetLocalVector(da, &Ulocal));

    // ghost exchange: fills MPI ghost cells; physical-boundary ghosts stay 0
    PetscCall(DMGlobalToLocalBegin(da, Uglobal, INSERT_VALUES, Ulocal));
    PetscCall(DMGlobalToLocalEnd(da, Uglobal, INSERT_VALUES, Ulocal));

    /* GetArrayDOF returns PetscScalar** (row-pointer array from VecGetArray2d).
       Each u[i] is a PetscScalar* pointing to the dof values for cell i.
       */
    PetscCall(DMDAVecGetArrayDOF(da, Ulocal, &u));

    // fill physical-boundary ghost cells with outflow BC values
    apply_bcs(u, &info);

    /* Scratch: primitive vars over [xs-2 .. xs+xm+1] (2 ghosts each side).
       MUSCL reconstruction at cell i needs W[i-2..i+2], so stencil=2 is
       required (set in DMDACreate1d).
    */
    PetscInt lo = info.xs - 2;
    PetscInt hi = info.xs + info.xm + 1; // inclusive: 2nd right ghost
    PetscInt nscratch = hi - lo + 1;     // = xm + 4 */

    PetscReal(*W)[3];
    PetscCall(PetscMalloc1(nscratch, &W));

    for (PetscInt i = lo; i <= hi; i++)
        cons_to_prim(u[i], W[i - lo], gam);

    /*
    Write updated conserved vars into Unew via flat VecGetArray to stay within
       the rank's local allocation ([0, xm*3)).  Cell i → offset (i-xs)*3+d.
    */
    PetscScalar *unewarr;
    PetscCall(VecGetArray(Unew, &unewarr));

    // loop over owned cells; compute face fluxes at i-1/2 and i+1/2
    for (PetscInt i = info.xs; i < info.xs + info.xm; i++)
    {
        PetscInt ii = i - lo;
        PetscInt li = (i - info.xs) * 3; // local flat offset for cell i

        // left face i-1/2
        PetscReal ifL_m[3], ifR_m[3];
        reconstruct(W[ii - 2], W[ii - 1], W[ii], W[ii + 1],
                    ifL_m, ifR_m, ltype);

        // right face i+1/2
        PetscReal ifL_p[3], ifR_p[3];
        reconstruct(W[ii - 1], W[ii], W[ii + 1], W[ii + 2],
                    ifL_p, ifR_p, ltype);

        // convert interface primitive states to conserved
        PetscReal UifL_m[3], UifR_m[3], UifL_p[3], UifR_p[3];
        prim_to_cons(ifL_m, UifL_m, gam);
        prim_to_cons(ifR_m, UifR_m, gam);
        prim_to_cons(ifL_p, UifL_p, gam);
        prim_to_cons(ifR_p, UifR_p, gam);

        // numerical fluxes
        PetscReal Fm[3], Fp[3];
        if (rtype == RIE_LF)
        {
            flux_lf(UifL_m, UifR_m, Fm, dx, dt, gam);
            flux_lf(UifL_p, UifR_p, Fp, dx, dt, gam);
        }
        else
        {
            flux_hllc(UifL_m, UifR_m, Fm, gam);
            flux_hllc(UifL_p, UifR_p, Fp, gam);
        }

        for (PetscInt d = 0; d < 3; d++)
            unewarr[li + d] = u[i][d] - (dt / dx) * (Fp[d] - Fm[d]);
    }

    PetscCall(PetscFree(W));
    PetscCall(DMDAVecRestoreArrayDOF(da, Ulocal, &u));
    PetscCall(VecRestoreArray(Unew, &unewarr));

    PetscCall(DMRestoreLocalVector(da, &Ulocal));
    return 0;
}

/* ------------------------------------------------------------------ */
/* main     loop                                                            
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    DM da;
    Vec Uglobal, Unew;
    DMDALocalInfo info;
    PetscMPIInt rank;
    PetscReal cfl = 0.4, t_end = 0.2, gam = 5.0 / 3.0;
    PetscInt output_freq = 10;
    char riemann_str[16], limiter_str[16], problem_str[16];
    PetscBool flg;

    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    /* ---------- parse options ---------- */
    PetscOptionsBegin(PETSC_COMM_WORLD, "", "Euler1D options", "");
    PetscCall(PetscOptionsReal("-cfl", "CFL number", "euler1d.c", cfl, &cfl, NULL));
    PetscCall(PetscOptionsReal("-t_end", "end time", "euler1d.c", t_end, &t_end, NULL));
    PetscCall(PetscOptionsReal("-gamma", "adiabatic index", "euler1d.c", gam, &gam, NULL));
    PetscCall(PetscOptionsInt("-output_freq", "output every N steps", "euler1d.c",
                              output_freq, &output_freq, NULL));

    PetscStrcpy(riemann_str, "hllc");
    PetscCall(PetscOptionsString("-riemann_type", "lf|hllc", "euler1d.c",
                                 riemann_str, riemann_str, sizeof(riemann_str), &flg));
    PetscStrcpy(limiter_str, "minmod");
    PetscCall(PetscOptionsString("-limiter_type", "none|minmod|vanleer|superbee",
                                 "euler1d.c", limiter_str, limiter_str,
                                 sizeof(limiter_str), &flg));
    PetscStrcpy(problem_str, "sod");
    PetscCall(PetscOptionsString("-problem", "sod|lax|vacuum", "euler1d.c",
                                 problem_str, problem_str, sizeof(problem_str), &flg));
    PetscOptionsEnd();

    /* interpret string options */
    RiemannType rtype = RIE_HLLC;
    if (strcmp(riemann_str, "lf") == 0 || strcmp(riemann_str, "LF") == 0)
        rtype = RIE_LF;

    LimiterType ltype = LIM_MINMOD;
    if (strcmp(limiter_str, "none") == 0 || strcmp(limiter_str, "NONE") == 0)
        ltype = LIM_NONE;
    else if (strcmp(limiter_str, "vanleer") == 0 || strcmp(limiter_str, "VL") == 0)
        ltype = LIM_VANLEER;
    else if (strcmp(limiter_str, "superbee") == 0 || strcmp(limiter_str, "SB") == 0)
        ltype = LIM_SUPERBEE;

    ProblemType ptype = PROB_SOD;
    if (strcmp(problem_str, "lax") == 0 || strcmp(problem_str, "LAX") == 0)
        ptype = PROB_LAX;
    else if (strcmp(problem_str, "vacuum") == 0 || strcmp(problem_str, "VAC") == 0)
        ptype = PROB_VACUUM;

    //// set up DMDA https://petsc.org/release/manualpages/DMDA/DMDA/
    /* 
    default 400 cells; override with -da_grid_x N 
    stencil width 2: MUSCL reconstruction needs 2 ghosts on each side 
    */
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_GHOSTED,
                           400, 3, 2, NULL, &da));
    PetscCall(DMSetFromOptions(da));
    PetscCall(DMSetUp(da));
    PetscCall(DMDAGetLocalInfo(da, &info));

    PetscReal dx = 1.0 / info.mx;

    PetscCall(DMCreateGlobalVector(da, &Uglobal));
    PetscCall(VecDuplicate(Uglobal, &Unew));
    // ICs
    PetscCall(set_initial_conditions(da, Uglobal, ptype, gam));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "euler1d: N=%d  dx=%.4e  riemann=%s  limiter=%s  problem=%s\n",
                          info.mx, (double)dx, riemann_str, limiter_str, problem_str));

    //LOOOOP
    PetscReal t = 0.0, dt = 0.0;
    PetscInt step_num = 0;

    /* Write t=0 snapshot */
    PetscCall(write_csv(da, Uglobal, t, gam, rank));

    Vec Ulocal;
    PetscCall(DMGetLocalVector(da, &Ulocal));

    while (t < t_end - 1e-14)
    {
        /* ghost exchange for CFL computation */
        PetscCall(DMGlobalToLocalBegin(da, Uglobal, INSERT_VALUES, Ulocal));
        PetscCall(DMGlobalToLocalEnd(da, Uglobal, INSERT_VALUES, Ulocal));

        PetscCall(compute_dt(da, Ulocal, gam, dx, cfl, t_end, t, &dt));

        PetscCall(step(da, Uglobal, Unew, dt, dx, gam, rtype, ltype));

        /* swap Uglobal <- Unew */
        Vec tmp = Uglobal;
        Uglobal = Unew;
        Unew = tmp;

        t += dt;
        step_num++;

        if (step_num % output_freq == 0 || t >= t_end - 1e-14)
        {
            PetscCall(write_csv(da, Uglobal, t, gam, rank));
            PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                                  "  step %4d  t = %.6f  dt = %.4e\n",
                                  step_num, (double)t, (double)dt));
        }
    }

    PetscCall(DMRestoreLocalVector(da, &Ulocal));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "Done: %d steps, t_final = %.6f\n", step_num, (double)t));

    PetscCall(VecDestroy(&Uglobal));
    PetscCall(VecDestroy(&Unew));
    PetscCall(DMDestroy(&da));
    PetscCall(PetscFinalize());
    return 0;
}
