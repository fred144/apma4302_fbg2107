#!/usr/bin/env bash
# run_sims.sh — build the solver and run all simulations, producing CSV output files.
# Usage:  bash run_sims.sh [NP]   (NP = MPI ranks, default 4)
set -euo pipefail # exit on error, undefined variable, or failed pipe
cd "$(dirname "$0")"
NP=${1:-4}

echo  "____________________________________"
echo "  Euler1D simulation runs  (np=$NP)"
echo  "____________________________________"

echo
echo "--- Building euler1d ---"
make euler1d


# Sod: N=400, HLLC+MinMod, t=0.2                                     
echo
echo "--- Sod N=400 HLLC+MinMod (comparison) ---"
mpiexec -np $NP ./euler1d \
    -da_grid_x 400 -problem sod -t_end 0.2 \
    -riemann_type hllc -limiter_type minmod \
    -output_freq 99999 -gamma 1.6667
cp results_t0.2000.csv sod_hllc_minmod_N400.csv

python3 analytic.py --problem sod --t 0.2   --N 400 --out analytic_sod.csv


# Lax: N=400, HLLC+MinMod, t=0.13                                    
echo
echo "--- Lax N=400 HLLC+MinMod (comparison) ---"
mpiexec -np $NP ./euler1d \
    -da_grid_x 400 -problem lax -t_end 0.13 \
    -riemann_type hllc -limiter_type minmod \
    -output_freq 99999 -gamma 1.6667
cp results_t0.1300.csv lax_hllc_minmod_N400.csv

python3 analytic.py --problem lax --t 0.13 --N 400 --out analytic_lax.csv


# Convergence study: Sod, HLLC, all limiters, N=100/200/400/800      
echo
echo "--- Convergence study (all limiters) ---"
for LIM in none minmod vanleer superbee; do
    echo "  limiter=$LIM"
    for N in 100 200 400 800; do
        mpiexec -np $NP ./euler1d \
            -da_grid_x $N -problem sod -t_end 0.2 \
            -riemann_type hllc -limiter_type $LIM \
            -output_freq 99999 -gamma 1.6667
        mv results_t0.2000.csv results_N${N}_${LIM}.csv
    done
done


# Limiter comparison: Sod, HLLC, N=400, all limiters                 
echo
echo "--- Limiter comparison ---"
for LIM in none minmod vanleer superbee; do
    echo "  limiter=$LIM"
    mpiexec -np $NP ./euler1d \
        -da_grid_x 400 -problem sod -t_end 0.2 \
        -riemann_type hllc -limiter_type $LIM \
        -output_freq 99999 -gamma 1.6667
    mv results_t0.2000.csv results_lim_${LIM}.csv
done


# Timelapse snapshots: Sod, HLLC+MinMod, N=400, t=0.05..0.20        
echo
echo "--- Sod timelapse snapshots ---"
for T in 0.05 0.10 0.15 0.20; do
    TPAD=$(printf "%.4f" $T)
    mpiexec -np $NP ./euler1d \
        -da_grid_x 400 -problem sod -t_end $T \
        -riemann_type hllc -limiter_type minmod \
        -output_freq 99999 -gamma 1.6667
    mv results_t${TPAD}.csv snap_t${TPAD}.csv
done

echo
echo  "____________________________________"
echo "  CSV files produced:"
for f in analytic_sod.csv analytic_lax.csv \
          sod_hllc_minmod_N400.csv lax_hllc_minmod_N400.csv \
          results_N100.csv results_N200.csv results_N400.csv results_N800.csv \
          results_lim_none.csv results_lim_minmod.csv \
          results_lim_vanleer.csv results_lim_superbee.csv \
          snap_t0.0500.csv snap_t0.1000.csv snap_t0.1500.csv snap_t0.2000.csv; do
    [ -f "$f" ] && echo "    $f" || echo "    $f  [MISSING]"
done
echo  "____________________________________"
