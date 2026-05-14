#!/usr/bin/env bash
# run_plots.sh — call all plotting routines, expecting CSVs already produced by run_sims.sh.
# Usage:  bash run_plots.sh
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=python3


echo "  Euler1D plotting"
echo "______________________"

# Sod comparison                                                     

echo
echo "--- comparison_sod.png ---"
$PYTHON plot_comparison.py \
    --numeric  sod_hllc_minmod_N400.csv \
    --analytic analytic_sod.csv \
    --problem  sod --t 0.2 \
    --out      comparison_sod.png

# Lax comparison                                                      
echo
echo "--- comparison_lax.png ---"
$PYTHON plot_comparison.py \
    --numeric  lax_hllc_minmod_N400.csv \
    --analytic analytic_lax.csv \
    --problem  lax --t 0.13 \
    --out      comparison_lax.png

# Convergence                                                    
echo
echo "--- convergence.png ---"
$PYTHON plot_convergence.py \
    --problem sod --t 0.2 \
    --grids 100 200 400 800 \
    --limiters none minmod vanleer superbee \
    --out convergence.png


# Limiter comparison                                

echo
echo "--- limiter_comparison.png ---"
$PYTHON plot_limiter_comparison.py \
    --csvs    results_lim_none.csv results_lim_minmod.csv \
              results_lim_vanleer.csv results_lim_superbee.csv \
    --labels  "none (1st order)" "MinMod" "van Leer" "Superbee" \
    --analytic analytic_sod.csv \
    --problem sod --t 0.2 --N 400 \
    --out limiter_comparison.png


# Timelapse                                                        
echo
echo "--- timelapse_sod.png ---"
$PYTHON plot_timelapse.py \
    --problem   sod --N 400 \
    --snapshots snap_t0.0500.csv snap_t0.1000.csv snap_t0.1500.csv snap_t0.2000.csv \
    --times     0.05 0.10 0.15 0.20 \
    --out       timelapse_sod.png

 
# Initial conditions                                                 
echo
echo "--- ics.png ---"
$PYTHON plot_ics.py

echo
echo "========================================"
echo "  fig produced:"
for f in comparison_sod.png comparison_lax.png convergence.png \
          limiter_comparison.png timelapse_sod.png ics.png; do
    [ -f "$f" ] && echo "    $f" || echo "    $f  [MISSING]"
done
echo "========================================"
