#!/bin/bash
#
# q4
# gamma=0 on 65x65, parallel, compare nonlinear vs linear solve
#

set -e

echo "=========================================="
echo "QUESTION 4: gamma=0 Linear Problem, 65x65 grid"
echo "=========================================="
echo ""

mkdir -p q4_results

# test 1: nonlinear solve (Newton)
echo "test 1: Nonlinear solve (SNES Newton)"
echo "config: gamma=0, 65x65, mumps, strict tolerances, parallel (2 proc)"
echo ""
mpirun -n 2 ./reaction2d \
    -rct_gamma 0 \
    -da_grid_x 9 \
    -da_grid_y 9 \
    -da_refine 3 \
    -ksp_type gmres \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -ksp_rtol 1e-12 \
    -ksp_atol 1e-14 \
    -snes_atol 1e-10 \
    -snes_rtol 0 \
    -snes_monitor \
    -ksp_monitor \
    -snes_converged_reason \
    -ksp_converged_reason \
    > q4_results/nonlinear.out 2>&1
echo ""
echo "=========================================="
echo ""

# test 2: linear reference solve (same problem, ksponly)
echo "test 2: Linear reference solve (SNES KSPONLY)"
echo "config: same as test 1 + -snes_type ksponly"
echo ""
mpirun -n 2 ./reaction2d \
    -rct_gamma 0 \
    -da_grid_x 9 \
    -da_grid_y 9 \
    -da_refine 3 \
    -snes_type ksponly \
    -ksp_type gmres \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -ksp_rtol 1e-12 \
    -ksp_atol 1e-14 \
    -snes_atol 1e-10 \
    -snes_rtol 0 \
    -snes_monitor \
    -ksp_monitor \
    -snes_converged_reason \
    -ksp_converged_reason \
    > q4_results/linear_ref.out 2>&1
echo ""
echo "=========================================="
echo ""

# summary extraction
fnorm=$(grep "SNES Function norm" q4_results/nonlinear.out | tail -1 | awk '{print $5}')
newton_steps=$(awk '/SNES Function norm/{n++} END{print (n>0)?n-1:0}' q4_results/nonlinear.out)
err_nl=$(grep -i "relative error" q4_results/nonlinear.out | tail -1 | awk '{print $NF}')
err_lin=$(grep -i "relative error" q4_results/linear_ref.out | tail -1 | awk '{print $NF}')
err_diff=$(awk -v a="$err_nl" -v b="$err_lin" 'BEGIN{d=a-b; if(d<0)d=-d; print d}')

echo "Q4 SUMMARY" | tee q4_results/summary.txt
echo "------------------------------------------" | tee -a q4_results/summary.txt
echo "Final nonlinear residual ||F(u)||_2 = $fnorm" | tee -a q4_results/summary.txt
echo "Newton iterations = $newton_steps" | tee -a q4_results/summary.txt
echo "Final relative error (nonlinear) = $err_nl" | tee -a q4_results/summary.txt
echo "Final relative error (linear ref) = $err_lin" | tee -a q4_results/summary.txt
echo "Absolute difference in relative error = $err_diff" | tee -a q4_results/summary.txt
echo "" | tee -a q4_results/summary.txt
echo "Linear iterations at each Newton step (from nonlinear run):" | tee -a q4_results/summary.txt

# KSP iterations per Newton step
awk '
/SNES Function norm/ {s=$1; next}
/KSP Residual norm/ {k=$1+0; if (k>max[s]) max[s]=k}
END{
  for(i=0;i<=20;i++){
    if(i in max) printf("  Newton step %d: %d\n", i+1, max[i])
  }
}
' q4_results/nonlinear.out | tee -a q4_results/summary.txt

echo ""
echo "Done. See q4_results/summary.txt"
echo "DONE testing Question 4"