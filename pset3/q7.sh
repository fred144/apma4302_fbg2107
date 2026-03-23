#!/bin/bash
#
# Q7: compare linear (gamma=0) vs nonlinear (gamma=100, p=3) with linear RHS
# run both cases and save VTK files for side-by-side comparison in ParaView
#

echo "=========================================="
echo "Q7: nonlinearity comparison"
echo "=========================================="
echo ""
echo "  Case 1: gamma=0   (linear, baseline)"
echo "  Case 2: gamma=100, p=3 (nonlinear)"
echo ""

# create dedicated Q7 directory
mkdir -p q7_results


echo "=========================================="
echo "case 1: Linear (gamma=0)"
echo "=========================================="
echo ""

mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -rct_linear_f \
    -da_refine 3 \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -ksp_rtol 1e-12 \
    -snes_rtol 1e-10 \
    -snes_monitor \
    -ksp_monitor \
    | tee q7_results/case1_gamma0_output.txt

# rename VTK file
mv ./reaction2d.vtr q7_results/case1_gamma0.vtr

echo ""
echo "Saved: case1_gamma0.vtr"
echo ""

# extract summary for Case 1
newton_iters_1=$(grep -c "SNES Function norm" q7_results/case1_gamma0_output.txt)
final_residual_1=$(grep "SNES Function norm" q7_results/case1_gamma0_output.txt | tail -1 | awk '{print $5}')
rel_error_1=$(grep "relative error" q7_results/case1_gamma0_output.txt | awk '{print $NF}')

echo "case 1 Summary:"
echo "  iterations: $newton_iters_1"
echo "  final residual:    $final_residual_1"
echo "  relative error:    $rel_error_1"
echo ""

echo "=========================================="
echo "case 2: nonlinear (gamma=100, p=3)"
echo "=========================================="
echo ""

mpirun -n 1 ./reaction2d \
    -rct_gamma 100 \
    -rct_p 3 \
    -rct_linear_f \
    -da_refine 3 \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -ksp_rtol 1e-12 \
    -snes_rtol 1e-10 \
    -snes_monitor \
    -ksp_monitor \
    | tee q7_results/case2_gamma100_output.txt

# rename VTK file
mv ./reaction2d.vtr q7_results/case2_gamma100.vtr

echo ""
echo "saved: case2_gamma100.vtr"
echo ""

# extract summary for Case 2
newton_iters_2=$(grep -c "SNES Function norm" q7_results/case2_gamma100_output.txt)
final_residual_2=$(grep "SNES Function norm" q7_results/case2_gamma100_output.txt | tail -1 | awk '{print $5}')
rel_error_2=$(grep "relative error" q7_results/case2_gamma100_output.txt | awk '{print $NF}')

echo "case 2 Summary:"
echo "  iterations: $newton_iters_2"
echo "  final residual:    $final_residual_2"
echo "  relative error:    $rel_error_2"
echo ""


echo "=========================================="
