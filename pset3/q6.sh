#!/bin/bash
#
# question 6: Nonlinear problem (gamma=100, p=3)
# same stopping conditions as Q4
#

echo "=========================================="
echo "QUESTION 6: Nonlinear Problem"
echo "gamma=100, p=3, 65x65 grid"
echo "=========================================="
echo ""

# Create results directory
mkdir -p q6_results

echo "Configuration: Same as Q4 but with gamma=100, p=3"
echo ""

# run with same stopping conditions as Q4
mpirun -n 1 ./reaction2d \
    -rct_gamma 100 \
    -rct_p 3 \
    -da_refine 3 \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -ksp_rtol 1e-12 \
    -snes_rtol 1e-10 \
    -snes_monitor \
    -ksp_monitor \
    | tee q6_results/output.txt

echo ""
echo "=========================================="
echo "extracting data for plotting..."
echo "=========================================="
echo ""

# extract convergence data: iteration number and residual
echo "# Newton_Iteration  SNES_Residual_Norm" > q6_results/convergence_data.txt
grep "SNES Function norm" q6_results/output.txt | \
    awk '{print NR-1, $5}' >> q6_results/convergence_data.txt

echo "Saved: q6_results/convergence_data.txt"
echo ""

# Print summary
newton_iters=$(grep -c "SNES Function norm" q6_results/output.txt)
final_residual=$(grep "SNES Function norm" q6_results/output.txt | tail -1 | awk '{print $5}')
rel_error=$(grep "relative error" q6_results/output.txt | awk '{print $NF}')

echo "=========================================="
echo "we got that"
echo "=========================================="
echo ""
echo "newton iterations: $newton_iters"
echo "final residual:    $final_residual"
echo "relative error:    $rel_error"
echo ""

# print KSP iterations per Newton step
echo "KSP iterations at each Newton step:"
echo "------------------------------------"
awk '/SNES Function norm/{
    if (iter >= 0) {
        printf "Newton %d: %d KSP iterations\n", iter, ksp_count
    }
    iter = $1
    ksp_count = 0
}
/KSP Residual norm/{
    ksp_count++
}
END {
    printf "Newton %d: %d KSP iterations\n", iter, ksp_count
}' q6_results/output.txt

echo "=========================================="