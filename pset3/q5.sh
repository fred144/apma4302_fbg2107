#!/bin/bash
#
# question 5: compare analytical vs finite-difference Jacobian
# test same problem (gamma=0, 65x65) with different Jacobian methods
#

echo "=========================================="
echo "Q5: Analytical vs FD Jacobian"
echo "=========================================="
echo ""

mkdir -p q5_results

# test 0: to verify Jacobian correctness
echo "test 0: Verify Jacobian correctness (-snes_test_jacobian)"
echo "Configuration: -rct_gamma 0 -da_refine 2 -snes_test_jacobian"
echo ""
mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -da_refine 2 \
    -snes_test_jacobian
echo ""
echo "Check output above for 'Norm of matrix ratio' and 'Norm of matrix difference'"
echo "Values should be small for correct Jacobian"
echo "=========================================="
echo ""
read -p "paused, press ENTER to continue with timing tests..."

# test 1: Analytical Jacobian (baseline from Q4)
echo "test 1: Analytical Jacobian (user-provided)"
echo "Configuration: -rct_gamma 0 -da_refine 3 -pc_type lu -pc_factor_mat_solver_type mumps -snes_monitor -ksp_monitor -log_view"
echo ""
mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -da_refine 3 \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -snes_monitor \
    -ksp_monitor \
    -log_view :q5_results/q5_test_1_analytical_timing.txt
echo ""
echo "Timing saved to: q5_results/q5_test_1_analytical_timing.txt"
echo "=========================================="
echo ""

# test 2: Finite-difference Jacobian (-snes_fd)
echo "test 2: Finite-difference Jacobian (-snes_fd)"
echo "Configuration: -rct_gamma 0 -da_refine 3 -pc_type lu -pc_factor_mat_solver_type mumps -snes_fd -snes_monitor -ksp_monitor -log_view"
echo ""
mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -da_refine 3 \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -snes_fd \
    -snes_monitor \
    -ksp_monitor \
    -log_view :q5_results/q5_test_2_fd_timing.txt
echo ""
echo "Timing saved to: q5_results/q5_test_2_fd_timing.txt"
echo "=========================================="
echo ""

# test 3: Colored finite-difference Jacobian (-snes_fd_color)
echo "test 3: Colored FD Jacobian (-snes_fd_color, recommended)"
echo "Configuration: -rct_gamma 0 -da_refine 3 -pc_type lu -pc_factor_mat_solver_type mumps -snes_fd_color -snes_monitor -ksp_monitor -log_view"
echo ""
mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -da_refine 3 \
    -pc_type lu \
    -pc_factor_mat_solver_type mumps \
    -snes_fd_color \
    -snes_monitor \
    -ksp_monitor \
    -log_view :q5_results/q5_test_3_fd_color_timing.txt
echo ""
echo "Timing saved to: q5_results/q5_test_3_fd_color_timing.txt"
echo "=========================================="
echo ""

# test 4: Matrix-free (-snes_mf)
echo "test 4: Matrix-free Jacobian (-snes_mf)"
echo "Configuration: -rct_gamma 0 -da_refine 3 -snes_mf -snes_monitor -ksp_monitor -log_view"
echo ""
mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -da_refine 3 \
    -snes_mf \
    -snes_monitor \
    -ksp_monitor \
    -log_view :q5_results/q5_test_4_mf_timing.txt
echo ""
echo "Timing saved to: q5_results/q5_test_4_mf_timing.txt"
echo "=========================================="
echo ""

# test 5: Matrix-free with FD operator (-snes_mf_operator)
echo "test 5: Matrix-free with FD operator (-snes_mf_operator)"
echo "Configuration: -rct_gamma 0 -da_refine 3 -snes_mf_operator -snes_monitor -ksp_monitor -log_view"
echo ""
mpirun -n 1 ./reaction2d \
    -rct_gamma 0 \
    -da_refine 3 \
    -snes_mf_operator \
    -snes_monitor \
    -ksp_monitor \
    -log_view :q5_results/q5_test_5_mf_operator_timing.txt
echo ""
echo "Timing saved to: q5_results/q5_test_5_mf_operator_timing.txt"
echo "=========================================="
echo ""

echo "SUMMARY OF tests"
echo "=========================================="
echo ""
echo "extractting timing comparison..."
echo ""

# Function to extract time
get_time() {
    local file=$1
    if [ -f "$file" ]; then
        grep "Time (sec):" "$file" | head -1 | awk '{print $3}'
    else
        echo "N/A"
    fi
}

# Create comparison
echo "Method               | Time (sec)"
echo "---------------------|------------"
echo "Analytical (Q4)      | $(get_time q5_results/q5_test_1_analytical_timing.txt)"
echo "FD (-snes_fd)        | $(get_time q5_results/q5_test_2_fd_timing.txt)"
echo "FD Color             | $(get_time q5_results/q5_test_3_fd_color_timing.txt)"
echo "Matrix-free          | $(get_time q5_results/q5_test_4_mf_timing.txt)"
echo "MF Operator          | $(get_time q5_results/q5_test_5_mf_operator_timing.txt)"
echo ""
echo "=========================================="
echo ""
echo "for detailed timing, see q5_results/q5_test_*_timing.txt files"
echo "should be SNESFunctionEval, SNESJacobianEval"
echo ""
echo "preferred save to  options_file_fd: -snes_fd_color"

