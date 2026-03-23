#!/bin/bash
#
# Q 8: performance and Scaling Study
# test grid sizes: 33x33, 65x65, 129x129, 257x257, 513x513
# test processor counts: 1, 2, 4
# Nonlinear problem: gamma=100, p=3
#

echo "=========================================="
echo "Q8: performance and Scaling Study"
echo "=========================================="
echo ""
echo "testing nonlinear problem (gamma=100, p=3)"
echo "Grid sizes: 33x33, 65x65, 129x129, 257x257, 513x513"
echo "Processors: 1, 2, 4"
echo ""
echo "15 cases total..."
echo ""

# Create results directory
mkdir -p q8_results

# file header
echo "# scaling Study Data" > ./q8_results/scaling_data.txt
echo "# refine  grid_size  nprocs  time_sec  newton_iters  rel_error" >> ./q8_results/scaling_data.txt

# grid refinement levels
refine_levels=(2 3 4 5 6)
grid_sizes=(33 65 129 257 513)

# n procs
nprocs_list=(1 2 4)

# total runs
total_runs=$((${#refine_levels[@]} * ${#nprocs_list[@]}))
current_run=0

# Loop over refinement levels and processor counts
for idx in ${!refine_levels[@]}; do
    refine=${refine_levels[$idx]}
    grid=${grid_sizes[$idx]}
    
    for nprocs in ${nprocs_list[@]}; do
        current_run=$((current_run + 1))
        
        echo "=========================================="
        echo "running $current_run/$total_runs: refine=$refine (${grid}x${grid}), nprocs=$nprocs"
        echo "=========================================="
        
        # Output file name
        outfile="run_refine${refine}_np${nprocs}.txt"
        
        # Run the solver
        mpirun -n $nprocs ./reaction2d \
            -rct_gamma 100 \
            -rct_p 3 \
            -da_refine $refine \
            -pc_type lu \
            -pc_factor_mat_solver_type mumps \
            -ksp_rtol 1e-12 \
            -snes_rtol 1e-10 \
            -snes_monitor \
            -log_view :./q8_results/${outfile}.log \
            > ./q8_results/$outfile 2>&1
        
        # Extract data
        time_sec=$(grep "Time (sec):" ./q8_results/${outfile}.log | head -1 | awk '{print $3}')
        newton_iters=$(grep -c "SNES Function norm" ./q8_results/$outfile)
        rel_error=$(grep "relative error" ./q8_results/$outfile | awk '{print $NF}')
        
        # append to data file
        echo "$refine  ${grid}x${grid}  $nprocs  $time_sec  $newton_iters  $rel_error" >> ./q8_results/scaling_data.txt
        
        echo "  Time: $time_sec sec"
        echo "  Newton iters: $newton_iters"
        echo "  Rel error: $rel_error"
        echo ""
    done
done

echo ""
echo "results saved in q8_results/"
echo "=========================================="