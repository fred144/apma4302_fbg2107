#!/bin/bash
# for each (k, m) pair:
#   1. runs ./bvp and captures the relative error from stdout
#   2. saves bvp_solution.h5 with a unique name for that (k,m)
#
# at the end calls plot_bvp.py (modified for convergence plot).
#
#   ./run_3c.sh              # serial
#   ./run_3c.sh 4            # parallel with 4 processes


# number of MPI processes (default 1 for serial)
NP=${1:-1}

#fixed parameters for Q3(c)
GAMMA=0.0
C=3.0
K_VALUES=(1 5 10)
M_VALUES=(40 80 160 320 640 1280)


# icc works for serial; bjacobi+icc for parallel
if [ "$NP" -eq 1 ]; then
    KSP_OPTS="-ksp_type cg -pc_type icc"
else
    KSP_OPTS="-ksp_type cg -pc_type bjacobi -sub_pc_type icc"
fi

# tight solver tolerance so solver error doesn't pollute FD error
TOL_OPTS="-ksp_rtol 1e-12 -ksp_atol 1e-14"

# ── output file for errors (read by plot_bvp.py) ─────────────
ERRFILE="convergence_data.txt"
echo "# k  m  h  relative_error" > $ERRFILE

#  check executable exists 
if [ ! -f "./bvp" ]; then
    echo "ERROR: ./bvp not found. Run 'make bvp' first."
    exit 1
fi

echo "══════════════════════════════════════════════════════"
echo " Q3(c)  automated runs"
echo " NP=$NP  gamma=$GAMMA  c=$C"
echo " k values: ${K_VALUES[*]}"
echo " m values: ${M_VALUES[*]}"
echo "══════════════════════════════════════════════════════"

for K in "${K_VALUES[@]}"; do
    echo ""
    echo "─── k = $K ──────────────────────────────────────────"
    printf "%-8s %-12s %-14s\n" "m" "h" "rel_error"

    for M in "${M_VALUES[@]}"; do
        H=$(python3 -c "print(1.0/$M)")

        # run the solver, capture stdout
        OUTPUT=$(mpiexec -np $NP ./bvp \
            -bvp_m     $M      \
            -bvp_gamma $GAMMA  \
            -bvp_k     $K      \
            -bvp_c     $C      \
            $KSP_OPTS          \
            $TOL_OPTS          \
            2>/dev/null)

        # check it ran ok
        if [ $? -ne 0 ]; then
            echo "ERROR: bvp failed for k=$K m=$M"
            continue
        fi

        # extract relative error from line:
        # "m=XX  h=Y.Ye-ZZ  ||u-uexact||/||uexact|| = W.We-VV"
        REL_ERR=$(echo "$OUTPUT" \
            | grep "||u-uexact||" \
            | awk '{print $NF}')

        printf "%-8d %-12s %-14s\n" $M $H $REL_ERR

        # write to data file for python
        echo "$K  $M  $H  $REL_ERR" >> $ERRFILE

        # save HDF5 output for this (k,m) pair
        # (bvp.c always writes bvp_solution.h5, we rename it)
        if [ -f "bvp_solution.h5" ]; then
            cp bvp_solution.h5 "./bvp_hdf5_files/bvp_k${K}_m${M}.h5"
        fi
    done
done

echo ""
echo "══════════════════════════════════════════════════════"
echo " raw data written to: $ERRFILE"
echo " HDF5 files: bvp_k{k}_m{m}.h5"
echo "══════════════════════════════════════════════════════"

echo ""
echo "Generating convergence plot..."
python3 plot_bvp.py --convergence $ERRFILE

echo "done. Plot saved to q3c_convergence.png"