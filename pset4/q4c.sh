#!/bin/bash
# Q4(c): Ra=1e4, N=16,32,64,128
# Nu vs mesh size convergence, compare to Blankenbach Nu=4.884

export OMP_NUM_THREADS=1
mkdir -p q4c_result

for N in 16 32 64 128; do
    echo "======================================="
    echo "running Ra=1e4, N=${N}x${N} mesh..."
    echo "======================================="
    python ./python/convection.py \
        --Ra 1e4 --N $N --dt 0.01 --t_max 1e5 --outdir q4c_result \
        2>&1 | tee q4c_result/STDOUT_Q4c_N${N}
    echo ""
done

echo "--- final Nu vs mesh size (Blankenbach benchmark = 4.884) ---"
for N in 16 32 64 128; do
    echo "  N=${N}: $(grep 'final Nu' q4c_result/STDOUT_Q4c_N${N})"
done