#!/bin/bash
# Q4(b): Ra=1e4, 1e5, 1e6 on 64x64 Q1 mesh
# BDF-2, monolithic MUMPS, t_max as per pset
export OMP_NUM_THREADS=1
mkdir -p q4b_result

echo "======================================="
echo "running Ra=1e4, 64x64, t_max=1e5..."
echo "======================================="
python ./python/convection.py \
    --Ra 1e4 --N 64 --dt 0.01 --t_max 1e5 --outdir q4b_result \
    2>&1 | tee q4b_result/STDOUT_Q4b_Ra1e4

echo "======================================="
echo "running Ra=1e5, 64x64, t_max=1e5..."
echo "======================================="
python ./python/convection.py \
    --Ra 1e5 --N 64 --dt 0.01 --t_max 1e5 --outdir q4b_result \
    2>&1 | tee q4b_result/STDOUT_Q4b_Ra1e5

echo "======================================="
echo "running Ra=1e6, 64x64, t_max=1e5..."
echo "======================================="
python ./python/convection.py \
    --Ra 1e6 --N 64 --dt 0.01 --t_max 1e5 --outdir q4b_result \
    2>&1 | tee q4b_result/STDOUT_Q4b_Ra1e6

echo ""
echo "--- final Nu values ---"
for Ra in 1e4 1e5 1e6; do
    echo "  Ra=${Ra}: $(grep 'final Nu' q4b_result/STDOUT_Q4b_Ra${Ra})"
done