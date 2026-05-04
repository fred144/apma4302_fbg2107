#!/bin/bash

mkdir -p Q2_pb
mkdir -p result

echo "running direct solver..."
python python/biharm_direct.py \
    2>&1 | tee Q2_pb/output_direct.txt

echo "running fieldsplit + direct..."
python python/biharm_split_direct.py \
    2>&1 | tee Q2_pb/output_split_direct.txt

echo "running fieldsplit + multigrid..."
python python/biharm_split_mg.py \
    2>&1 | tee Q2_pb/output_split_mg.txt

echo ""
echo "results saved in Q2_pb/"
echo ""
echo "timings and errors"
echo "  format: solver | wall time (s) | KSP iters | L2 rel error (vorticity) | L2 rel error (streamfun)"
for label in "direct" "split_direct" "split_mg"; do
    file="Q2_pb/output_${label}.txt"
    time=$(grep 'Solve time' $file 2>/dev/null || echo "see file")
    iters=$(grep -c 'KSP Residual norm' $file)
    errors=$(grep 'L2 error' $file)
    echo ""
    echo "$label"
    echo "  KSP iters: $iters"
    echo "  $errors"
done