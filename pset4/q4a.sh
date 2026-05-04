#!/bin/bash
mkdir -p q4a_result
echo "running Ra=1e2 on 64x64 mesh..."
python ./python/convection.py \
    --Ra 1e2 --N 64 --t_max 1e5 --outdir q4a_result \
    2>&1 | tee q4a_result/STDOUT_Q4a
echo "done. final Nu should be ~1.0 (non-convective)"