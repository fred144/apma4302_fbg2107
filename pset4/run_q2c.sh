#!/bin/bash

mkdir -p Q2_pc

echo "running C split_mg with log_view..."
./c/biharm -options_file ./c/options_file_split_mg -log_view \
    2>&1 | tee Q2_pc/c_split_mg.txt

echo "running Firedrake split_mg with log_view on 513x513 grid..."
python ./python/biharm_split_mg.py -log_view \
    2>&1 | tee Q2_pc/firedrake_split_mg.txt

echo ""
echo "results saved in Q2_pc/"
echo ""
echo "============================================"
echo " key event comparison: C vs Firedrake"
echo " columns: Time(s)  %T  Mflop/s"
echo "============================================"

for event in "SNESSolve" "KSPSolve" "KSPSetUp" "MatMult" "MatSOR" "MatAssemblyBegin" "MatAssemblyEnd" "MatLUFactorNum" "MatLUFactorSym" "MatSolve" "DMCreateMat" "PCSetUp" "PCSetUpOnBlocks" "PCApply"; do
    c_line=$(grep    "^${event} "  Q2_pc/c_split_mg.txt)
    fd_line=$(grep   "^${event} "  Q2_pc/firedrake_split_mg.txt)
    if [ -n "$c_line" ] || [ -n "$fd_line" ]; then
        echo ""
        echo "--- $event ---"
        [ -n "$c_line"  ] && echo "  C:         $(echo $c_line  | awk '{print "time="$4"s  %T="$10"  Mflop/s="$NF}')"
        [ -n "$fd_line" ] && echo "  Firedrake: $(echo $fd_line | awk '{print "time="$4"s  %T="$10"  Mflop/s="$NF}')"
    fi
done