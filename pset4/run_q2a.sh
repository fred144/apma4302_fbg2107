#!/bin/bash

mkdir -p Q2_pa

echo "running direct solver..."
./c/biharm -options_file ./c/options_file_direct -log_view \
    > Q2_pa/output_direct.txt 2>&1

echo "running fieldsplit + direct..."
./c/biharm -options_file ./c/options_file_split_direct -log_view \
    > Q2_pa/output_split_direct.txt 2>&1

echo "running fieldsplit + multigrid..."
./c/biharm -options_file ./c/options_file_split_mg -log_view \
    > Q2_pa/output_split_mg.txt 2>&1

echo "results saved in Q2_pa/"
echo ""
echo "--- SNESSolve timings ---"
echo "  format: Event  Count  Ratio  Time(s)  Ratio  Flops  Ratio  ...  %T %F %M %L %R  Mflop/s"
echo "direct:        $(grep 'SNESSolve' Q2_pa/output_direct.txt)"
echo "split_direct:  $(grep 'SNESSolve' Q2_pa/output_split_direct.txt)"
echo "split_mg:      $(grep 'SNESSolve' Q2_pa/output_split_mg.txt)"