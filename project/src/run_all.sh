#!/usr/bin/env bash
# run_figures.sh — full pipeline: simulations then plots.
# Usage:  bash run_figures.sh [NP]   (NP = MPI ranks, default 4)
set -euo pipefail
cd "$(dirname "$0")"
NP=${1:-4}

bash run_sims.sh  $NP
bash run_plots.sh
