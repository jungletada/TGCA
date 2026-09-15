#!/usr/bin/env bash
set -euo pipefail
cd /home/peng/code/TGCA
exec /home/peng/anaconda3/envs/tgca-repro/bin/python -u -m experiments.run_diag_trajectory --dataset VOC12 "$@"
