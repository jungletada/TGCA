#!/usr/bin/env bash
set -euo pipefail

/home/peng/anaconda3/envs/tgca-repro/bin/python /home/peng/code/TGCA/analysis/spatial_graph_stability/run_phase_c.py --output-dir results/spatial_graph_stability/phase_c_graph_stability/20260907-phase-c-full-0d6a1b3 --phase-b-dir results/spatial_graph_stability/phase_b_graph_validation/20260907-phase-b-full-1a54d71 --mct-checkpoint results/final_ln_ablation/20260906-mctformerplus-patch-final-ln-s0-64f2aa9/checkpoints/patch_final_ln/mctformerplus_final.pth --test-log results/spatial_graph_stability/test_logs/phase_cd_0d6a1b3.log
