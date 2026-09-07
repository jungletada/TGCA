#!/usr/bin/env bash
set -euo pipefail

/home/peng/anaconda3/envs/tgca-repro/bin/python /home/peng/code/TGCA/analysis/spatial_graph_stability/run_phase_b.py --output-dir results/spatial_graph_stability/phase_b_graph_validation/20260907-phase-b-full-1a54d71 --phase-a-dir results/spatial_graph_stability/phase_a_basis_dependence/20260907-phase-a-full-1a54d71 --mct-checkpoint results/final_ln_ablation/20260906-mctformerplus-patch-final-ln-s0-64f2aa9/checkpoints/patch_final_ln/mctformerplus_final.pth --test-log results/spatial_graph_stability/test_logs/1a54d71.log

# Test record: /home/peng/code/TGCA/results/spatial_graph_stability/test_logs/1a54d71.log
# SHA256: 54214b395a249f3773a547b110aa45c95ee0d13abb41475b7bc14ee170878f9a
