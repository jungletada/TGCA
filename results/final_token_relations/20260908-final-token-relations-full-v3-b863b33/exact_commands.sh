#!/usr/bin/env bash
set -euo pipefail

/home/peng/anaconda3/envs/tgca-repro/bin/python /home/peng/code/TGCA/analysis/final_token_relations/run_final_token_relations.py --output-dir results/final_token_relations/20260908-final-token-relations-full-v3-b863b33 --batch-size 32 --num-workers 4 --bootstrap-repeats 5000 --bootstrap-seed 20260901 --test-log results/final_token_relations/test_logs/20260908-final-token-b863b33.log
