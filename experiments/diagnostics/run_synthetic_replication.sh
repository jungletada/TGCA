#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

tgca_commit=$(git rev-parse --short HEAD)
tgca_run_id=${TGCA_RUN_ID:-"$(date +%Y%m%d)-synthetic-replication-${tgca_commit}"}
tgca_run_dir="$tgca_repo_root/results/mechanism/synthetic/$tgca_run_id"
if [[ -e "$tgca_run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $tgca_run_dir" >&2
    exit 2
fi
mkdir -p "$tgca_run_dir"
printf "bash experiments/diagnostics/run_synthetic_replication.sh\n" \
    > "$tgca_run_dir/command.txt"
printf '{"commit":"%s","branch":"%s","dirty":false}\n' \
    "$(git rev-parse HEAD)" "$(git branch --show-current)" \
    > "$tgca_run_dir/git_state.json"
python tools/test_token_replication.py \
    --output-dir "$tgca_run_dir/output" \
    --seed 2027 2>&1 | tee "$tgca_run_dir/pipeline.log"
printf "PIPELINE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)" \
    | tee -a "$tgca_run_dir/pipeline.log"
