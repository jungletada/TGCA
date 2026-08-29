#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start Phase 2 from a tracked dirty worktree." >&2
    exit 2
fi

commit=$(git rev-parse --short=7 HEAD)
screen_id=${PSL_SCREEN_ID:-"$(date +%Y%m%d-%H%M%S)"}
queue_dir="$repo_root/results/queues/persistent-semantic"
queue_log="$queue_dir/${screen_id}-psl-phase2-${commit}.log"
baseline_dir="$repo_root/results/mctformerplus/voc/20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3"
if [[ ! -f "$baseline_dir/mctformerplus_final.pth" ]] \
        || [[ ! -f "$baseline_dir/raw_cam_diagnostics/metrics.json" ]] \
        || ! grep -q 'PIPELINE_COMPLETE' "$baseline_dir/pipeline.log"; then
    echo "Frozen E0 baseline is incomplete: $baseline_dir" >&2
    exit 2
fi
if [[ -e "$queue_log" ]]; then
    echo "Refusing to overwrite queue log: $queue_log" >&2
    exit 2
fi
mkdir -p "$queue_dir"
exec > >(tee -a "$queue_log") 2>&1

printf 'QUEUE_STARTED started=%s commit=%s screen_id=%s\n' \
    "$(date --iso-8601=seconds)" "$commit" "$screen_id"
run_args=()
for variant in read_only write_only read_write; do
    printf 'QUEUE_VARIANT_STARTED variant=%s started=%s\n' \
        "$variant" "$(date --iso-8601=seconds)"
    run_id="${screen_id}-mctformerplus-voc-psl-${variant}-s${PSL_SEED:-0}-${commit}"
    PSL_RUN_ID="$run_id" bash experiments/ablations/run_psl_voc_variant.sh "$variant"
    run_args+=(--run-dir "$repo_root/results/persistent_semantic/phase2/voc/$run_id")
    printf 'QUEUE_VARIANT_COMPLETE variant=%s finished=%s\n' \
        "$variant" "$(date --iso-8601=seconds)"
done

comparison_dir="$repo_root/results/persistent_semantic/phase2/voc/comparisons/${screen_id}-s${PSL_SEED:-0}-${commit}"
python tools/collect_psl_phase2.py \
    --baseline-run-dir "$baseline_dir" "${run_args[@]}" \
    --output-dir "$comparison_dir"
printf 'QUEUE_COMPLETE comparison=%s finished=%s\n' \
    "$comparison_dir" "$(date --iso-8601=seconds)"
