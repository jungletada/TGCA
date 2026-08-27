#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start BCSS queue from a tracked dirty worktree." >&2
    exit 2
fi

commit=$(git rev-parse --short HEAD)
queue_dir="$repo_root/results/queues/$commit"
screen_id=${BCSS_SCREEN_ID:-"$(date +%Y%m%d-%H%M%S)"}
queue_log="$queue_dir/bcss-voc-screen-${screen_id}.log"
if [[ -e "$queue_log" ]]; then
    echo "Refusing to overwrite queue log: $queue_log" >&2
    exit 2
fi
mkdir -p "$queue_dir"
exec > >(tee -a "$queue_log") 2>&1

printf "QUEUE_STARTED started=%s commit=%s screen_id=%s\n" \
    "$(date --iso-8601=seconds)" "$commit" "$screen_id"
run_dirs=()
for variant in e0 e1 e2 e4 e5 e6; do
    printf "QUEUE_VARIANT_STARTED variant=%s started=%s\n" "$variant" "$(date --iso-8601=seconds)"
    run_id="${screen_id}-mctformerplus-voc-bcss-${variant}-s${BCSS_SEED:-0}-${commit}"
    BCSS_RUN_ID="$run_id" bash experiments/ablations/run_bcss_voc_variant.sh "$variant"
    run_dirs+=(--run-dir "$repo_root/results/mctformerplus/voc/$run_id")
    printf "QUEUE_VARIANT_COMPLETE variant=%s finished=%s\n" "$variant" "$(date --iso-8601=seconds)"
done
e6_run_dir="$repo_root/results/mctformerplus/voc/${screen_id}-mctformerplus-voc-bcss-e6-s${BCSS_SEED:-0}-${commit}"
CUDA_VISIBLE_DEVICES="${BCSS_GPU_ID:-0}" python -m analysis.sweep_parameters \
    --checkpoint "$e6_run_dir/mctformerplus_final.pth" --variant e6 \
    --voc-root "$repo_root/data/VOCdevkit/VOC2012" \
    --id-list "$repo_root/data/VOCdevkit/VOC2012/ImageLists/val_id.txt" \
    --max-images "${BCSS_SWEEP_IMAGES:-200}" \
    --output-dir "$e6_run_dir/bcss_diagnostics/parameter_sweep"
comparison_dir="$repo_root/results/mctformerplus/voc/comparisons/bcss-screen-${screen_id}-s${BCSS_SEED:-0}-${commit}"
python tools/collect_bcss_screen.py "${run_dirs[@]}" --output-dir "$comparison_dir"
printf "QUEUE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
