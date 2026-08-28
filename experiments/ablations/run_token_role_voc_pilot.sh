#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

gpu_id=${TOKEN_ROLE_GPU_ID:-0}
seed=${TOKEN_ROLE_SEED:-0}
diagnostic_images=${TOKEN_ROLE_DIAGNOSTIC_IMAGES:-200}
commit=$(git rev-parse --short HEAD)
queue_id=${TOKEN_ROLE_QUEUE_ID:-"$(date +%Y%m%d-%H%M%S)"}
voc_root="$repo_root/data/VOCdevkit/VOC2012"
cam_list="$voc_root/ImageLists/train_id.txt"
baseline_run="$repo_root/results/mctformerplus/voc/20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3"
baseline_checkpoint="$baseline_run/mctformerplus_final.pth"
comparison_dir="$repo_root/results/mctformerplus/voc/comparisons/token-role-pilot-${queue_id}-s${seed}-${commit}"

if [[ -e "$comparison_dir" ]]; then
    echo "Refusing to overwrite comparison directory: $comparison_dir" >&2
    exit 2
fi
test -f "$baseline_checkpoint"
mkdir -p "$comparison_dir"

printf "STAGE=shared_diagnostics started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" python tools/analyze_token_roles.py \
    --checkpoint "$baseline_checkpoint" --voc-root "$voc_root" --id-list "$cam_list" \
    --output-dir "$comparison_dir/shared_token_role_diagnostics" \
    --token-role-specialization shared --resolution 448 \
    --max-images "$diagnostic_images"
CUDA_VISIBLE_DEVICES="$gpu_id" python tools/benchmark_mctformerplus.py \
    --checkpoint "$baseline_checkpoint" --mode vanilla \
    --token-role-specialization shared \
    --output "$comparison_dir/shared_benchmark.json" --input-size 448

run_specs=()
for variant in norm norm_qkv; do
    run_id="${queue_id}-mctformerplus-voc-role-${variant}-s${seed}-${commit}"
    TOKEN_ROLE_GPU_ID="$gpu_id" TOKEN_ROLE_SEED="$seed" \
        TOKEN_ROLE_RUN_ID="$run_id" \
        TOKEN_ROLE_DIAGNOSTIC_IMAGES="$diagnostic_images" \
        bash experiments/ablations/run_token_role_voc_variant.sh "$variant"
    run_specs+=(--run "$variant=$repo_root/results/mctformerplus/voc/$run_id")
done

python tools/collect_token_role_pilot.py \
    --baseline-run "$baseline_run" \
    --baseline-role-metrics "$comparison_dir/shared_token_role_diagnostics/metrics.json" \
    --baseline-benchmark "$comparison_dir/shared_benchmark.json" \
    "${run_specs[@]}" --output-dir "$comparison_dir"

printf "QUEUE_COMPLETE finished=%s comparison=%s\n" \
    "$(date --iso-8601=seconds)" "$comparison_dir"
