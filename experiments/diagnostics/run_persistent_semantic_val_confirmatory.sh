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

checkpoint="$repo_root/results/mctformerplus/voc/20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3/mctformerplus_final.pth"
baseline_metrics="$repo_root/results/mctformerplus/voc/20260827-190911-mctformerplus-voc-bcss-e0-s0-4147fc3/raw_cam_diagnostics/metrics.json"
checkpoint_sha256="41ac9ce47f6a22875cba32edb92c31c150e804ae5ae19824c2585e4e3cda7a2a"
voc_root="$repo_root/data/VOCdevkit/VOC2012"
id_list="$voc_root/ImageLists/val_id.txt"
commit=$(git rev-parse --short=7 HEAD)
timestamp=$(date +%Y%m%d-%H%M%S)
run_id=${1:-"${timestamp}-persistent-semantic-val-confirmatory-${commit}"}
run_dir="$repo_root/results/persistent_semantic/voc/$run_id"

for required in "$checkpoint" "$baseline_metrics" "$id_list"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 2
    fi
done
if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite run directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/pipeline.log") 2>&1

printf 'run_id=%s\ncommit=%s\nbranch=%s\nstarted=%s\n' \
    "$run_id" "$(git rev-parse HEAD)" "$(git branch --show-current)" \
    "$(date --iso-8601=seconds)" > "$run_dir/run_manifest.txt"
printf '%s\n' \
    '{' \
    '  "split": "val_id",' \
    '  "selection": "fixed_before_observation",' \
    '  "primary_layer_zero_based": 11,' \
    '  "primary_head_reduction": "mean_of_all_six_heads",' \
    '  "secondary_head_zero_based": 5,' \
    '  "primary_relation": "pc_all",' \
    '  "majority_control": "paired macro-image accuracy difference",' \
    '  "primary_semantic_threshold": 0.5,' \
    '  "sensitivity_thresholds": [0.05, 0.10, 0.25, 0.50],' \
    '  "bootstrap_resamples": 10000,' \
    '  "permutation_resamples": 10000,' \
    '  "permutation_seed": 2027' \
    '}' > "$run_dir/frozen_protocol.json"
printf '%s\n' \
    "python tools/analyze_patch_to_class.py --checkpoint $checkpoint --baseline-metrics $baseline_metrics --expected-checkpoint-sha256 $checkpoint_sha256 --voc-root $voc_root --id-list $id_list --split val_id --output-dir $run_dir/analysis --run-id $run_id --resolutions 224,320,448,512 --semantic-resolution 448 --confirmatory-layer 11 --diagnostic-head 5 --cam-threshold 0.5 --semantic-threshold 0.5 --semantic-thresholds 0.05,0.10,0.25,0.50 --bootstrap-resamples 10000 --permutation-resamples 10000 --permutation-seed 2027 --device cuda" \
    > "$run_dir/command.txt"
git status --short --branch > "$run_dir/git_status.txt"
git log -1 --format=fuller > "$run_dir/git_commit.txt"
conda list --explicit > "$run_dir/conda_explicit.txt"
python -m pip freeze > "$run_dir/pip_freeze.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
    --format=csv,noheader > "$run_dir/gpu_start.csv"
sha256sum "$checkpoint" "$baseline_metrics" "$id_list" > "$run_dir/input_sha256.txt"
df -h "$repo_root" > "$run_dir/disk_start.txt"

analysis_args=(
    --checkpoint "$checkpoint"
    --baseline-metrics "$baseline_metrics"
    --expected-checkpoint-sha256 "$checkpoint_sha256"
    --voc-root "$voc_root"
    --id-list "$id_list"
    --split val_id
    --output-dir "$run_dir/analysis"
    --run-id "$run_id"
    --resolutions 224,320,448,512
    --semantic-resolution 448
    --confirmatory-layer 11
    --diagnostic-head 5
    --cam-threshold 0.5
    --semantic-threshold 0.5
    --semantic-thresholds 0.05,0.10,0.25,0.50
    --sample-dumps "${PSL_SAMPLE_DUMPS:-12}"
    --raw-dump-images "${PSL_RAW_DUMP_IMAGES:-2}"
    --bootstrap-resamples "${PSL_BOOTSTRAP_RESAMPLES:-10000}"
    --permutation-resamples "${PSL_PERMUTATION_RESAMPLES:-10000}"
    --permutation-seed 2027
    --device "${PSL_DEVICE:-cuda}"
)
if [[ -n "${PSL_MAX_IMAGES:-}" ]]; then
    analysis_args+=(--max-images "$PSL_MAX_IMAGES")
fi

printf 'STAGE=val_confirmatory started=%s\n' "$(date --iso-8601=seconds)"
python tools/analyze_patch_to_class.py "${analysis_args[@]}"
python tools/review_patch_to_class_result.py --analysis-dir "$run_dir/analysis"
printf '{"complete":true,"finished":"%s","analysis":"%s"}\n' \
    "$(date --iso-8601=seconds)" "$run_dir/analysis" > "$run_dir/completion.json"
printf 'QUEUE_COMPLETE run_dir=%s finished=%s\n' \
    "$run_dir" "$(date --iso-8601=seconds)"
