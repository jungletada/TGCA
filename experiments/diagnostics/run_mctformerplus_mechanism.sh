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

tgca_gpu_id=${TGCA_GPU_ID:-0}
tgca_max_images=${TGCA_MAX_IMAGES:-0}
tgca_commit=$(git rev-parse --short HEAD)
tgca_run_id=${TGCA_RUN_ID:-"$(date +%Y%m%d)-mctformerplus-voc-vanilla-scale-diagnostic-${tgca_commit}"}
tgca_run_dir="$tgca_repo_root/results/mctformerplus/voc/$tgca_run_id"
tgca_voc_root="$tgca_repo_root/data/VOCdevkit/VOC2012"
tgca_id_list="$tgca_voc_root/ImageLists/train_id.txt"
tgca_checkpoint="$tgca_repo_root/results/mctformerplus/voc/20260825-mctformerplus-voc-vanilla-s0-63d8877/mctformerplus_final.pth"

if [[ -e "$tgca_run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $tgca_run_dir" >&2
    exit 2
fi
test -f "$tgca_checkpoint"
mkdir -p "$tgca_run_dir"
exec > >(tee -a "$tgca_run_dir/pipeline.log") 2>&1

printf "TGCA_GPU_ID=%q TGCA_MAX_IMAGES=%q bash %q\n" \
    "$tgca_gpu_id" "$tgca_max_images" \
    "experiments/diagnostics/run_mctformerplus_mechanism.sh" \
    > "$tgca_run_dir/command.txt"
printf '{"commit":"%s","branch":"%s","dirty":false}\n' \
    "$(git rev-parse HEAD)" "$(git branch --show-current)" \
    > "$tgca_run_dir/git_state.json"
sha256sum "$tgca_checkpoint" > "$tgca_run_dir/checkpoint_manifest.txt"
sha256sum "$tgca_id_list" "$tgca_voc_root/ImageLabel/cls_labels.npy" \
    > "$tgca_run_dir/dataset_manifest.txt"
{
    python --version
    python -c "import torch, torchvision, timm; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('cuda', torch.version.cuda)"
    printf "conda_env=%s\n" "$CONDA_DEFAULT_ENV"
} > "$tgca_run_dir/environment.txt" 2>&1
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader > "$tgca_run_dir/hardware.txt"

printf "STAGE=attention_scale_diagnostic started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu_id" python tools/analyze_attention_groups.py \
    --checkpoint "$tgca_checkpoint" \
    --voc-root "$tgca_voc_root" \
    --id-list "$tgca_id_list" \
    --output-dir "$tgca_run_dir" \
    --run-id "$tgca_run_id" \
    --mode vanilla \
    --gamma 1.0 \
    --resolutions 224,320,448,512 \
    --max-images "$tgca_max_images"
printf "PIPELINE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
