#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi

variant=${1:?Usage: run_bcss_voc_variant.sh VARIANT}
case "$variant" in
    e0|e1|e2|e4|e5|e6) ;;
    *) echo "Unsupported BCSS screening variant: $variant" >&2; exit 2 ;;
esac

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

gpu_id=${BCSS_GPU_ID:-0}
seed=${BCSS_SEED:-0}
fixed_threshold=${BCSS_FIXED_THRESHOLD:-0.45}
tau=${BCSS_TAU:-0.5}
beta=${BCSS_BETA:-0.5}
lambda_fg=${BCSS_LAMBDA_FG:-0.5}
lambda_bg=${BCSS_LAMBDA_BG:-0.1}
commit=$(git rev-parse --short HEAD)
full_commit=$(git rev-parse HEAD)
run_id=${BCSS_RUN_ID:-"$(date +%Y%m%d)-mctformerplus-voc-bcss-${variant}-s${seed}-${commit}"}
run_dir="$repo_root/results/mctformerplus/voc/$run_id"
voc_root="$repo_root/data/VOCdevkit/VOC2012"
train_list="$voc_root/ImageLists/train_aug_id.txt"
val_list="$voc_root/ImageLists/val_id.txt"
cam_list="$voc_root/ImageLists/train_id.txt"
pretrain="$HOME/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth"

if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/pipeline.log") 2>&1

printf "run_id=%s\nrun_dir=%s\nvariant=%s\n" "$run_id" "$run_dir" "$variant"
printf "BCSS_GPU_ID=%q BCSS_SEED=%q BCSS_FIXED_THRESHOLD=%q BCSS_TAU=%q BCSS_BETA=%q bash %q %q\n" \
    "$gpu_id" "$seed" "$fixed_threshold" "$tau" "$beta" \
    "experiments/ablations/run_bcss_voc_variant.sh" "$variant" > "$run_dir/command.txt"
printf '{"commit":"%s","branch":"%s","dirty":false,"official_mctformer_commit":"%s"}\n' \
    "$full_commit" "$(git branch --show-current)" \
    "0acc27ada87a5582053efb14648442d8644168aa" > "$run_dir/git_state.json"
printf '{"variant":"%s","seed":%s,"epochs":45,"input_size":448,"tau":%s,"beta":%s,"lambda_fg":%s,"lambda_bg":%s,"background_slots":1,"fixed_background_threshold":%s,"attention_normalization":"vanilla","warmup":{"epochs_0_2":"beta=0,refinement=0,tau=1","epochs_3_8":"linear ramp"}}\n' \
    "$variant" "$seed" "$tau" "$beta" "$lambda_fg" "$lambda_bg" \
    "$fixed_threshold" > "$run_dir/config.json"

{
    python --version
    python -c "import torch, torchvision, timm; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('cuda', torch.version.cuda)"
    printf "conda_env=%s\n" "$CONDA_DEFAULT_ENV"
} > "$run_dir/environment.txt" 2>&1
python -m pip freeze > "$run_dir/pip_freeze.txt"
conda list --explicit > "$run_dir/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
    > "$run_dir/hardware.txt"
{
    sha256sum "$train_list" "$val_list" "$cam_list"
    sha256sum "$voc_root/ImageLabel/cls_labels.npy"
    printf "train_aug_ids=%s\n" "$(awk 'NF {n++} END {print n+0}' "$train_list")"
    printf "val_ids=%s\n" "$(awk 'NF {n++} END {print n+0}' "$val_list")"
    printf "cam_eval_ids=%s\n" "$(awk 'NF {n++} END {print n+0}' "$cam_list")"
} > "$run_dir/dataset_manifest.txt"
sha256sum "$pretrain" > "$run_dir/pretrained_manifest.txt"

printf "STAGE=train started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" python train_model_v2.py \
    --dataset VOC12 --model mctformerplus --voc12_root "$voc_root" \
    --train_list "$train_list" --val_list "$val_list" \
    --work_space "$run_dir" --input-size 448 --epochs 45 --batch_size 32 \
    --seed "$seed" --lr 5e-4 --min-lr 1e-5 --num_workers 10 \
    --attention-normalization vanilla --bcss-variant "$variant" \
    --bcss-num-background-slots 1 --bcss-tau "$tau" --bcss-beta "$beta" \
    --bcss-lambda-fg "$lambda_fg" --bcss-lambda-bg "$lambda_bg"

checkpoint="$run_dir/mctformerplus_final.pth"
sha256sum "$checkpoint" > "$run_dir/checkpoint_manifest.txt"

printf "STAGE=cam started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" python make_cam.py \
    --dataset VOC12 --model mctformerplus --voc12_root "$voc_root" \
    --work_space "$run_dir" --cam_out_dir cam_train --train_list "$cam_list" \
    --input_size 448 --scales 1.0,0.75,1.25 --checkpoint "$checkpoint" \
    --attention-normalization vanilla --bcss-variant "$variant" \
    --bcss-num-background-slots 1 --bcss-tau "$tau" --bcss-beta "$beta"

printf "STAGE=raw_cam_metrics started=%s\n" "$(date --iso-8601=seconds)"
python tools/collect_cam_metrics.py \
    --cam-dir "$run_dir/cam_train" --voc-root "$voc_root" \
    --id-list "$cam_list" --threshold "$fixed_threshold" \
    --output-dir "$run_dir/raw_cam_diagnostics"

printf "STAGE=bcss_diagnostics started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" bash experiments/diagnostics/run_bcss_voc_diagnostics.sh \
    "$run_dir" "$variant"

printf "PIPELINE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
