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

variant=${1:?Usage: run_psl_voc_variant.sh VARIANT}
case "$variant" in
    read_only|write_only|read_write) ;;
    *) echo "Unsupported Phase 2 variant: $variant" >&2; exit 2 ;;
esac

gpu_id=${PSL_GPU_ID:-0}
seed=${PSL_SEED:-0}
fixed_threshold=${PSL_FIXED_THRESHOLD:-0.45}
interaction_layers=11
relation_dim=384
commit=$(git rev-parse --short=7 HEAD)
full_commit=$(git rev-parse HEAD)
run_id=${PSL_RUN_ID:-"$(date +%Y%m%d)-mctformerplus-voc-psl-${variant}-s${seed}-${commit}"}
run_dir="$repo_root/results/persistent_semantic/phase2/voc/$run_id"
voc_root="$repo_root/data/VOCdevkit/VOC2012"
train_list="$voc_root/ImageLists/train_aug_id.txt"
val_list="$voc_root/ImageLists/val_id.txt"
cam_list="$voc_root/ImageLists/train_id.txt"
pretrain="$HOME/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth"

for required in "$train_list" "$val_list" "$cam_list" "$pretrain"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 2
    fi
done
if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/pipeline.log") 2>&1

printf 'run_id=%s\nrun_dir=%s\nvariant=%s\n' "$run_id" "$run_dir" "$variant"
printf 'PSL_GPU_ID=%q PSL_SEED=%q PSL_RUN_ID=%q PSL_FIXED_THRESHOLD=%q bash %q %q\n' \
    "$gpu_id" "$seed" "$run_id" "$fixed_threshold" \
    "experiments/ablations/run_psl_voc_variant.sh" "$variant" \
    > "$run_dir/command.txt"
printf '{"commit":"%s","branch":"%s","dirty":false,"official_mctformer_commit":"%s"}\n' \
    "$full_commit" "$(git branch --show-current)" \
    "0acc27ada87a5582053efb14648442d8644168aa" > "$run_dir/git_state.json"
printf '{"phase":2,"variant":"%s","seed":%s,"epochs":45,"input_size":448,"semantic_dim":384,"patch_dim":384,"relation_dim":384,"interaction_layers_zero_based":[11],"relation":"shared","ordering":"read_then_write","background":"one_static_semantic_latent","write_gate_initialization":0.0,"fixed_background_threshold":%s,"attention_normalization":"vanilla","bcss_variant":"e0"}\n' \
    "$variant" "$seed" "$fixed_threshold" > "$run_dir/config.json"

{
    python --version
    python -c "import torch, torchvision, timm; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('cuda', torch.version.cuda)"
    printf 'conda_env=%s\n' "$CONDA_DEFAULT_ENV"
} > "$run_dir/environment.txt" 2>&1
python -m pip freeze > "$run_dir/pip_freeze.txt"
conda list --explicit > "$run_dir/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader \
    > "$run_dir/hardware.txt"
{
    sha256sum "$train_list" "$val_list" "$cam_list"
    sha256sum "$voc_root/ImageLabel/cls_labels.npy"
    printf 'train_aug_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$train_list")"
    printf 'val_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$val_list")"
    printf 'cam_eval_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$cam_list")"
} > "$run_dir/dataset_manifest.txt"
sha256sum "$pretrain" > "$run_dir/pretrained_manifest.txt"

printf 'STAGE=train started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" python train_model_v2.py \
    --dataset VOC12 --model mctformerplus --voc12_root "$voc_root" \
    --train_list "$train_list" --val_list "$val_list" \
    --work_space "$run_dir" --input-size 448 --epochs 45 --batch_size 32 \
    --seed "$seed" --lr 5e-4 --min-lr 1e-5 --num_workers 10 \
    --attention-normalization vanilla --bcss-variant e0 \
    --psl-variant "$variant" --psl-interaction-layers "$interaction_layers" \
    --psl-relation-dim "$relation_dim" --psl-num-background-latents 1

checkpoint="$run_dir/mctformerplus_final.pth"
sha256sum "$checkpoint" > "$run_dir/checkpoint_manifest.txt"

printf 'STAGE=cam started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" python make_cam.py \
    --dataset VOC12 --model mctformerplus --voc12_root "$voc_root" \
    --work_space "$run_dir" --cam_out_dir cam_train --train_list "$cam_list" \
    --input_size 448 --scales 1.0,0.75,1.25 --checkpoint "$checkpoint" \
    --attention-normalization vanilla --bcss-variant e0 \
    --psl-variant "$variant" --psl-interaction-layers "$interaction_layers" \
    --psl-relation-dim "$relation_dim" --psl-num-background-latents 1

printf 'STAGE=raw_cam_metrics started=%s\n' "$(date --iso-8601=seconds)"
python tools/collect_cam_metrics.py \
    --cam-dir "$run_dir/cam_train" --voc-root "$voc_root" \
    --id-list "$cam_list" --threshold "$fixed_threshold" \
    --output-dir "$run_dir/raw_cam_diagnostics"

printf 'STAGE=relation_diagnostics started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$gpu_id" python tools/analyze_psl_relations.py \
    --checkpoint "$checkpoint" --voc-root "$voc_root" --id-list "$val_list" \
    --output-dir "$run_dir/relation_diagnostics" \
    --max-images "${PSL_DIAGNOSTIC_IMAGES:-200}" --input-size 448 --device cuda

printf '{"complete":true,"variant":"%s","finished":"%s"}\n' \
    "$variant" "$(date --iso-8601=seconds)" > "$run_dir/completion.json"
printf 'PIPELINE_COMPLETE variant=%s finished=%s\n' \
    "$variant" "$(date --iso-8601=seconds)"
