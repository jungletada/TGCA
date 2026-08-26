#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate the tgca-repro Conda environment before running this script." >&2
    exit 2
fi

tgca_mode=${1:?Usage: run_mctformerplus_voc_mode.sh MODE}
case "$tgca_mode" in
    vanilla|split_11|split_05|tgca|tgca_bias) ;;
    *) echo "Unsupported core ablation mode: $tgca_mode" >&2; exit 2 ;;
esac

tgca_gpu_id=${TGCA_GPU_ID:-0}
tgca_seed=${TGCA_SEED:-0}
tgca_fixed_threshold=${TGCA_FIXED_THRESHOLD:-0.45}
tgca_commit=$(git rev-parse --short HEAD)
tgca_run_id=${TGCA_RUN_ID:-"$(date +%Y%m%d)-mctformerplus-voc-${tgca_mode}-s${tgca_seed}-${tgca_commit}"}
tgca_run_dir="$tgca_repo_root/results/mctformerplus/voc/$tgca_run_id"
tgca_voc_root="$tgca_repo_root/data/VOCdevkit/VOC2012"
tgca_train_list="$tgca_voc_root/ImageLists/train_aug_id.txt"
tgca_val_list="$tgca_voc_root/ImageLists/val_id.txt"
tgca_cam_list="$tgca_voc_root/ImageLists/train_id.txt"
tgca_pretrain="$HOME/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth"

if [[ -e "$tgca_run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $tgca_run_dir" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

mkdir -p "$tgca_run_dir"
exec > >(tee -a "$tgca_run_dir/pipeline.log") 2>&1

tgca_full_commit=$(git rev-parse HEAD)
tgca_branch=$(git branch --show-current)
printf "run_id=%s\nrun_dir=%s\nmode=%s\n" "$tgca_run_id" "$tgca_run_dir" "$tgca_mode"
printf "TGCA_GPU_ID=%q TGCA_SEED=%q TGCA_FIXED_THRESHOLD=%q bash %q %q\n" \
    "$tgca_gpu_id" "$tgca_seed" "$tgca_fixed_threshold" \
    "experiments/ablations/run_mctformerplus_voc_mode.sh" "$tgca_mode" \
    > "$tgca_run_dir/command.txt"
printf '{"commit":"%s","branch":"%s","dirty":false,"official_mctformer_commit":"%s"}\n' \
    "$tgca_full_commit" "$tgca_branch" \
    "0acc27ada87a5582053efb14648442d8644168aa" \
    > "$tgca_run_dir/git_state.json"
printf '{"mode":"%s","gamma":1.0,"relation_bias":%s,"fixed_background_threshold":%s}\n' \
    "$tgca_mode" "$([[ "$tgca_mode" == "tgca_bias" ]] && echo true || echo false)" \
    "$tgca_fixed_threshold" > "$tgca_run_dir/config.json"

{
    python --version
    python -c "import torch, torchvision, timm; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('cuda', torch.version.cuda)"
    printf "conda_env=%s\n" "$CONDA_DEFAULT_ENV"
} > "$tgca_run_dir/environment.txt" 2>&1
python -m pip freeze > "$tgca_run_dir/pip_freeze.txt"
conda list --explicit > "$tgca_run_dir/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader > "$tgca_run_dir/hardware.txt"
{
    sha256sum "$tgca_train_list" "$tgca_val_list" "$tgca_cam_list"
    sha256sum "$tgca_voc_root/ImageLabel/cls_labels.npy"
    printf "train_aug_ids=%s\n" "$(awk 'NF {n++} END {print n+0}' "$tgca_train_list")"
    printf "val_ids=%s\n" "$(awk 'NF {n++} END {print n+0}' "$tgca_val_list")"
    printf "cam_eval_ids=%s\n" "$(awk 'NF {n++} END {print n+0}' "$tgca_cam_list")"
} > "$tgca_run_dir/dataset_manifest.txt"
sha256sum "$tgca_pretrain" > "$tgca_run_dir/pretrained_manifest.txt"

printf "STAGE=train started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu_id" python train_model_v2.py \
    --dataset VOC12 \
    --model mctformerplus \
    --voc12_root "$tgca_voc_root" \
    --train_list "$tgca_train_list" \
    --val_list "$tgca_val_list" \
    --work_space "$tgca_run_dir" \
    --input-size 448 \
    --attention-normalization "$tgca_mode" \
    --attention-gamma 1.0 \
    --epochs 45 \
    --batch_size 32 \
    --seed "$tgca_seed" \
    --lr 5e-4 \
    --min-lr 1e-5 \
    --num_workers 10

tgca_checkpoint="$tgca_run_dir/mctformerplus_final.pth"
sha256sum "$tgca_checkpoint" > "$tgca_run_dir/checkpoint_manifest.txt"

printf "STAGE=cam started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu_id" python make_cam.py \
    --dataset VOC12 \
    --model mctformerplus \
    --voc12_root "$tgca_voc_root" \
    --work_space "$tgca_run_dir" \
    --cam_out_dir cam_train \
    --train_list "$tgca_cam_list" \
    --input_size 448 \
    --attention-normalization "$tgca_mode" \
    --attention-gamma 1.0 \
    --scales 1.0,0.75,1.25 \
    --checkpoint "$tgca_checkpoint"

printf "STAGE=raw_cam_eval_fixed started=%s threshold=%s\n" \
    "$(date --iso-8601=seconds)" "$tgca_fixed_threshold"
CUDA_VISIBLE_DEVICES="$tgca_gpu_id" python eval_cam_crf.py \
    --dataset VOC12 \
    --voc12_root "$tgca_voc_root" \
    --work_space "$tgca_run_dir" \
    --eval_cam_dir cam_train \
    --id_list "$tgca_cam_list" \
    --threshold "$tgca_fixed_threshold"

tgca_eval_log=$(find "$tgca_run_dir/log_dir" -maxdepth 1 -type f \
    -name 'eval-cam-crf-train-*.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
python - "$tgca_eval_log" "$tgca_run_dir/metrics.json" "$tgca_mode" \
    "$tgca_seed" "$tgca_fixed_threshold" "$tgca_full_commit" "$tgca_checkpoint" <<'PY'
import hashlib
import json
import re
import sys

log_path, output_path, mode, seed, threshold, commit, checkpoint_path = sys.argv[1:]
text = open(log_path, encoding="utf-8").read()
matches = re.findall(r"mIoU for all images: ([0-9.]+)%", text)
if not matches:
    raise RuntimeError(f"Raw CAM mIoU not found in {log_path}")
digest = hashlib.sha256()
with open(checkpoint_path, "rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
metrics = {
    "dataset": "PASCAL VOC 2012 train",
    "host": "MCTformer+",
    "normalization": mode,
    "gamma": 1.0,
    "metric": "raw_cam_miou_fixed_threshold",
    "raw_cam_miou_percent": float(matches[-1]),
    "background_threshold": float(threshold),
    "threshold_selected_by": "vanilla_seed0",
    "num_images": 1464,
    "seed": int(seed),
    "input_size": 448,
    "cam_scales": [1.0, 0.75, 1.25],
    "epochs": 45,
    "checkpoint": checkpoint_path,
    "checkpoint_sha256": digest.hexdigest(),
    "commit": commit,
}
with open(output_path, "w", encoding="utf-8") as stream:
    json.dump(metrics, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps(metrics, sort_keys=True))
PY

printf "STAGE=attention_scale_diagnostic started=%s\n" "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu_id" python tools/analyze_attention_groups.py \
    --checkpoint "$tgca_checkpoint" \
    --voc-root "$tgca_voc_root" \
    --id-list "$tgca_cam_list" \
    --output-dir "$tgca_run_dir/attention_diagnostics" \
    --run-id "$tgca_run_id" \
    --mode "$tgca_mode" \
    --gamma 1.0 \
    --resolutions 224,320,448,512

printf "PIPELINE_COMPLETE finished=%s\n" "$(date --iso-8601=seconds)"
