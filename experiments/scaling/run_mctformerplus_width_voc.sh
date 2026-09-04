#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

tgca_variant=""
tgca_seed=0
tgca_gpu=0
tgca_stage=all
tgca_micro_batch=""
tgca_accum_iter=""
tgca_val_batch_size=""
tgca_run_id=""
tgca_small_run=${TGCA_SMALL_RUN_DIR:-}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant) tgca_variant=${2:?}; shift 2 ;;
        --seed) tgca_seed=${2:?}; shift 2 ;;
        --gpu) tgca_gpu=${2:?}; shift 2 ;;
        --stage) tgca_stage=${2:?}; shift 2 ;;
        --micro-batch) tgca_micro_batch=${2:?}; shift 2 ;;
        --accum-iter) tgca_accum_iter=${2:?}; shift 2 ;;
        --val-batch-size) tgca_val_batch_size=${2:?}; shift 2 ;;
        --run-id) tgca_run_id=${2:?}; shift 2 ;;
        --small-run) tgca_small_run=${2:?}; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

case "$tgca_variant" in
    tiny|small|base) ;;
    *) echo "--variant must be tiny, small, or base" >&2; exit 2 ;;
esac
case "$tgca_stage" in
    all|smoke|reanalysis) ;;
    *) echo "--stage must be all, smoke, or reanalysis" >&2; exit 2 ;;
esac
if [[ "$tgca_variant" == small && "$tgca_stage" != reanalysis ]]; then
    echo "Small is immutable and may only use --stage reanalysis." >&2
    exit 2
fi
if [[ "$tgca_variant" != small && "$tgca_stage" == reanalysis ]]; then
    echo "--stage reanalysis is reserved for the existing Small run." >&2
    exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != tgca-repro ]]; then
    echo "Activate the tgca-repro Conda environment before running." >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != main ]]; then
    echo "Scientific width runs must start from main." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

tgca_model_name=mctformerplus
tgca_pretrain_name=deit_small_patch16_224-cd65a155.pth
tgca_pretrain_hash=cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f
case "$tgca_variant" in
    tiny)
        tgca_model_name=mctformerplus_tiny
        tgca_pretrain_name=deit_tiny_patch16_224-a1311bcf.pth
        tgca_pretrain_hash=a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97
        : "${tgca_micro_batch:=32}"
        : "${tgca_accum_iter:=1}"
        : "${tgca_val_batch_size:=32}"
        ;;
    small)
        : "${tgca_val_batch_size:=16}"
        ;;
    base)
        tgca_model_name=mctformerplus_base
        tgca_pretrain_name=deit_base_patch16_224-b5f2ef4d.pth
        tgca_pretrain_hash=b5f2ef4d686982dcdab24fe285fd08fff40db01550d8d4833167a73dd85ca7a8
        if [[ -z "$tgca_micro_batch" || -z "$tgca_accum_iter" ]]; then
            tgca_capacity="$tgca_repo_root/results/mctformerplus_width_scaling/voc/references/base_capacity_probe.json"
            if [[ ! -f "$tgca_capacity" ]]; then
                echo "Run the Base capacity probe or provide --micro-batch/--accum-iter." >&2
                exit 2
            fi
            read -r tgca_micro_batch tgca_accum_iter < <(
                python - "$tgca_capacity" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))['selected']
print(value['micro_batch_size'], value['accum_iter'])
PY
            )
        fi
        : "${tgca_val_batch_size:=$tgca_micro_batch}"
        ;;
esac

if [[ "$tgca_variant" != small ]]; then
    if (( tgca_micro_batch * tgca_accum_iter != 32 )); then
        echo "micro_batch * accum_iter must equal 32." >&2
        exit 2
    fi
fi

tgca_voc_root="$tgca_repo_root/data/VOCdevkit/VOC2012"
tgca_train_aug="$tgca_voc_root/ImageLists/train_aug_id.txt"
tgca_train_cam="$tgca_voc_root/ImageLists/train_id.txt"
tgca_val_list="$tgca_voc_root/ImageLists/val_id.txt"
tgca_pretrain="/home/peng/.cache/torch/hub/checkpoints/$tgca_pretrain_name"
for tgca_required in "$tgca_voc_root" "$tgca_train_aug" "$tgca_train_cam" \
        "$tgca_val_list" "$tgca_pretrain"; do
    if [[ ! -e "$tgca_required" ]]; then
        echo "Required input is absent: $tgca_required" >&2
        exit 2
    fi
done
if [[ "$(sha256sum "$tgca_pretrain" | awk '{print $1}')" != "$tgca_pretrain_hash" ]]; then
    echo "Official pretrained hash mismatch: $tgca_pretrain" >&2
    exit 2
fi

tgca_commit=$(git rev-parse HEAD)
tgca_short_commit=$(git rev-parse --short HEAD)
if [[ -z "$tgca_run_id" ]]; then
    if [[ "$tgca_stage" == smoke ]]; then
        tgca_run_id="$(date +%Y%m%d-%H%M%S)-mctformerplus-${tgca_variant}-voc-smoke-s${tgca_seed}-${tgca_short_commit}"
    elif [[ "$tgca_stage" == reanalysis ]]; then
        tgca_run_id="$(date +%Y%m%d-%H%M%S)-mctformerplus-small-voc-reanalysis-${tgca_short_commit}"
    else
        tgca_run_id="$(date +%Y%m%d)-mctformerplus-${tgca_variant}-voc-vanilla-s${tgca_seed}-${tgca_short_commit}"
    fi
fi
if [[ "$tgca_stage" == reanalysis ]]; then
    tgca_run_dir="$tgca_repo_root/results/mctformerplus_width_scaling/voc/small_reanalysis/$tgca_run_id"
else
    tgca_run_dir="$tgca_repo_root/results/mctformerplus_width_scaling/voc/$tgca_variant/$tgca_run_id"
fi
if [[ -e "$tgca_run_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $tgca_run_dir" >&2
    exit 2
fi
mkdir -p "$tgca_run_dir"
exec > >(tee -a "$tgca_run_dir/pipeline.log") 2>&1

printf '%s\n' "run_id=$tgca_run_id" "run_dir=$tgca_run_dir" \
    "variant=$tgca_variant" "stage=$tgca_stage" "commit=$tgca_commit"
printf 'bash experiments/scaling/run_mctformerplus_width_voc.sh --variant %q --seed %q --gpu %q --stage %q --micro-batch %q --accum-iter %q --val-batch-size %q --run-id %q --small-run %q\n' \
    "$tgca_variant" "$tgca_seed" "$tgca_gpu" "$tgca_stage" \
    "$tgca_micro_batch" "$tgca_accum_iter" "$tgca_val_batch_size" \
    "$tgca_run_id" "$tgca_small_run" > "$tgca_run_dir/command.txt"
printf '{"commit":"%s","branch":"main","tracked_dirty":false}\n' \
    "$tgca_commit" > "$tgca_run_dir/git_state.json"
printf '{"variant":"%s","model_name":"%s","seed":%s,"epochs":%s,"input_size":448,"micro_batch_size":%s,"accum_iter":%s,"effective_batch_size":32,"optimizer":"adamw","nominal_lr":0.0005,"optimizer_lr":0.00003125,"minimum_lr":0.00001,"weight_decay":0.05,"scheduler":"cosine","warmup_epochs":5,"drop":0.0,"drop_path":0.1,"train_interpolation":"bicubic","attention_normalization":"vanilla","attention_gamma":1.0,"bcss_variant":"e0","psl_variant":"baseline","cti_bgt":false,"cam_scales":[1.0,0.75,1.25],"cam_class_to_patch_layers":3,"cam_patch_to_patch_layers":12,"checkpoint_policy":"final","stage":"%s"}\n' \
    "$tgca_variant" "$tgca_model_name" "$tgca_seed" \
    "$([[ "$tgca_stage" == smoke ]] && echo 1 || echo 45)" \
    "${tgca_micro_batch:-32}" "${tgca_accum_iter:-1}" "$tgca_stage" \
    > "$tgca_run_dir/config.json"
{
    python --version
    python -c "import torch, torchvision, timm; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('cuda', torch.version.cuda)"
    printf 'conda_env=%s\n' "$CONDA_DEFAULT_ENV"
} > "$tgca_run_dir/environment.txt" 2>&1
python -m pip freeze > "$tgca_run_dir/pip_freeze.txt"
conda list --explicit > "$tgca_run_dir/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader > "$tgca_run_dir/hardware.txt"
{
    sha256sum "$tgca_train_aug" "$tgca_train_cam" "$tgca_val_list"
    sha256sum "$tgca_voc_root/ImageLabel/cls_labels.npy"
    printf 'train_aug_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_train_aug")"
    printf 'train_cam_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_train_cam")"
    printf 'val_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_val_list")"
} > "$tgca_run_dir/dataset_manifest.txt"
sha256sum "$tgca_pretrain" > "$tgca_run_dir/pretrained_manifest.txt"

tgca_epochs=45
tgca_bootstrap=10000
tgca_semantic_bootstrap=5000
tgca_class_batch=$tgca_val_batch_size
tgca_semantic_batch=8
tgca_benchmark_warmup=20
tgca_benchmark_iterations=100
tgca_training_list=$tgca_train_aug
tgca_cam_train_list=$tgca_train_cam
tgca_cam_val_list=$tgca_val_list
if [[ "$tgca_stage" == smoke ]]; then
    tgca_epochs=1
    tgca_bootstrap=0
    tgca_semantic_bootstrap=20
    tgca_benchmark_warmup=1
    tgca_benchmark_iterations=2
    mkdir "$tgca_run_dir/smoke_lists"
    head -n 128 "$tgca_train_cam" > "$tgca_run_dir/smoke_lists/train_id.txt"
    head -n 4 "$tgca_val_list" > "$tgca_run_dir/smoke_lists/val_id.txt"
    tgca_training_list="$tgca_run_dir/smoke_lists/train_id.txt"
    tgca_cam_train_list="$tgca_run_dir/smoke_lists/val_id.txt"
    tgca_cam_val_list="$tgca_run_dir/smoke_lists/val_id.txt"
    tgca_semantic_batch=2
fi

if [[ "$tgca_stage" == reanalysis ]]; then
    if [[ -z "$tgca_small_run" || ! -d "$tgca_small_run" ]]; then
        echo "Set TGCA_SMALL_RUN_DIR or pass --small-run for Small reanalysis." >&2
        exit 2
    fi
    tgca_checkpoint="$tgca_small_run/mctformerplus_final.pth"
    if [[ ! -f "$tgca_checkpoint" ]]; then
        echo "Canonical Small final checkpoint is absent." >&2
        exit 2
    fi
    printf '{"path":"%s","sha256":"%s","mode":"read_only"}\n' \
        "$(realpath "$tgca_small_run")" \
        "$(sha256sum "$tgca_checkpoint" | awk '{print $1}')" \
        > "$tgca_run_dir/small_source_pointer.json"
    python - "$tgca_small_run" "$tgca_run_dir/training_runtime.json" <<'PY'
import json, pathlib, sys
source, output = map(pathlib.Path, sys.argv[1:])
payload = {
    'available': False,
    'reason': 'canonical Small training was completed before this unified runtime schema',
    'comparability_note': 'historical Small training speed is not compared with current-host Tiny/Base timing',
    'source_pipeline_log': str((source/'pipeline.log').resolve()),
    'source_hardware': str((source/'hardware.txt').resolve()),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True)+'\n')
PY
else
    printf 'STAGE=train started=%s\n' "$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$tgca_gpu" python train_model_v2.py \
        --dataset VOC12 --model "$tgca_model_name" \
        --voc12_root "$tgca_voc_root" \
        --train_list "$tgca_training_list" --val_list "$tgca_cam_val_list" \
        --work_space "$tgca_run_dir" --input-size 448 \
        --epochs "$tgca_epochs" --batch_size "$tgca_micro_batch" \
        --accum-iter "$tgca_accum_iter" \
        --val-batch-size "$tgca_val_batch_size" --seed "$tgca_seed" \
        --opt adamw --sched cosine --warmup-epochs 5 \
        --lr 5e-4 --min-lr 1e-5 --weight-decay 0.05 \
        --drop 0.0 --drop-path 0.1 --train-interpolation bicubic \
        --attention-normalization vanilla --attention-gamma 1.0 \
        --bcss-variant e0 --psl-variant baseline \
        --finetune "$tgca_pretrain" --num_workers 10
    tgca_checkpoint="$tgca_run_dir/${tgca_model_name}_final.pth"
    sha256sum "$tgca_checkpoint" > "$tgca_run_dir/checkpoint_manifest.txt"
    printf 'STAGE=train finished=%s\n' "$(date --iso-8601=seconds)"
fi

printf 'STAGE=checkpoint_audit started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu" python tools/audit_mctformerplus_variant.py \
    --checkpoint "$tgca_checkpoint" --model "$tgca_model_name" \
    --official-pretrained "$tgca_pretrain" \
    --expected-pretrained-sha256 "$tgca_pretrain_hash" \
    --expected-epochs "$tgca_epochs" --expected-seed "$tgca_seed" \
    --output "$tgca_run_dir/checkpoint_audit.json"
if [[ ! -f "$tgca_run_dir/model_spec.json" ]]; then
    python - "$tgca_run_dir/checkpoint_audit.json" "$tgca_run_dir/model_spec.json" <<'PY'
import json, sys
source, output = sys.argv[1:]
value = json.load(open(source))['model_spec']
with open(output, 'x') as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write('\n')
PY
fi

printf 'STAGE=classification started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu" python tools/evaluate_mctformerplus_classification.py \
    --checkpoint "$tgca_checkpoint" --model "$tgca_model_name" \
    --voc-root "$tgca_voc_root" --list-path "$tgca_cam_val_list" \
    --input-size 448 --batch-size "$tgca_class_batch" --num-workers 8 \
    --bootstrap-resamples "$tgca_bootstrap" --bootstrap-seed 2027 \
    --output-dir "$tgca_run_dir/classification"

if [[ "$tgca_stage" == reanalysis ]]; then
    tgca_train_cam_dir="$tgca_small_run/cam_train"
else
    printf 'STAGE=cam_train started=%s\n' "$(date --iso-8601=seconds)"
    CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
        --dataset VOC12 --model "$tgca_model_name" \
        --voc12_root "$tgca_voc_root" --work_space "$tgca_run_dir" \
        --cam_out_dir cam_train --train_list "$tgca_cam_train_list" \
        --input_size 448 --scales 1.0,0.75,1.25 \
        --attention-normalization vanilla --bcss-variant e0 \
        --psl-variant baseline --checkpoint "$tgca_checkpoint"
    tgca_train_cam_dir="$tgca_run_dir/cam_train"
fi

printf 'STAGE=threshold_train started=%s\n' "$(date --iso-8601=seconds)"
python tools/evaluate_cam_threshold_grid.py \
    --cam-dir "$tgca_train_cam_dir" --voc-root "$tgca_voc_root" \
    --id-list "$tgca_cam_train_list" \
    --output-dir "$tgca_run_dir/cam_evaluation_train" \
    --threshold-start 0 --threshold-stop 0.59 --threshold-step 0.01 \
    --fixed-threshold 0.45

tgca_calibration_file="$tgca_repo_root/results/mctformerplus_width_scaling/voc/references/small_calibrated_threshold.json"
if [[ "$tgca_stage" == reanalysis ]]; then
    python - "$tgca_run_dir/cam_evaluation_train/metrics.json" \
            "$tgca_calibration_file" "$tgca_checkpoint" <<'PY'
import hashlib, json, pathlib, sys
metrics_path, output_path, checkpoint_path = map(pathlib.Path, sys.argv[1:])
metrics = json.load(open(metrics_path))
h = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
payload = {
    'threshold': metrics['oracle_selection']['threshold'],
    'selection_source': str(metrics_path.resolve()),
    'selection_split': 'VOC train',
    'grid': metrics['threshold_grid'],
    'tie_break': metrics['oracle_selection']['tie_break'],
    'small_checkpoint': str(checkpoint_path.resolve()),
    'small_checkpoint_sha256': h,
    'frozen_before_tiny_base_val_evaluation': True,
}
text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
if output_path.exists():
    existing = json.load(open(output_path))
    for key in ('threshold', 'grid', 'small_checkpoint_sha256', 'selection_split'):
        if existing.get(key) != payload.get(key):
            raise RuntimeError(f'Existing calibration differs for {key}: {output_path}')
else:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
print(payload['threshold'])
PY
elif [[ ! -f "$tgca_calibration_file" ]]; then
    echo "Small-calibrated threshold is absent; run immutable Small calibration first." >&2
    exit 2
fi
tgca_small_threshold=$(python -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["threshold"])' \
    "$tgca_calibration_file")

printf 'STAGE=cam_val started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
    --dataset VOC12 --model "$tgca_model_name" \
    --voc12_root "$tgca_voc_root" --work_space "$tgca_run_dir" \
    --cam_out_dir cam_val --train_list "$tgca_cam_val_list" \
    --input_size 448 --scales 1.0,0.75,1.25 \
    --attention-normalization vanilla --bcss-variant e0 \
    --psl-variant baseline --checkpoint "$tgca_checkpoint"

printf 'STAGE=threshold_val started=%s\n' "$(date --iso-8601=seconds)"
python tools/evaluate_cam_threshold_grid.py \
    --cam-dir "$tgca_run_dir/cam_val" --voc-root "$tgca_voc_root" \
    --id-list "$tgca_cam_val_list" \
    --output-dir "$tgca_run_dir/cam_evaluation_val" \
    --threshold-start 0 --threshold-stop 0.59 --threshold-step 0.01 \
    --fixed-threshold 0.45 --calibrated-threshold "$tgca_small_threshold"
cp --no-clobber "$tgca_run_dir/cam_evaluation_train/threshold_curve.csv" \
    "$tgca_run_dir/threshold_curve_train.csv"
cp --no-clobber "$tgca_run_dir/cam_evaluation_val/threshold_curve.csv" \
    "$tgca_run_dir/threshold_curve_val.csv"

printf 'STAGE=benchmark started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu" python tools/benchmark_mctformerplus.py \
    --checkpoint "$tgca_checkpoint" --model "$tgca_model_name" \
    --mode vanilla --output "$tgca_run_dir/benchmark.json" \
    --input-size 448 --batch-size 1 --warmup "$tgca_benchmark_warmup" \
    --iterations "$tgca_benchmark_iterations" --device cuda

printf 'STAGE=semantic_ownership started=%s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES="$tgca_gpu" python tools/evaluate_mctformerplus_semantic_ownership.py \
    --checkpoint "$tgca_checkpoint" --model "$tgca_model_name" \
    --voc-root "$tgca_voc_root" --list-path "$tgca_cam_val_list" \
    --output-dir "$tgca_run_dir/semantic_ownership" \
    --batch-size "$tgca_semantic_batch" --num-workers 8 \
    --bootstrap-resamples "$tgca_semantic_bootstrap" --bootstrap-seed 2027

python - "$tgca_run_dir" "$tgca_checkpoint" "$tgca_cam_train_list" \
        "$tgca_cam_val_list" "$tgca_stage" <<'PY'
import hashlib, json, pathlib, sys
root, checkpoint, train_list, val_list, stage = sys.argv[1:]
root, checkpoint = pathlib.Path(root), pathlib.Path(checkpoint)
ids = lambda p: [x.strip() for x in pathlib.Path(p).read_text().splitlines() if x.strip()]
train_expected, val_expected = len(ids(train_list)), len(ids(val_list))
train_source = pathlib.Path(json.load(open(root/'small_source_pointer.json'))['path'])/'cam_train' if stage == 'reanalysis' else root/'cam_train'
checks = {
    'checkpoint_exists': checkpoint.is_file(),
    'classification_complete': (root/'classification/CLASSIFICATION_COMPLETE').is_file(),
    'train_threshold_complete': (root/'cam_evaluation_train/THRESHOLD_EVALUATION_COMPLETE').is_file(),
    'val_cam_complete': (root/'cam_val/CAM_COMPLETE').is_file(),
    'val_threshold_complete': (root/'cam_evaluation_val/THRESHOLD_EVALUATION_COMPLETE').is_file(),
    'semantic_complete': (root/'semantic_ownership/SEMANTIC_OWNERSHIP_COMPLETE').is_file(),
    'benchmark_exists': (root/'benchmark.json').is_file(),
    'training_runtime_exists': (root/'training_runtime.json').is_file(),
    'train_cam_count': len(list(train_source.glob('*.npy'))) == train_expected,
    'val_cam_count': len(list((root/'cam_val').glob('*.npy'))) == val_expected,
}
if not all(checks.values()):
    raise RuntimeError(checks)
report = json.load(open(root/'checkpoint_audit.json'))
report['pipeline_checks'] = checks
report['checkpoint']['sha256_rechecked'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
(root/'audit_report.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
PY
printf 'PIPELINE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" | tee "$tgca_run_dir/PIPELINE_COMPLETE"
