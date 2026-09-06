#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != tgca-repro ]]; then
    echo "Activate the tgca-repro Conda environment before running." >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != main ]]; then
    echo "The FinalLN matched experiment must start from main." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

tgca_gpu=${TGCA_GPU_ID:-0}
tgca_seed=0
tgca_commit=$(git rev-parse HEAD)
tgca_short_commit=$(git rev-parse --short HEAD)
tgca_run_id=${TGCA_RUN_ID:-"$(date +%Y%m%d)-mctformerplus-final-ln-matched-s0-${tgca_short_commit}"}
tgca_run_root="$tgca_repo_root/results/final_ln_ablation/$tgca_run_id"
tgca_voc_root="$tgca_repo_root/data/VOCdevkit/VOC2012"
tgca_train_aug="$tgca_voc_root/ImageLists/train_aug_id.txt"
tgca_train_cam="$tgca_voc_root/ImageLists/train_id.txt"
tgca_val="$tgca_voc_root/ImageLists/val_id.txt"
tgca_pretrained=/home/peng/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth
tgca_pretrained_sha=cd65a15597004d0ce19d7a9daef969903972db5b398e3a5febcd3c4df1d8f59f

if [[ -e "$tgca_run_root" ]]; then
    echo "Refusing to overwrite existing run directory: $tgca_run_root" >&2
    exit 2
fi
for tgca_path in "$tgca_voc_root" "$tgca_train_aug" "$tgca_train_cam" \
        "$tgca_val" "$tgca_pretrained"; do
    if [[ ! -e "$tgca_path" ]]; then
        echo "Required input is absent: $tgca_path" >&2
        exit 2
    fi
done
if [[ "$(sha256sum "$tgca_pretrained" | awk '{print $1}')" != "$tgca_pretrained_sha" ]]; then
    echo "Official DeiT-S pretrained hash mismatch." >&2
    exit 2
fi

mkdir -p "$tgca_run_root"/{audit,checkpoints,evaluations,training_logs}
tgca_exact="$tgca_run_root/exact_commands.sh"
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
    "cd $(printf %q "$tgca_repo_root")" \
    'source /home/peng/anaconda3/etc/profile.d/conda.sh' \
    'conda activate tgca-repro' > "$tgca_exact"
chmod 755 "$tgca_exact"
exec > >(tee "$tgca_run_root/pipeline.log") 2>&1

run_exact() {
    printf ' ' >> "$tgca_exact"
    printf '%q ' "$@" >> "$tgca_exact"
    printf '\n' >> "$tgca_exact"
    "$@"
}

printf 'run_id=%s\nrun_root=%s\ncommit=%s\n' \
    "$tgca_run_id" "$tgca_run_root" "$tgca_commit"
printf '{"commit":"%s","branch":"main","tracked_dirty":false}\n' \
    "$tgca_commit" > "$tgca_run_root/git_state.json"
{
    python --version
    python -c "import torch, torchvision, timm, sklearn; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('sklearn', sklearn.__version__); print('cuda', torch.version.cuda)"
    printf 'conda_env=%s\n' "$CONDA_DEFAULT_ENV"
} > "$tgca_run_root/environment.txt" 2>&1
python -m pip freeze > "$tgca_run_root/pip_freeze.txt"
conda list --explicit > "$tgca_run_root/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total \
    --format=csv,noheader > "$tgca_run_root/hardware.txt"
{
    sha256sum "$tgca_train_aug" "$tgca_train_cam" "$tgca_val"
    sha256sum "$tgca_voc_root/ImageLabel/cls_labels.npy"
    printf 'train_aug_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_train_aug")"
    printf 'train_cam_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_train_cam")"
    printf 'val_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_val")"
} > "$tgca_run_root/dataset_manifest.txt"
sha256sum "$tgca_pretrained" > "$tgca_run_root/pretrained_manifest.txt"

python - "$tgca_run_root" "$tgca_commit" <<'PY'
import json, pathlib, sys
root, commit = pathlib.Path(sys.argv[1]), sys.argv[2]
common = {
    'dataset': 'PASCAL VOC 2012', 'seed': 0, 'model': 'mctformerplus',
    'deit_variant': 'small', 'input_size': 448, 'epochs': 45,
    'micro_batch_size': 32, 'accum_iter': 1, 'effective_batch_size': 32,
    'optimizer': 'adamw', 'nominal_lr': 0.0005,
    'optimizer_lr': 0.00003125, 'minimum_lr': 0.00001,
    'weight_decay': 0.05, 'scheduler': 'cosine', 'warmup_epochs': 5,
    'drop': 0.0, 'drop_path': 0.1, 'train_interpolation': 'bicubic',
    'attention_normalization': 'vanilla', 'attention_gamma': 1.0,
    'bcss_variant': 'e0', 'psl_variant': 'baseline', 'cti_bgt': False,
    'patch_head': 'existing 3x3 convolution', 'patch_pooling': 'existing GWRP',
    'cam_scales': [1.0, 0.75, 1.25],
    'cam_class_to_patch_layers': [10, 11, 12],
    'cam_patch_to_patch_layers': list(range(1, 13)),
    'cam_sqrt_refinement': True, 'checkpoint_policy': 'final',
    'commit': commit,
}
for name, enabled in (('baseline', False), ('final_ln', True)):
    value = dict(common)
    value.update({
        'experiment_name': 'MCTformer+' if not enabled else 'MCTformer+-FinalLN',
        'final_norm': enabled,
        'final_norm_placement': (
            'disabled' if not enabled
            else 'after block 12 on complete token sequence, before class/patch split'
        ),
        'cct_all_x_cls': 'raw post-block class tokens; never final-normalized',
    })
    (root / f'config_{name}.json').write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n'
    )
PY

printf 'STAGE=tests started=%s\n' "$(date --iso-8601=seconds)"
run_exact python -m pytest -q \
    tests/test_mctformerplus_final_norm.py \
    tests/test_mctformerplus_final_norm_diagnostics.py \
    tests/test_mctformerplus_variants.py \
    tests/test_mctformerplus_pretrained_loading.py \
    tests/test_lazy_assignment_token_collector.py \
    tests/test_experiment2_native_cam.py \
    tests/test_width_scaling_aggregation.py
printf 'STAGE=tests finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/training_logs/tests_complete.txt"

run_variant() {
    local tgca_key=$1
    local tgca_enabled=$2
    local tgca_train_root="$tgca_run_root/checkpoints/$tgca_key"
    local tgca_eval_root="$tgca_run_root/evaluations/$tgca_key"
    local -a tgca_final_args=()
    if [[ "$tgca_enabled" == true ]]; then
        tgca_final_args+=(--final-norm)
    fi

    printf 'STAGE=%s_train started=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python train_model_v2.py \
        --dataset VOC12 --model mctformerplus \
        --voc12_root "$tgca_voc_root" \
        --train_list "$tgca_train_aug" --val_list "$tgca_val" \
        --work_space "$tgca_train_root" --input-size 448 \
        --epochs 45 --batch_size 32 --accum-iter 1 --val-batch-size 32 \
        --seed "$tgca_seed" --opt adamw --sched cosine --warmup-epochs 5 \
        --lr 5e-4 --min-lr 1e-5 --weight-decay 0.05 \
        --drop 0.0 --drop-path 0.1 --train-interpolation bicubic \
        --attention-normalization vanilla --attention-gamma 1.0 \
        --bcss-variant e0 --psl-variant baseline \
        --finetune "$tgca_pretrained" --num_workers 10 \
        "${tgca_final_args[@]}"
    local tgca_checkpoint="$tgca_train_root/mctformerplus_final.pth"
    sha256sum "$tgca_checkpoint" > "$tgca_train_root/checkpoint_manifest.txt"
    local tgca_source_log
    tgca_source_log=$(find "$tgca_train_root/log_dir" -maxdepth 1 -type f \
        -name 'train-*.log' -print -quit)
    cp --no-clobber "$tgca_source_log" "$tgca_run_root/training_logs/${tgca_key}.log"
    printf 'STAGE=%s_train finished=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"

    printf 'STAGE=%s_audit started=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
        tools/audit_mctformerplus_variant.py \
        --checkpoint "$tgca_checkpoint" --model mctformerplus \
        --official-pretrained "$tgca_pretrained" \
        --expected-pretrained-sha256 "$tgca_pretrained_sha" \
        --expected-epochs 45 --expected-effective-batch 32 --expected-seed 0 \
        --output "$tgca_run_root/audit/${tgca_key}.json" \
        "${tgca_final_args[@]}"

    printf 'STAGE=%s_classification started=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
        tools/evaluate_mctformerplus_classification.py \
        --checkpoint "$tgca_checkpoint" --model mctformerplus \
        --voc-root "$tgca_voc_root" --list-path "$tgca_val" \
        --input-size 448 --batch-size 16 --num-workers 8 \
        --bootstrap-resamples 5000 --bootstrap-seed 20270906 \
        --output-dir "$tgca_eval_root/classification" \
        "${tgca_final_args[@]}"

    printf 'STAGE=%s_cam_train started=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
        --dataset VOC12 --model mctformerplus \
        --voc12_root "$tgca_voc_root" --work_space "$tgca_eval_root" \
        --cam_out_dir cam_train --train_list "$tgca_train_cam" \
        --input_size 448 --scales 1.0,0.75,1.25 \
        --attention-normalization vanilla --bcss-variant e0 \
        --psl-variant baseline --checkpoint "$tgca_checkpoint" \
        "${tgca_final_args[@]}"

    printf 'STAGE=%s_cam_evaluation started=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
    run_exact python tools/evaluate_cam_threshold_grid.py \
        --cam-dir "$tgca_eval_root/cam_train" --voc-root "$tgca_voc_root" \
        --id-list "$tgca_train_cam" \
        --output-dir "$tgca_eval_root/cam_evaluation" \
        --threshold-start 0 --threshold-stop 0.59 --threshold-step 0.01 \
        --fixed-threshold 0.45

    printf 'STAGE=%s_focused_diagnostics started=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
        tools/evaluate_mctformerplus_final_ln_diagnostics.py \
        --checkpoint "$tgca_checkpoint" --voc-root "$tgca_voc_root" \
        --list-path "$tgca_val" --output-dir "$tgca_eval_root/diagnostics" \
        --input-size 448 --batch-size 4 --num-workers 8 \
        --bootstrap-resamples 5000 --bootstrap-seed 20270906 \
        "${tgca_final_args[@]}"
    printf 'STAGE=%s_complete finished=%s\n' "$tgca_key" "$(date --iso-8601=seconds)"
}

run_variant baseline false
run_variant final_ln true

printf 'STAGE=summary started=%s\n' "$(date --iso-8601=seconds)"
run_exact python tools/summarize_mctformerplus_final_ln.py \
    --run-root "$tgca_run_root"
printf 'PIPELINE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/PIPELINE_COMPLETE"
