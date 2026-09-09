#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != tgca-repro ]]; then
    echo "Activate the tgca-repro Conda environment before running." >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != main ]]; then
    echo "The decoupled experiment must start from main." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Refusing to start from a tracked dirty worktree." >&2
    exit 2
fi

tgca_gpu=${TGCA_GPU_ID:-0}
tgca_seed=0
tgca_variants=(
    full
    no_p2c
    no_c2c
    no_p2c_no_c2c
    c2p_update_off
    p2p_update_off
    p2c_middle
    p2c_early
)
tgca_commit=$(git rev-parse HEAD)
tgca_short_commit=$(git rev-parse --short HEAD)
tgca_run_id=${TGCA_RUN_ID:-"$(date +%Y%m%d)-mctformerplus-decoupled-s0-${tgca_short_commit}"}
tgca_run_root="$tgca_repo_root/results/decoupled_bidirectional/$tgca_run_id"
tgca_baseline_root=${TGCA_BASELINE_SOURCE_RUN:-"$tgca_repo_root/results/final_ln_ablation/20260906-mctformerplus-final-ln-matched-s0-0ef6bd8"}
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
for tgca_path in \
        "$tgca_train_aug" "$tgca_train_cam" "$tgca_val" \
        "$tgca_voc_root/ImageLabel/cls_labels.npy" "$tgca_pretrained" \
        "$tgca_baseline_root/EXPERIMENT_COMPLETE" \
        "$tgca_baseline_root/audit/baseline.json" \
        "$tgca_baseline_root/evaluations/baseline/classification/classification_metrics.json" \
        "$tgca_baseline_root/evaluations/baseline/cam_evaluation/metrics.json"; do
    if [[ ! -e "$tgca_path" ]]; then
        echo "Required input is absent: $tgca_path" >&2
        exit 2
    fi
done
if [[ "$(sha256sum "$tgca_pretrained" | awk '{print $1}')" != "$tgca_pretrained_sha" ]]; then
    echo "Official DeiT-S pretrained hash mismatch." >&2
    exit 2
fi

mkdir -p "$tgca_run_root"/{smoke,variants}
tgca_exact="$tgca_run_root/exact_commands.sh"
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
    "cd $(printf %q "$tgca_repo_root")" \
    'source /home/peng/anaconda3/etc/profile.d/conda.sh' \
    'conda activate tgca-repro' > "$tgca_exact"
chmod 755 "$tgca_exact"
printf 'bash experiments/ablations/run_mctformerplus_decoupled_queue.sh\n' \
    > "$tgca_run_root/command.txt"
exec > >(tee "$tgca_run_root/pipeline.log") 2>&1

run_exact() {
    printf ' ' >> "$tgca_exact"
    printf '%q ' "$@" >> "$tgca_exact"
    printf '\n' >> "$tgca_exact"
    "$@"
}

run_exact_logged() {
    local tgca_log=$1
    shift
    printf ' ' >> "$tgca_exact"
    printf '%q ' "$@" >> "$tgca_exact"
    printf '\n' >> "$tgca_exact"
    "$@" 2>&1 | tee "$tgca_log"
}

printf 'run_id=%s\nrun_root=%s\ncommit=%s\n' \
    "$tgca_run_id" "$tgca_run_root" "$tgca_commit"
printf '{"commit":"%s","branch":"main","tracked_dirty":false}\n' \
    "$tgca_commit" > "$tgca_run_root/git_state.json"
python - "$tgca_run_root/config.json" "$tgca_commit" \
        "$tgca_baseline_root" <<'PY'
import json, pathlib, sys
output, commit, baseline = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = {
    'experiment_name': 'Bidirectional Decoupled MCTformer+ Ablation Queue',
    'model': 'mctformerplus', 'variant': 'small',
    'dataset': 'PASCAL VOC 2012', 'seed': 0, 'input_size': 448,
    'variants_in_order': [
        'full', 'no_p2c', 'no_c2c', 'no_p2c_no_c2c',
        'c2p_update_off', 'p2p_update_off', 'p2c_middle', 'p2c_early',
    ],
    'epochs': 45, 'micro_batch_size': 32, 'accum_iter': 1,
    'effective_batch_size': 32, 'optimizer': 'adamw',
    'nominal_lr': 0.0005, 'optimizer_lr': 0.00003125,
    'minimum_lr': 0.00001, 'weight_decay': 0.05,
    'scheduler': 'cosine', 'warmup_epochs': 5,
    'drop': 0.0, 'drop_path': 0.1,
    'train_interpolation': 'bicubic',
    'attention_normalization': 'vanilla', 'attention_gamma': 1.0,
    'bcss_variant': 'e0', 'psl_variant': 'baseline', 'cti_bgt': False,
    'final_norm': False, 'patch_final_norm': False,
    'last_mct': False, 'class_stable_last': False,
    'class_token_init': 'baseline',
    'patch_head': 'original Conv2d(384,20,kernel_size=3,padding=1) + GWRP',
    'class_readout': 'raw L12 class tokens -> mean(dim=-1)',
    'cct': 'raw post-block class tokens from all 12 blocks',
    'cam': 'native last-three C2P + sqrt + all-layer P2P propagation',
    'cam_scales': [1.0, 0.75, 1.25], 'fixed_cam_threshold': 0.45,
    'threshold_grid': {'start': 0.0, 'stop': 0.59, 'step': 0.01},
    'checkpoint_policy': 'final',
    'baseline_root': str(pathlib.Path(baseline).resolve()),
    'baseline_mode': 'read_only', 'commit': commit,
}
output.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
PY
{
    python --version
    python -c "import torch, torchvision, timm, sklearn; print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('timm', timm.__version__); print('sklearn', sklearn.__version__); print('cuda', torch.version.cuda)"
    printf 'conda_env=%s\n' "$CONDA_DEFAULT_ENV"
} > "$tgca_run_root/environment.txt" 2>&1
python -m pip freeze > "$tgca_run_root/pip_freeze.txt"
conda list --explicit > "$tgca_run_root/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free \
    --format=csv,noheader > "$tgca_run_root/hardware.txt"
{
    sha256sum "$tgca_train_aug" "$tgca_train_cam" "$tgca_val"
    sha256sum "$tgca_voc_root/ImageLabel/cls_labels.npy" "$tgca_pretrained"
} > "$tgca_run_root/input_manifest.txt"
{
    sha256sum "$tgca_baseline_root/audit/baseline.json"
    sha256sum "$tgca_baseline_root/evaluations/baseline/classification/classification_metrics.json"
    sha256sum "$tgca_baseline_root/evaluations/baseline/cam_evaluation/metrics.json"
} > "$tgca_run_root/baseline_read_only_manifest.txt"

printf 'STAGE=tests started=%s\n' "$(date --iso-8601=seconds)"
run_exact_logged "$tgca_run_root/tests.txt" python -m pytest -q \
    tests/test_mctformerplus_decoupled.py \
    tests/test_mctformerplus_attention.py \
    tests/test_mctformerplus_variants.py \
    tests/test_mctformerplus_pretrained_loading.py \
    tests/test_mctformerplus_final_norm.py \
    tests/test_experiment2_native_cam.py
printf 'STAGE=tests finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee -a "$tgca_run_root/tests.txt"

printf 'STAGE=smoke started=%s\n' "$(date --iso-8601=seconds)"
mkdir -p "$tgca_run_root/smoke"/{audit,checkpoints,lists}
head -n 3200 "$tgca_train_aug" > "$tgca_run_root/smoke/lists/train_aug_id.txt"
head -n 4 "$tgca_val" > "$tgca_run_root/smoke/lists/val_id.txt"
tgca_smoke_log="$tgca_run_root/smoke/training_stdout.log"
run_exact_logged "$tgca_smoke_log" \
    env CUDA_VISIBLE_DEVICES="$tgca_gpu" python train_model_v2.py \
    --dataset VOC12 --model mctformerplus \
    --token-interaction decoupled_bidirectional --decoupled-variant full \
    --voc12_root "$tgca_voc_root" \
    --train_list "$tgca_run_root/smoke/lists/train_aug_id.txt" \
    --val_list "$tgca_run_root/smoke/lists/val_id.txt" \
    --work_space "$tgca_run_root/smoke/checkpoints" --input-size 448 \
    --epochs 1 --batch_size 32 --accum-iter 1 --val-batch-size 4 \
    --seed "$tgca_seed" --opt adamw --sched cosine --warmup-epochs 5 \
    --lr 5e-4 --min-lr 1e-5 --weight-decay 0.05 \
    --drop 0.0 --drop-path 0.1 --train-interpolation bicubic \
    --attention-normalization vanilla --attention-gamma 1.0 \
    --bcss-variant e0 --psl-variant baseline \
    --finetune "$tgca_pretrained" --num_workers 4
tgca_smoke_checkpoint="$tgca_run_root/smoke/checkpoints/mctformerplus_final.pth"
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python tools/audit_mctformerplus_variant.py \
    --checkpoint "$tgca_smoke_checkpoint" --model mctformerplus \
    --token-interaction decoupled_bidirectional --decoupled-variant full \
    --official-pretrained "$tgca_pretrained" \
    --expected-pretrained-sha256 "$tgca_pretrained_sha" \
    --expected-epochs 1 --expected-effective-batch 32 --expected-seed 0 \
    --output "$tgca_run_root/smoke/audit/full.json"
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
    --dataset VOC12 --model mctformerplus \
    --token-interaction decoupled_bidirectional --decoupled-variant full \
    --voc12_root "$tgca_voc_root" --work_space "$tgca_run_root/smoke" \
    --cam_out_dir cam --train_list "$tgca_run_root/smoke/lists/val_id.txt" \
    --input_size 448 --scales 1.0 --attention-normalization vanilla \
    --bcss-variant e0 --psl-variant baseline --checkpoint "$tgca_smoke_checkpoint"
python - "$tgca_smoke_log" "$tgca_run_root/smoke/audit/full.json" \
        "$tgca_run_root/smoke/cam" "$tgca_run_root/smoke/smoke_summary.json" <<'PY'
import json, math, pathlib, re, sys
train_log, audit_path, cam_dir, output = map(pathlib.Path, sys.argv[1:])
text = train_log.read_text()
metrics = {
    name: [float(item) for item in re.findall(fr'{name}: ([0-9.eE+-]+)', text)]
    for name in ('mct_loss', 'attn_loss', 'pat_loss', 'loss')
}
if any(not sequence for sequence in metrics.values()):
    raise SystemExit(f'missing smoke losses: {metrics}')
if not all(math.isfinite(value) for seq in metrics.values() for value in seq):
    raise SystemExit('non-finite smoke loss')
audit = json.loads(audit_path.read_text())
cams = sorted(cam_dir.glob('*.npy'))
payload = {
    'status': 'pass', 'training_iterations': 100,
    'strict_audit_passed': bool(audit.get('passed')),
    'finite_losses': True, 'cam_files': len(cams),
    'cam_complete': (cam_dir / 'CAM_COMPLETE').is_file(),
}
if (not payload['strict_audit_passed'] or not payload['cam_complete']
        or len(cams) != 4):
    raise SystemExit(f'smoke validation failed: {payload}')
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY
printf 'SMOKE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/smoke/SMOKE_COMPLETE"

for tgca_variant in "${tgca_variants[@]}"; do
    tgca_variant_root="$tgca_run_root/variants/$tgca_variant"
    tgca_train_root="$tgca_variant_root/checkpoints"
    tgca_eval_root="$tgca_variant_root/evaluations"
    mkdir -p "$tgca_variant_root"
    printf 'STAGE=%s_train started=%s\n' \
        "$tgca_variant" "$(date --iso-8601=seconds)"
    run_exact_logged "$tgca_variant_root/training_stdout.log" \
        env CUDA_VISIBLE_DEVICES="$tgca_gpu" python train_model_v2.py \
        --dataset VOC12 --model mctformerplus \
        --token-interaction decoupled_bidirectional \
        --decoupled-variant "$tgca_variant" \
        --voc12_root "$tgca_voc_root" \
        --train_list "$tgca_train_aug" --val_list "$tgca_val" \
        --work_space "$tgca_train_root" --input-size 448 \
        --epochs 45 --batch_size 32 --accum-iter 1 --val-batch-size 32 \
        --seed "$tgca_seed" --opt adamw --sched cosine --warmup-epochs 5 \
        --lr 5e-4 --min-lr 1e-5 --weight-decay 0.05 \
        --drop 0.0 --drop-path 0.1 --train-interpolation bicubic \
        --attention-normalization vanilla --attention-gamma 1.0 \
        --bcss-variant e0 --psl-variant baseline \
        --finetune "$tgca_pretrained" --num_workers 10
    tgca_checkpoint="$tgca_train_root/mctformerplus_final.pth"
    sha256sum "$tgca_checkpoint" > "$tgca_variant_root/checkpoint_manifest.txt"
    printf 'STAGE=%s_train finished=%s\n' \
        "$tgca_variant" "$(date --iso-8601=seconds)"

    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
        tools/audit_mctformerplus_variant.py \
        --checkpoint "$tgca_checkpoint" --model mctformerplus \
        --token-interaction decoupled_bidirectional \
        --decoupled-variant "$tgca_variant" \
        --official-pretrained "$tgca_pretrained" \
        --expected-pretrained-sha256 "$tgca_pretrained_sha" \
        --expected-epochs 45 --expected-effective-batch 32 --expected-seed 0 \
        --output "$tgca_variant_root/audit.json"

    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
        tools/evaluate_mctformerplus_classification.py \
        --checkpoint "$tgca_checkpoint" --model mctformerplus \
        --token-interaction decoupled_bidirectional \
        --decoupled-variant "$tgca_variant" \
        --voc-root "$tgca_voc_root" --list-path "$tgca_val" \
        --input-size 448 --batch-size 16 --num-workers 8 \
        --bootstrap-resamples 5000 --bootstrap-seed 20270909 \
        --output-dir "$tgca_eval_root/classification"

    run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
        --dataset VOC12 --model mctformerplus \
        --token-interaction decoupled_bidirectional \
        --decoupled-variant "$tgca_variant" \
        --voc12_root "$tgca_voc_root" --work_space "$tgca_eval_root" \
        --cam_out_dir cam_train --train_list "$tgca_train_cam" \
        --input_size 448 --scales 1.0,0.75,1.25 \
        --attention-normalization vanilla --bcss-variant e0 \
        --psl-variant baseline --checkpoint "$tgca_checkpoint"

    run_exact python tools/evaluate_cam_threshold_grid.py \
        --cam-dir "$tgca_eval_root/cam_train" --voc-root "$tgca_voc_root" \
        --id-list "$tgca_train_cam" \
        --output-dir "$tgca_eval_root/cam_evaluation" \
        --threshold-start 0 --threshold-stop 0.59 --threshold-step 0.01 \
        --fixed-threshold 0.45
    printf 'VARIANT_COMPLETE variant=%s finished=%s\n' \
        "$tgca_variant" "$(date --iso-8601=seconds)" \
        | tee "$tgca_variant_root/VARIANT_COMPLETE"
done

run_exact python tools/summarize_mctformerplus_decoupled.py \
    --run-root "$tgca_run_root" --baseline-root "$tgca_baseline_root"
printf 'PIPELINE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/PIPELINE_COMPLETE"
