#!/usr/bin/env bash
set -euo pipefail

tgca_repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$tgca_repo_root"

if [[ "${CONDA_DEFAULT_ENV:-}" != tgca-repro ]]; then
    echo "Activate the tgca-repro Conda environment before running." >&2
    exit 2
fi
if [[ "$(git branch --show-current)" != main ]]; then
    echo "The Last-MCT experiment must start from main." >&2
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
tgca_run_id=${TGCA_RUN_ID:-"$(date +%Y%m%d)-last-mct-voc-s0-${tgca_short_commit}"}
tgca_run_root="$tgca_repo_root/results/last_mct/$tgca_run_id"
tgca_source_root=${TGCA_FINAL_LN_SOURCE_RUN:-"$tgca_repo_root/results/final_ln_ablation/20260906-mctformerplus-final-ln-matched-s0-0ef6bd8"}
tgca_patch_root=${TGCA_PATCH_FINAL_LN_SOURCE_RUN:-"$tgca_repo_root/results/final_ln_ablation/20260906-mctformerplus-patch-final-ln-s0-64f2aa9"}
tgca_source_summary_sha=${TGCA_FINAL_LN_SOURCE_SUMMARY_SHA:-0b845989429919ce7336f8e9525a7f41d69ada9536f5cde0178438b5310054e6}
tgca_patch_summary_sha=${TGCA_PATCH_FINAL_LN_SOURCE_SUMMARY_SHA:-38f514939ffcd6ab420c0f4b19270ecee59911f87f73ca3d6d68315522a993b5}
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
        "$tgca_source_root/PIPELINE_COMPLETE" \
        "$tgca_source_root/EXPERIMENT_COMPLETE" \
        "$tgca_patch_root/PIPELINE_COMPLETE" \
        "$tgca_patch_root/EXPERIMENT_COMPLETE" \
        "$tgca_train_aug" "$tgca_train_cam" "$tgca_val" \
        "$tgca_voc_root/ImageLabel/cls_labels.npy" "$tgca_pretrained"; do
    if [[ ! -e "$tgca_path" ]]; then
        echo "Required input is absent: $tgca_path" >&2
        exit 2
    fi
done
if [[ "$(sha256sum "$tgca_pretrained" | awk '{print $1}')" != "$tgca_pretrained_sha" ]]; then
    echo "Official DeiT-S pretrained hash mismatch." >&2
    exit 2
fi
if [[ "$(sha256sum "$tgca_source_root/summary.json" | awk '{print $1}')" != "$tgca_source_summary_sha" ]]; then
    echo "Original MCTformer+ source summary hash mismatch." >&2
    exit 2
fi
if [[ "$(sha256sum "$tgca_patch_root/summary.json" | awk '{print $1}')" != "$tgca_patch_summary_sha" ]]; then
    echo "PatchFinalLN source summary hash mismatch." >&2
    exit 2
fi

mkdir -p "$tgca_run_root"/{audit,checkpoints,evaluations,training_logs,smoke}
tgca_exact="$tgca_run_root/exact_commands.sh"
printf '%s\n' '#!/usr/bin/env bash' 'set -euo pipefail' \
    "cd $(printf %q "$tgca_repo_root")" \
    'source /home/peng/anaconda3/etc/profile.d/conda.sh' \
    'conda activate tgca-repro' > "$tgca_exact"
chmod 755 "$tgca_exact"
printf 'bash experiments/ablations/run_last_mct_voc.sh\n' \
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
python - "$tgca_run_root/config.json" "$tgca_commit" <<'PY'
import json, math, pathlib, sys
output, commit = pathlib.Path(sys.argv[1]), sys.argv[2]
value = {
    'experiment_name': 'Last-MCT',
    'dataset': 'PASCAL VOC 2012',
    'model': 'mctformerplus',
    'deit_variant': 'small',
    'seed': 0,
    'input_size': 448,
    'epochs': 45,
    'micro_batch_size': 32,
    'accum_iter': 1,
    'effective_batch_size': 32,
    'optimizer': 'adamw',
    'nominal_lr': 0.0005,
    'optimizer_lr': 0.00003125,
    'minimum_lr': 0.00001,
    'weight_decay': 0.05,
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'drop': 0.0,
    'drop_path': 0.1,
    'train_interpolation': 'bicubic',
    'attention_normalization': 'vanilla',
    'attention_gamma': 1.0,
    'bcss_variant': 'e0',
    'psl_variant': 'baseline',
    'cti_bgt': False,
    'final_norm': False,
    'patch_final_norm': True,
    'class_token_readout': 'raw L12 class tokens -> mean(dim=-1)',
    'cct_readout': 'raw post-block class tokens from all 12 blocks',
    'last_mct': True,
    'last_topk': 1,
    'last_sigma': math.sqrt(384),
    'last_eps': 1e-6,
    'last_score_formula': 'P / abs(P_lowpass - P).clamp_min(eps)',
    'last_fft_dimension': 'embedding',
    'last_topk_dimension': 'patch, independently per embedding channel',
    'patch_head': 'shared Conv2d(384,20,kernel_size=1) / F.linear',
    'cam_scales': [1.0, 0.75, 1.25],
    'cam_class_to_patch_layers': [10, 11, 12],
    'cam_patch_to_patch_layers': list(range(1, 13)),
    'cam_sqrt_refinement': True,
    'fixed_cam_threshold': 0.45,
    'threshold_grid': {'start': 0.0, 'stop': 0.59, 'step': 0.01},
    'checkpoint_policy': 'final',
    'commit': commit,
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
df -h "$tgca_repo_root" > "$tgca_run_root/disk.txt"
{
    sha256sum "$tgca_train_aug" "$tgca_train_cam" "$tgca_val"
    sha256sum "$tgca_voc_root/ImageLabel/cls_labels.npy"
    printf 'train_aug_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_train_aug")"
    printf 'train_cam_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_train_cam")"
    printf 'val_ids=%s\n' "$(awk 'NF {n++} END {print n+0}' "$tgca_val")"
} > "$tgca_run_root/dataset_manifest.txt"
sha256sum "$tgca_pretrained" > "$tgca_run_root/pretrained_manifest.txt"

python - "$tgca_source_root" "$tgca_patch_root" \
        "$tgca_source_summary_sha" "$tgca_patch_summary_sha" \
        "$tgca_run_root/source_reference.json" <<'PY'
import hashlib, json, pathlib, sys
source, patch = map(pathlib.Path, sys.argv[1:3])
expected_source, expected_patch, output = sys.argv[3], sys.argv[4], pathlib.Path(sys.argv[5])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
source_summary = json.loads((source / 'summary.json').read_text())
patch_summary = json.loads((patch / 'summary.json').read_text())
if source_summary.get('status') != 'complete' or patch_summary.get('status') != 'complete':
    raise SystemExit('immutable matched source is incomplete')
if sha(source / 'summary.json') != expected_source or sha(patch / 'summary.json') != expected_patch:
    raise SystemExit('immutable matched source summary hash mismatch')
files = {
    'original_summary': source / 'summary.json',
    'original_classification': source / 'evaluations/baseline/classification/classification_metrics.json',
    'original_cam': source / 'evaluations/baseline/cam_evaluation/metrics.json',
    'original_audit': source / 'audit/baseline.json',
    'patch_summary': patch / 'summary.json',
    'patch_classification': patch / 'evaluations/patch_final_ln/classification/classification_metrics.json',
    'patch_cam': patch / 'evaluations/patch_final_ln/cam_evaluation/metrics.json',
    'patch_audit': patch / 'audit/patch_final_ln.json',
}
checkpoints = {
    'original': source_summary['source_checkpoints']['baseline'],
    'patch_final_ln': patch_summary['patch_checkpoint'],
}
for item in checkpoints.values():
    path = pathlib.Path(item['path'])
    if sha(path) != item['sha256']:
        raise SystemExit(f'immutable checkpoint hash mismatch: {path}')
payload = {
    'mode': 'read_only',
    'files': {key: {'path': str(path.resolve()), 'sha256': sha(path)} for key, path in files.items()},
    'checkpoints': checkpoints,
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
PY

printf 'STAGE=tests started=%s\n' "$(date --iso-8601=seconds)"
run_exact python -m pytest -q \
    tests/test_mctformerplus_last_mct.py \
    tests/test_mctformerplus_final_norm.py \
    tests/test_mctformerplus_attention.py \
    tests/test_mctformerplus_variants.py \
    tests/test_mctformerplus_pretrained_loading.py \
    tests/test_experiment2_native_cam.py
printf 'STAGE=tests finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/tests.txt" "$tgca_run_root/training_logs/tests_complete.txt"

printf 'STAGE=smoke started=%s\n' "$(date --iso-8601=seconds)"
mkdir -p "$tgca_run_root/smoke"/{audit,checkpoints,lists}
head -n 320 "$tgca_train_aug" > "$tgca_run_root/smoke/lists/train_aug_id.txt"
head -n 4 "$tgca_val" > "$tgca_run_root/smoke/lists/val_id.txt"
tgca_smoke_train_log="$tgca_run_root/smoke/training_stdout.log"
run_exact_logged "$tgca_smoke_train_log" \
    env CUDA_VISIBLE_DEVICES="$tgca_gpu" python train_model_v2.py \
    --dataset VOC12 --model mctformerplus \
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
    --finetune "$tgca_pretrained" --num_workers 4 \
    --patch-final-norm --last-mct
tgca_smoke_checkpoint="$tgca_run_root/smoke/checkpoints/mctformerplus_final.pth"
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
    tools/audit_mctformerplus_variant.py \
    --checkpoint "$tgca_smoke_checkpoint" --model mctformerplus \
    --official-pretrained "$tgca_pretrained" \
    --expected-pretrained-sha256 "$tgca_pretrained_sha" \
    --expected-epochs 1 --expected-effective-batch 32 --expected-seed 0 \
    --output "$tgca_run_root/smoke/audit/last_mct.json" \
    --patch-final-norm --last-mct
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
    --dataset VOC12 --model mctformerplus \
    --voc12_root "$tgca_voc_root" --work_space "$tgca_run_root/smoke" \
    --cam_out_dir cam --train_list "$tgca_run_root/smoke/lists/val_id.txt" \
    --input_size 448 --scales 1.0 \
    --attention-normalization vanilla --bcss-variant e0 \
    --psl-variant baseline --checkpoint "$tgca_smoke_checkpoint" \
    --patch-final-norm --last-mct
python - "$tgca_smoke_train_log" "$tgca_run_root/smoke/audit/last_mct.json" \
        "$tgca_run_root/smoke/cam" "$tgca_run_root/smoke/smoke_summary.json" <<'PY'
import json, math, pathlib, re, sys
log_path, audit_path, cam_dir, output = map(pathlib.Path, sys.argv[1:])
text = log_path.read_text()
values = [
    float(match.group(1))
    for line in text.splitlines()
    if line.startswith('Epoch: [0]')
    for match in [re.search(r'pat_loss: ([0-9.eE+-]+)', line)]
    if match
]
if len(values) < 2 or not all(math.isfinite(value) for value in values):
    raise SystemExit('smoke training did not report finite patch losses')
audit = json.loads(audit_path.read_text())
cams = sorted(cam_dir.glob('*.npy'))
payload = {
    'status': 'pass',
    'patch_loss_observations': values,
    'patch_loss_decrease_observed': min(values[1:]) < values[0],
    'checkpoint_strict_audit_passed': bool(audit.get('passed')),
    'cam_files': len(cams),
    'cam_complete': (cam_dir / 'CAM_COMPLETE').is_file(),
}
if not payload['patch_loss_decrease_observed']:
    raise SystemExit('smoke patch loss did not decrease after its first observation')
if not payload['checkpoint_strict_audit_passed'] or not payload['cam_complete'] or len(cams) != 4:
    raise SystemExit(f'smoke checkpoint/CAM validation failed: {payload}')
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
print(json.dumps(payload, sort_keys=True))
PY
printf 'SMOKE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/smoke/SMOKE_COMPLETE"

tgca_train_root="$tgca_run_root/checkpoints/last_mct"
tgca_eval_root="$tgca_run_root/evaluations/last_mct"
printf 'STAGE=last_mct_train started=%s\n' "$(date --iso-8601=seconds)"
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
    --patch-final-norm --last-mct
tgca_checkpoint="$tgca_train_root/mctformerplus_final.pth"
sha256sum "$tgca_checkpoint" > "$tgca_run_root/checkpoint_manifest.txt"
tgca_source_log=$(find "$tgca_train_root/log_dir" -maxdepth 1 -type f \
    -name 'train-*.log' -print -quit)
cp --no-clobber "$tgca_source_log" "$tgca_run_root/training_logs/last_mct.log"
printf 'STAGE=last_mct_train finished=%s\n' "$(date --iso-8601=seconds)"

printf 'STAGE=last_mct_audit started=%s\n' "$(date --iso-8601=seconds)"
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
    tools/audit_mctformerplus_variant.py \
    --checkpoint "$tgca_checkpoint" --model mctformerplus \
    --official-pretrained "$tgca_pretrained" \
    --expected-pretrained-sha256 "$tgca_pretrained_sha" \
    --expected-epochs 45 --expected-effective-batch 32 --expected-seed 0 \
    --output "$tgca_run_root/audit/last_mct.json" \
    --patch-final-norm --last-mct

printf 'STAGE=last_mct_classification started=%s\n' "$(date --iso-8601=seconds)"
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python \
    tools/evaluate_mctformerplus_classification.py \
    --checkpoint "$tgca_checkpoint" --model mctformerplus \
    --voc-root "$tgca_voc_root" --list-path "$tgca_val" \
    --input-size 448 --batch-size 16 --num-workers 8 \
    --bootstrap-resamples 5000 --bootstrap-seed 20270906 \
    --output-dir "$tgca_eval_root/classification" \
    --patch-final-norm --last-mct

printf 'STAGE=last_mct_cam_train started=%s\n' "$(date --iso-8601=seconds)"
run_exact env CUDA_VISIBLE_DEVICES="$tgca_gpu" python make_cam.py \
    --dataset VOC12 --model mctformerplus \
    --voc12_root "$tgca_voc_root" --work_space "$tgca_eval_root" \
    --cam_out_dir cam_train --train_list "$tgca_train_cam" \
    --input_size 448 --scales 1.0,0.75,1.25 \
    --attention-normalization vanilla --bcss-variant e0 \
    --psl-variant baseline --checkpoint "$tgca_checkpoint" \
    --patch-final-norm --last-mct

printf 'STAGE=last_mct_cam_evaluation started=%s\n' "$(date --iso-8601=seconds)"
run_exact python tools/evaluate_cam_threshold_grid.py \
    --cam-dir "$tgca_eval_root/cam_train" --voc-root "$tgca_voc_root" \
    --id-list "$tgca_train_cam" \
    --output-dir "$tgca_eval_root/cam_evaluation" \
    --threshold-start 0 --threshold-stop 0.59 --threshold-step 0.01 \
    --fixed-threshold 0.45

printf 'STAGE=summary started=%s\n' "$(date --iso-8601=seconds)"
run_exact python tools/summarize_last_mct.py \
    --source-run-root "$tgca_source_root" \
    --patch-run-root "$tgca_patch_root" \
    --last-run-root "$tgca_run_root" \
    --expected-source-summary-sha256 "$tgca_source_summary_sha" \
    --expected-patch-summary-sha256 "$tgca_patch_summary_sha"

python - "$tgca_run_root/source_reference.json" <<'PY'
import hashlib, json, pathlib, sys
reference = json.loads(pathlib.Path(sys.argv[1]).read_text())
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()
for label, item in reference['files'].items():
    path = pathlib.Path(item['path'])
    if sha(path) != item['sha256']:
        raise SystemExit(f'immutable source changed during run: {label}: {path}')
for label, item in reference['checkpoints'].items():
    path = pathlib.Path(item['path'])
    if sha(path) != item['sha256']:
        raise SystemExit(f'immutable checkpoint changed during run: {label}: {path}')
print('immutable source hashes unchanged')
PY
printf 'PIPELINE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)" \
    | tee "$tgca_run_root/PIPELINE_COMPLETE"
