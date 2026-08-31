#!/usr/bin/env bash
# Matched VOC seed-0 CTI-BGT validation; no threshold sweep or COCO expansion.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
if [[ "${CONDA_DEFAULT_ENV:-}" != "tgca-repro" ]]; then
    echo "Activate tgca-repro before running this script." >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "Refusing to start from a dirty worktree." >&2
    exit 2
fi
full_commit=$(git rev-parse HEAD)
if [[ "$full_commit" != "${CTI_BGT_EXPECTED_COMMIT:?Set the pushed launch commit}" ]]; then
    echo "Launch commit differs from the requested immutable revision." >&2
    exit 2
fi
gpu_id=${CTI_BGT_GPU_ID:-0}
seed=0
run_id=${CTI_BGT_RUN_ID:?Set a unique run ID}
run_dir="${CTI_BGT_OUTPUT_ROOT:-$repo_root/results/mctformerplus/voc}/$run_id"
voc_root=${CTI_BGT_VOC_ROOT:-$repo_root/data/VOCdevkit/VOC2012}
baseline_dir=${CTI_BGT_BASELINE_DIR:?Set the completed matched E0 run directory}
train_list="$voc_root/ImageLists/train_aug_id.txt"
val_list="$voc_root/ImageLists/val_id.txt"
cam_list="$voc_root/ImageLists/train_id.txt"
pretrain="$HOME/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth"
for required in "$train_list" "$val_list" "$cam_list" "$pretrain" \
    "$baseline_dir/raw_cam_diagnostics/metrics.json" "$baseline_dir/checkpoint_manifest.txt"; do
    [[ -f "$required" ]] || { echo "Missing input: $required" >&2; exit 2; }
done
if [[ -e "$run_dir" ]]; then
    echo "Refusing to overwrite run directory: $run_dir" >&2
    exit 2
fi
mkdir -p "$run_dir"
exec > >(tee -a "$run_dir/pipeline.log") 2>&1
trap 'exit_code=$?; printf "{\"exit_code\":%s,\"finished\":\"%s\"}\n" "$exit_code" "$(date --iso-8601=seconds)" > "$run_dir/exit_status.json"' EXIT
printf 'run_id=%s\nrun_dir=%s\nsource_checkout=%s\ncommit=%s\n' "$run_id" "$run_dir" "$repo_root" "$full_commit"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONUNBUFFERED=1
printf 'CTI_BGT_EXPECTED_COMMIT=%q CTI_BGT_RUN_ID=%q CTI_BGT_OUTPUT_ROOT=%q CTI_BGT_VOC_ROOT=%q CTI_BGT_BASELINE_DIR=%q CTI_BGT_GPU_ID=%q bash %q\n' \
    "$full_commit" "$run_id" "$(dirname "$run_dir")" "$voc_root" "$baseline_dir" "$gpu_id" \
    "$repo_root/experiments/baselines/run_cti_bgt_voc.sh" > "$run_dir/command.txt"
cp "$0" "$run_dir/runner.sh"
python - "$run_dir" "$full_commit" "$repo_root" "$baseline_dir" <<'META'
import json, subprocess, sys
from pathlib import Path
run, commit, checkout, baseline = sys.argv[1:]
run = Path(run)
(run / 'git_state.json').write_text(json.dumps({
    'commit': commit, 'source_checkout': checkout, 'dirty': False,
    'published_branch': 'research/mctformerplus-cti-bgt',
    'cti_reference': subprocess.check_output(['git', 'rev-parse', 'HEAD:hosts/CTI'], text=True).strip(),
}, indent=2) + '\n')
(run / 'config.json').write_text(json.dumps({
    'model': 'mctformerplus', 'dataset': 'VOC12', 'seed': 0, 'epochs': 45,
    'input_size': 448, 'batch_size': 32, 'lr': 5e-4, 'min_lr': 1e-5,
    'cti_bgt': {'enabled': True, 'weight': 0.1, 'n_layers': 6, 'affinity_start': 4},
    'attention_normalization': 'vanilla', 'bcss_variant': 'e0', 'psl_variant': 'baseline',
    'cam_scales': [1.0, 0.75, 1.25], 'fixed_background_threshold': 0.45,
    'cam_split': 'train_id (1464)', 'classification_validation_split': 'val_id (1449)',
    'checkpoint_policy': 'final epoch', 'baseline_dir': baseline,
    'postprocessing': 'host CAM normalization; no CRF',
}, indent=2) + '\n')
META
python --version > "$run_dir/environment.txt" 2>&1
python -c "import torch,torchvision,timm; print('torch',torch.__version__,'torchvision',torchvision.__version__,'timm',timm.__version__,'cuda',torch.version.cuda)" >> "$run_dir/environment.txt"
python -m pip freeze > "$run_dir/pip_freeze.txt"
"${CONDA_EXE:?Activated Conda must provide CONDA_EXE}" list --explicit > "$run_dir/conda_explicit.txt"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader > "$run_dir/hardware.txt"
sha256sum "$train_list" "$val_list" "$cam_list" "$voc_root/ImageLabel/cls_labels.npy" > "$run_dir/dataset_manifest.txt"
sha256sum "$pretrain" > "$run_dir/pretrained_manifest.txt"
sha256sum -c "$baseline_dir/checkpoint_manifest.txt" > "$run_dir/baseline_checkpoint_check.txt"

stage() {
    if [[ "$(git rev-parse HEAD)" != "$full_commit" || -n "$(git status --porcelain)" ]]; then
        echo "Source checkout changed during the experiment; stopping." >&2
        exit 2
    fi
    printf 'STAGE=%s started=%s\n' "$1" "$(date --iso-8601=seconds)"
}
stage tests
OMP_NUM_THREADS=1 python -m pytest -q tests > "$run_dir/tests.log" 2>&1
stage train
python train_model_v2.py \
    --dataset VOC12 --model mctformerplus --voc12_root "$voc_root" \
    --train_list "$train_list" --val_list "$val_list" \
    --work_space "$run_dir" --input-size 448 --epochs 45 --batch_size 32 \
    --seed "$seed" --lr 5e-4 --min-lr 1e-5 --num_workers 10 \
    --attention-normalization vanilla --bcss-variant e0 --psl-variant baseline \
    --cti-bgt --cti-bgt-weight 0.1 --cti-bgt-n-layers 6 --cti-bgt-affinity-start 4
checkpoint="$run_dir/mctformerplus_final.pth"
sha256sum "$checkpoint" > "$run_dir/checkpoint_manifest.txt"
stage cam
python make_cam.py \
    --dataset VOC12 --model mctformerplus --voc12_root "$voc_root" \
    --work_space "$run_dir" --cam_out_dir cam_train --train_list "$cam_list" \
    --input_size 448 --scales 1.0,0.75,1.25 --checkpoint "$checkpoint" \
    --attention-normalization vanilla --bcss-variant e0 --psl-variant baseline \
    --cti-bgt --cti-bgt-weight 0.1 --cti-bgt-n-layers 6 --cti-bgt-affinity-start 4
stage fixed_threshold_metrics
python tools/collect_cam_metrics.py \
    --cam-dir "$run_dir/cam_train" --voc-root "$voc_root" --id-list "$cam_list" \
    --threshold 0.45 --output-dir "$run_dir/raw_cam_diagnostics"
stage compare
python - "$run_dir" "$baseline_dir" "$cam_list" <<'COMPARE'
import csv, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
run, baseline, image_list = map(Path, sys.argv[1:])
ids = {line.strip() for line in image_list.read_text().splitlines() if line.strip()}
assert len(ids) == 1464
assert {p.stem for p in (run / 'cam_train').glob('*.npy')} == ids
current = json.loads((run / 'raw_cam_diagnostics/metrics.json').read_text())
reference = json.loads((baseline / 'raw_cam_diagnostics/metrics.json').read_text())
for data in (current, reference):
    assert data['num_images'] == 1464 and data['background_threshold'] == 0.45
    assert data['provenance']['id_list_sha256'] == hashlib.sha256(image_list.read_bytes()).hexdigest()
keys = ['mean_iou_percent', 'semantic_foreground_precision_percent',
        'semantic_foreground_recall_percent', 'background_false_positive_rate_percent']
rows = [{'metric': k, 'baseline': reference[k], 'cti_bgt': current[k],
         'delta': current[k] - reference[k]} for k in keys]
summary = {'baseline_dir': str(baseline), 'cti_bgt_dir': str(run),
           'fixed_threshold': 0.45, 'seed': 0, 'metrics': rows,
           'interpretation': 'single-seed matched screen; no automatic further experiments'}
(run / 'comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
with (run / 'comparison.csv').open('w') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(run / 'completion.json').write_text(json.dumps({
    'complete': True, 'num_cams': len(ids), 'finished': datetime.now(timezone.utc).isoformat(),
}) + '\n')
print(json.dumps(summary))
COMPARE
printf 'PIPELINE_COMPLETE finished=%s\n' "$(date --iso-8601=seconds)"
