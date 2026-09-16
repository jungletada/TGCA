"""Wait for the four VOC runs, then evaluate all epoch archives without CAM dumps."""
import argparse
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.baselines.run_default_voc_coco import REPO, dataset_spec, run
from experiments.ablations.run_c2p_pooling import classification
from experiments.ablations.run_voc_epoch_checkpoints import MATRIX, pooling_flags
from tools.epoch_checkpoints import sha256


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def training_ready(source):
    for name in ('QUEUE_FAILED', 'QUEUE_CANCELLED'):
        if (source/name).exists():
            raise RuntimeError(f'Training did not complete: {source/name}')
    return (source/'QUEUE_COMPLETE').is_file()


def inventory(source):
    jobs = []
    for seed, method in MATRIX:
        name = f'{method}_s{seed}'
        directory = source/name
        assert (directory/'RUN_COMPLETE').is_file(), name
        assert (directory/'EPOCH_CHECKPOINTS_VERIFIED').is_file(), name
        rows = [json.loads(line) for line in (directory/'epoch_checkpoints/index.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in rows] == list(range(45)), name
        assert len(list((directory/'epoch_checkpoints').glob('*.pth'))) == 45
        for row in rows:
            epoch = row['epoch'] + 1
            assert row['filename'] == f'mctformerplus_epoch_{epoch:03d}.pth'
            checkpoint = directory/'epoch_checkpoints'/row['filename']
            assert checkpoint.is_file() and checkpoint.stat().st_size == row['bytes']
            jobs.append(dict(method=method, seed=seed, epoch=epoch, run=name,
                             checkpoint=str(checkpoint), checkpoint_sha256=row['sha256']))
    # Interleave methods/seeds at each epoch so partial curves are balanced.
    return sorted(jobs, key=lambda j: (j['epoch'], MATRIX.index((j['seed'], j['method']))))


def cam_command(spec, directory, job, cam_list=None):
    return [sys.executable, '-u', 'make_cam.py', '--dataset', 'VOC12', '--model', 'mctformerplus',
            '--voc12_root', str(spec['root']), '--work_space', str(directory),
            '--train_list', str(cam_list or spec['cam']), '--input_size', '448',
            '--scales', '1.0,0.75,1.25', '--checkpoint', job['checkpoint'],
            '--cam_out_dir', 'cam_train', '--online-raw-eval', '--online-mask-dir', str(spec['masks']),
            '--online-output-dir', str(directory/'raw_cam'), *pooling_flags(job['method'])]


def read_result(directory, job, counts=(1449, 1464)):
    cls = json.loads((directory/'classification/classification_metrics.json').read_text())
    cam = json.loads((directory/'raw_cam/metrics.json').read_text())
    assert (directory/'classification/CLASSIFICATION_COMPLETE').is_file()
    assert (directory/'raw_cam/EVAL_COMPLETE').is_file()
    assert cls['num_images'] == counts[0] and cam['num_images'] == counts[1] and cls['finite']
    assert cls['checkpoint']['sha256'] == job['checkpoint_sha256']
    assert cls['checkpoint']['epoch'] == job['epoch'] - 1
    assert cam['fixed']['threshold'] == .45
    branch = 'patch_gwrp' if job['method'] == 'gwrp' else 'patch_c2p'
    row = dict(job, class_macro_mAP_percent=cls['metrics_percent']['class_token']['macro_class_ap'],
               patch_macro_mAP_percent=cls['metrics_percent'][branch]['macro_class_ap'],
               class_val_loss=cls['classification_loss']['class_token_multilabel_soft_margin_mean'],
               patch_val_loss=cls['classification_loss'][branch+'_multilabel_soft_margin_mean'],
               classification_images=counts[0], cam_images=counts[1])
    for policy in ('fixed', 'best'):
        for field in ('mean_iou', 'semantic_foreground_precision', 'semantic_foreground_recall'):
            row[f'{policy}_{field}_percent'] = 100*cam[policy][field]
        row[f'{policy}_threshold'] = cam[policy]['threshold']
    assert all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
    return row


def cleanup_predictions(directory):
    """Delete only named disposable outputs inside this completed eval attempt.

    Never traverse source results/checkpoints. Keep metrics, per-class AP,
    threshold curves and small dataset-global confusion counts for audit.
    """
    record = directory/'result.json'
    assert record.is_file(), 'Persist validated metrics before cleanup'
    json.loads(record.read_text())
    removed = []
    for relative in ('classification/classification_predictions.npz',
                     'classification/classification_per_image.csv',
                     'classification/bootstrap_samples.npz'):
        path = directory/relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Cleanup target escaped evaluation directory')
        if path.is_file():
            removed.append(dict(path=relative, bytes=path.stat().st_size, sha256=sha256(path)))
    # Record the plan before unlinking; recoverable by reevaluating the retained checkpoint.
    audit_path = directory/'cleanup.json'
    previous = json.loads(audit_path.read_text())['removed'] if audit_path.exists() else []
    write_json(audit_path, dict(removed=previous+removed, checkpoint_retained=True,
                              recovery='Re-run inference from retained epoch checkpoint'))
    for row in removed:
        (directory/row['path']).unlink()
    cam_dir = directory/'cam_train'
    if cam_dir.is_dir() and not any(cam_dir.iterdir()):
        cam_dir.rmdir()


def evaluate_one(spec, directory, job, smoke=False):
    checkpoint = Path(job['checkpoint'])
    assert sha256(checkpoint) == job['checkpoint_sha256']
    started = time.time()
    cam_list = None
    if smoke:
        cam_list = directory/'cam_train_id.txt'
        cam_list.write_text('\n'.join(spec['cam'].read_text().splitlines()[:2])+'\n')
    run(cam_command(spec, directory, job, cam_list), directory, 'cam')
    method = job['method']
    classification(checkpoint, directory, spec, 'gwrp' if method=='gwrp' else 'c2p', 'classification',
                   limit=4 if smoke else 0, c2p_pooling_layers='last3' if method=='gwrp' else 'all',
                   c2p_pooling_reduction='mean' if method=='gwrp' else 'product')
    row = read_result(directory, job, counts=(4, 2) if smoke else (1449, 1464))
    assert sha256(checkpoint) == job['checkpoint_sha256'], 'Source checkpoint changed'
    row.update(evaluation_seconds=time.time()-started, evaluation_finished_utc=datetime.datetime.now(
        datetime.timezone.utc).isoformat(), scope='smoke' if smoke else 'full')
    write_json(directory/'result.json', row)
    cleanup_predictions(directory)
    (directory/'EPOCH_EVAL_COMPLETE').write_text('Metrics validated; source hash unchanged; cleanup complete\n')
    return row


PANELS = [
    ('class_macro_mAP_percent', 'Class-token macro mAP (%)'),
    ('patch_macro_mAP_percent', 'Patch-head macro mAP (%)'),
    ('class_val_loss', 'Class-token validation loss'),
    ('patch_val_loss', 'Patch-head validation loss'),
    ('fixed_mean_iou_percent', 'Raw CAM mIoU, fixed threshold 0.45 (%)'),
    ('best_mean_iou_percent', 'Raw CAM best-threshold mIoU (%) [diagnostic]'),
    ('fixed_semantic_foreground_precision_percent', 'Raw CAM foreground precision at 0.45 (%)'),
    ('fixed_semantic_foreground_recall_percent', 'Raw CAM foreground recall at 0.45 (%)'),
]


def summarize(output, rows):
    if not rows:
        return
    rows = sorted(rows, key=lambda r: (r['epoch'], r['seed'], r['method']))
    temporary = output/'epoch_metrics.csv.tmp'
    with temporary.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, output/'epoch_metrics.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)
    for ax, (key, label) in zip(axes.flat, PANELS):
        for seed, method in MATRIX:
            points = [r for r in rows if r['seed']==seed and r['method']==method]
            ax.plot([r['epoch'] for r in points], [r[key] for r in points],
                    color='tab:blue' if method=='gwrp' else 'tab:orange',
                    linestyle='-' if seed==0 else '--', marker='.' if len(points)<2 else None,
                    label=f'{method}, seed={seed}')
        ax.set_title(label); ax.set_xlim(1, 45); ax.grid(alpha=.25); ax.set_xlabel('Epoch')
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'VOC epoch trajectory: {len(rows)}/180 evaluated checkpoints')
    fig.tight_layout()
    for extension in ('svg', 'png'):
        fig.savefig(output/f'epoch_curves.{extension}', dpi=140)
    plt.close(fig)
    (output/'EPOCH_TRAJECTORY_REPORT.md').write_text(
        f'# VOC epoch evaluation\n\nCompleted: {len(rows)}/180 checkpoints.\n\n'
        'Classification: VOC val1449, FP32 single-scale448; class-token and patch-head macro-class AP and losses.\n'
        'Raw CAM: VOC train1464, native scales1/.75/1.25 plus flip, fixed threshold .45; no CRF/segmentation.\n'
        'Best-threshold results use the same diagnostic grid, not independently selected performance.\n'
        'X-axis is epoch, not wall-clock time. Four separate curves, no seed-uncertainty claims.\n\n'
        'See epoch_metrics.csv and epoch_curves.svg/png. Per-epoch JSON, per-class AP, threshold curves,\n'
        'aggregate confusion counts, exact commands/logs and cleanup records are retained.\n'
        'No per-image CAM arrays are written. Classification predictions/per-image AP are removed only\n'
        'after validated results are saved. All source checkpoints and training results are retained.\n')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    os.chdir(REPO)
    source, output = args.source.resolve(), args.output.resolve()
    assert source.is_dir() and output.is_relative_to(REPO/'results')
    assert not output.is_relative_to(source) and not source.is_relative_to(output)
    if not args.resume:
        output.mkdir(parents=True, exist_ok=False)
    assert output.is_dir()
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
            assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no']).strip()
            manifest_path = output/'manifest.json'
            if args.resume:
                manifest = json.loads(manifest_path.read_text())
                assert manifest['source'] == str(source) and manifest['evaluation_git_sha'] == sha
            else:
                manifest = dict(source=str(source), evaluation_git_sha=sha,
                                command=[sys.executable, *sys.argv], expected_checkpoints=180,
                                cleanup='Only newly generated per-image classification outputs; retain checkpoints',
                                status='waiting_for_training')
                write_json(manifest_path, manifest)
                run([sys.executable, '-m', 'pip', 'freeze'], output, 'pip_freeze')
                run([sys.executable, '-m', 'pytest', '-q', 'tests/test_voc_epoch_trajectory.py',
                     'tests/test_raw_cam_streaming.py'], output, 'tests')
            while not training_ready(source):
                if not args.wait:
                    raise RuntimeError('Training incomplete: use --wait')
                print(f'WAIT training completion {datetime.datetime.now().isoformat()}', flush=True)
                time.sleep(60)
            jobs = inventory(source)
            if 'jobs' in manifest:
                assert manifest['jobs'] == jobs
            manifest.update(jobs=jobs, status='evaluating',
                            training_manifest_sha256=sha256(source/'manifest.json'))
            write_json(manifest_path, manifest)
            (output/'WAIT_COMPLETE').write_text('All four training runs complete\n')
            spec = dataset_spec('VOC12')
            # Defer GPU smoke checks until training has finished.
            for method in ('gwrp', 'all_product'):
                job = next(j for j in jobs if j['method']==method and j['seed']==0)
                parent = output/(method+'_smoke')
                parent.mkdir(exist_ok=True)
                if not list(parent.glob('attempt_*/EPOCH_EVAL_COMPLETE')):
                    directory = parent/f'attempt_{len(list(parent.glob("attempt_*")))+1:03d}'
                    directory.mkdir(exist_ok=False)
                    evaluate_one(spec, directory, job, smoke=True)
            rows = []
            for job in jobs:
                parent = output/job['run']/f"epoch_{job['epoch']:03d}"
                parent.mkdir(parents=True, exist_ok=True)
                complete = sorted(parent.glob('attempt_*/EPOCH_EVAL_COMPLETE'))
                if complete:
                    assert len(complete)==1
                    row = json.loads((complete[0].parent/'result.json').read_text())
                    assert all(row[k]==v for k,v in job.items())
                    assert sha256(Path(job['checkpoint']))==job['checkpoint_sha256']
                else:
                    directory = parent/f'attempt_{len(list(parent.glob("attempt_*")))+1:03d}'
                    directory.mkdir(exist_ok=False)
                    print(f'EVALUATE {job["run"]} epoch {job["epoch"]}/45', flush=True)
                    row = evaluate_one(spec, directory, job)
                rows.append(row)
                summarize(output, rows)
                manifest.update(completed=len(rows), status='evaluating')
                write_json(manifest_path, manifest)
            assert len(rows)==180
            manifest.update(status='complete', all_evaluated_checkpoint_hashes_unchanged=True)
            write_json(manifest_path, manifest)
            (output/'QUEUE_COMPLETE').write_text('180 checkpoints evaluated; curves saved\n')
        except BaseException as exc:
            write_json(output/'QUEUE_FAILED.json', dict(error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
