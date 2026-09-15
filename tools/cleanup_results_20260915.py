"""One-off user-authorized cleanup. Inventory first; unlink only exact reviewed files.

No recursive deletion, no symlink traversal, no changes to live COCO dependencies.
The plan records sizes/identity; retained static files are hashed before/after.
"""
import argparse
from collections import Counter
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

REPO = Path('/home/peng/code/TGCA')
ROOT = REPO / 'results'
ACTIVE = ROOT / 'c2p_pooling/20260915-coco-all-product-s0-r2'
AUDIT = ROOT / 'cleanup/20260915-keep-checkpoints-final'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns]


def safe_path(relative):
    path = ROOT / relative
    if not relative or '..' in Path(relative).parts or path == ROOT:
        raise ValueError('Invalid exact target')
    if path.resolve(strict=True) != path or ROOT not in path.parents:
        raise ValueError('Outside root or symlink target: ' + str(path))
    return path


def plan():
    assert ROOT.resolve() == ROOT and ROOT.is_dir()
    assert not AUDIT.exists()
    assert not subprocess.check_output(['git', 'ls-files', 'results'], cwd=REPO).strip()
    active_manifest = json.loads((ACTIVE / 'manifest.json').read_text())
    sources = active_manifest['source_sha256_before']
    for path, sha in sources.items():
        assert digest(Path(path)) == sha
    assert (ACTIVE / 'COCO_smoke/ONLINE_PARITY_COMPLETE').exists()
    assert (ACTIVE / 'baseline_smoke_classification/CLASSIFICATION_COMPLETE').exists()
    assert (ACTIVE / 'COCO/train.log').exists()  # queue has passed ALL smoke reads
    files, smoke_dirs = [], []
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        for name in dirs:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError('Review symlink before cleanup: ' + str(path))
            if 'smoke' in name.lower():
                smoke_dirs.append(path)
        for name in names:
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError('Review symlink before cleanup: ' + str(path))
            files.append(path)
    smoke_roots = [p for p in smoke_dirs if not any(q in p.parents for q in smoke_dirs)]
    targets, retained = [], {}
    for path in files:
        reason = None
        if any(p in path.parents for p in smoke_roots) or 'smoke' in path.name.lower():
            reason = 'completed_smoke'
        elif path == ACTIVE or ACTIVE in path.parents or str(path) in sources:
            pass
        elif path.suffix == '.npy' and path.parent.name == 'cam_train':
            reason = 'per_image_CAM'
        elif path.name.endswith('_positive_attention.npz'):
            reason = 'raw_attention_tensor'
        if reason:
            assert str(path) not in sources
            assert not (ACTIVE / 'COCO' in path.parents)
            targets.append([str(path.relative_to(ROOT)), identity(path), reason])
        elif not (path == ACTIVE or ACTIVE in path.parents or path == Path(str(ACTIVE) + '.queue.log')):
            retained[str(path.relative_to(ROOT))] = digest(path)
    smoke_all_dirs = []
    for path in smoke_roots:
        smoke_all_dirs.extend(str(Path(base).relative_to(ROOT)) for base, _, _ in os.walk(path))
    counts = Counter(t[2] for t in targets)
    byte_counts = {reason: sum(t[1][2] for t in targets if t[2] == reason) for reason in counts}
    payload = dict(targets=targets, smoke_directories=sorted(smoke_all_dirs, key=lambda p: -len(Path(p).parts)),
                   retained_sha256=retained, live_source_sha256=sources)
    AUDIT.mkdir(parents=True, exist_ok=False)
    with gzip.open(AUDIT / 'exact_plan.json.gz', 'wt') as stream:
        json.dump(payload, stream, separators=(',', ':'))
    summary = dict(created=datetime.datetime.now().isoformat(), root=str(ROOT),
        authorisation='2026-09-15 latest user: retain non-smoke checkpoints; delete all completed smoke directories and per-image NPY/CAM intermediates',
        counts=dict(counts), bytes_by_reason=byte_counts, total_bytes=sum(byte_counts.values()),
        smoke_roots=[str(p.relative_to(ROOT)) for p in smoke_roots],
        retained_static_files=len(retained), live_root=str(ACTIVE),
        live_source_files=list(sources), disk_free_before=shutil.disk_usage(ROOT).free,
        code_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        historical_integrity_notice='Old manifests stay unchanged; deleted CAM/attention/smoke paths no longer exist. Non-smoke checkpoints and all retained results still verify.',
        recovery='All non-smoke checkpoints retained. No backup of deleted payloads; CAM/attention can be regenerated by inference, smoke requires rerunning.',
        plan_sha256=digest(AUDIT / 'exact_plan.json.gz'))
    (AUDIT / 'summary_before.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def apply():
    summary = json.loads((AUDIT / 'summary_before.json').read_text())
    assert digest(AUDIT / 'exact_plan.json.gz') == summary['plan_sha256']
    assert not (AUDIT / 'deletions.jsonl.gz').exists()
    with gzip.open(AUDIT / 'exact_plan.json.gz', 'rt') as f:
        planned = json.load(f)
    for relative, expected, _ in planned['targets']:
        path = safe_path(relative)
        assert path.is_file() and identity(path) == expected, str(path)
    for relative, sha in planned['retained_sha256'].items():
        assert digest(safe_path(relative)) == sha, relative
    removed, total = 0, 0
    with gzip.open(AUDIT / 'deletions.jsonl.gz', 'xt') as log:
        for relative, expected, reason in planned['targets']:
            path = safe_path(relative)
            assert identity(path) == expected, str(path)
            path.unlink()
            log.write(json.dumps([relative, expected[2], reason]) + '\n')
            removed += 1
            total += expected[2]
            if removed % 10000 == 0:
                log.flush()
                print(f'deleted {removed} exact files, {total / 1024**3:.2f} GiB logical', flush=True)
    for relative in planned['smoke_directories']:
        safe_path(relative).rmdir()  # empty directories only; unexpected new file aborts
    for relative, sha in planned['retained_sha256'].items():
        assert digest(safe_path(relative)) == sha, relative
    for absolute, sha in planned['live_source_sha256'].items():
        assert digest(Path(absolute)) == sha, absolute
    assert not any((ROOT / p).exists() for p in summary['smoke_roots'])
    result = dict(completed=datetime.datetime.now().isoformat(), deleted_files=removed,
                  deleted_logical_bytes=total, removed_smoke_roots=len(summary['smoke_roots']),
                  retained_static_hashes_unchanged=True, live_dependency_hashes_unchanged=True,
                  disk_free_after=shutil.disk_usage(ROOT).free)
    (AUDIT / 'summary_after.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['plan', 'apply'])
    args = parser.parse_args()
    plan() if args.stage == 'plan' else apply()
