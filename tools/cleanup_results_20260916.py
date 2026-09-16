"""Authorized post-cancellation cleanup: review exact plan, then apply it.

Retain checkpoints, numeric metrics, aggregate diagnostic arrays and logs.
No symlink traversal or recursive deletion. Every retained file is hash checked.
"""
import argparse
import json
import os
from pathlib import Path
from tools.cleanup_results_20260915 import ROOT, digest, identity, safe_path

AUDIT = ROOT / 'cleanup/20260916-stop-diagnostics'


def plan():
    targets, retained = [], {}
    smoke_dirs = []
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        for name in dirs + names:
            if (Path(base) / name).is_symlink():
                raise RuntimeError('Review symlink: ' + str(Path(base) / name))
        if 'smoke' in str(Path(base).relative_to(ROOT)).lower():
            smoke_dirs.append(str(Path(base).relative_to(ROOT)))
        for name in names:
            path = Path(base) / name
            relative = str(path.relative_to(ROOT))
            reason = None
            if 'smoke' in relative.lower():
                reason = 'smoke_artifact'
            elif path.name == 'classification_predictions.npz':
                assert (path.parent / 'classification_metrics.json').exists()
                reason = 'per_image_classification_arrays'
            elif path.suffix in {'.npy', '.npz'} and 'cam_train' in path.parts:
                reason = 'per_image_CAM'
            elif path.name.endswith('_positive_attention.npz'):
                reason = 'raw_attention_tensor'
            if reason:
                targets.append([relative, identity(path), reason])
            else:
                retained[relative] = digest(path)
    AUDIT.mkdir(parents=True, exist_ok=False)
    payload = dict(targets=targets, retained_sha256=retained,
                   smoke_dirs=sorted(smoke_dirs, key=lambda x: -len(Path(x).parts)))
    (AUDIT / 'exact_plan.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(dict(targets=targets, bytes=sum(t[1][2] for t in targets),
                         retained_files=len(retained)), indent=2))


def apply():
    assert not (AUDIT / 'summary_after.json').exists()
    payload = json.loads((AUDIT / 'exact_plan.json').read_text())
    for relative, expected, _ in payload['targets']:
        assert identity(safe_path(relative)) == expected
    for relative, sha in payload['retained_sha256'].items():
        assert digest(safe_path(relative)) == sha, relative
    with (AUDIT / 'deletions.jsonl').open('x') as log:
        for relative, expected, reason in payload['targets']:
            path = safe_path(relative)
            assert identity(path) == expected
            path.unlink()
            log.write(json.dumps(dict(path=relative, bytes=expected[2], reason=reason)) + '\n')
    for relative in payload['smoke_dirs']:
        safe_path(relative).rmdir()
    for relative, sha in payload['retained_sha256'].items():
        assert digest(safe_path(relative)) == sha, relative
    summary = dict(deleted_files=len(payload['targets']),
                   deleted_bytes=sum(t[1][2] for t in payload['targets']),
                   retained_files=len(payload['retained_sha256']),
                   retained_hashes_unchanged=True,
                   recovery='No payload backup; regenerate predictions from retained checkpoints if needed.')
    (AUDIT / 'summary_after.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['plan', 'apply'])
    args = parser.parse_args()
    plan() if args.stage == 'plan' else apply()
