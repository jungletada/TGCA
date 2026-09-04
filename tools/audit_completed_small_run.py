#!/usr/bin/env python3
"""Read-only audit and pointer capture for the canonical completed Small run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import (  # noqa: E402
    build_mctformerplus,
    resolve_mctformerplus_checkpoint_variant,
)


REQUIRED_FILES = (
    'mctformerplus_final.pth', 'checkpoint_manifest.txt',
    'dataset_manifest.txt', 'git_state.json', 'environment.txt',
    'hardware.txt', 'pipeline.log', 'metrics.json', 'config.json',
    'command.txt', 'pretrained_manifest.txt',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def execute(args):
    output_paths = (
        args.output_dir / 'small_run_audit.json',
        args.output_dir / 'small_run_pointer.json',
    )
    if any(path.exists() for path in output_paths):
        raise FileExistsError(f'Refusing to overwrite: {output_paths}')
    run = args.run_dir.expanduser().resolve()
    voc = args.voc_root.expanduser().resolve()
    issues = []

    def require(condition, code, observed=None, expected=None):
        if not condition:
            issues.append({
                'code': code, 'observed': observed, 'expected': expected
            })

    for name in REQUIRED_FILES:
        require((run / name).is_file(), f'missing:{name}')
    require((run / 'cam_train').is_dir(), 'missing:cam_train')
    if issues:
        raise FileNotFoundError(issues)

    checkpoint_path = run / 'mctformerplus_final.pth'
    checkpoint_hash = sha256_file(checkpoint_path)
    manifest_match = re.match(
        r'^([0-9a-f]{64})\s+',
        (run / 'checkpoint_manifest.txt').read_text().strip(),
    )
    require(
        manifest_match is not None and manifest_match.group(1) == checkpoint_hash,
        'checkpoint_manifest_mismatch', checkpoint_hash,
        manifest_match.group(1) if manifest_match else None,
    )
    pipeline = (run / 'pipeline.log').read_text(errors='replace')
    require('PIPELINE_COMPLETE' in pipeline, 'pipeline_incomplete')
    metrics = json.loads((run / 'metrics.json').read_text())
    config = json.loads((run / 'config.json').read_text())
    git_state = json.loads((run / 'git_state.json').read_text())
    expected_scalar = {
        'seed': (metrics.get('seed'), 0),
        'input_size': (metrics.get('input_size'), 448),
        'epochs': (metrics.get('epochs'), 45),
        'normalization': (metrics.get('normalization'), 'vanilla'),
        'config_mode': (config.get('mode'), 'vanilla'),
        'training_dirty': (git_state.get('dirty'), False),
    }
    for name, (observed, expected) in expected_scalar.items():
        require(observed == expected, name, observed, expected)

    cam_ids = _ids(voc / 'ImageLists/train_id.txt')
    cam_files = sorted((run / 'cam_train').glob('*.npy'))
    require(
        len(cam_files) == len(cam_ids), 'cam_count', len(cam_files), len(cam_ids)
    )
    require(
        {path.stem for path in cam_files} == set(cam_ids), 'cam_id_set'
    )

    manifest_hashes = {}
    for line in (run / 'dataset_manifest.txt').read_text().splitlines():
        match = re.match(r'^([0-9a-f]{64})\s+(.+)$', line)
        if match:
            manifest_hashes[Path(match.group(2)).name] = match.group(1)
    dataset_paths = (
        voc / 'ImageLists/train_aug_id.txt', voc / 'ImageLists/val_id.txt',
        voc / 'ImageLists/train_id.txt', voc / 'ImageLabel/cls_labels.npy',
    )
    for path in dataset_paths:
        actual = sha256_file(path)
        require(
            actual == manifest_hashes.get(path.name),
            f'dataset_hash:{path.name}', actual, manifest_hashes.get(path.name),
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, 'mctformerplus'
    )
    model = build_mctformerplus(
        'small', cam=True, num_classes=20, input_size=448,
        attention_normalization='vanilla', bcss_variant='e0',
        psl_variant='baseline', cti_bgt=False,
    )
    incompatibility = model.load_state_dict(checkpoint['model'], strict=True)
    require(
        not incompatibility.missing_keys and not incompatibility.unexpected_keys,
        'strict_checkpoint_load',
    )
    require(checkpoint.get('epoch') == 44, 'checkpoint_epoch', checkpoint.get('epoch'), 44)

    pretrain_match = re.match(
        r'^([0-9a-f]{64})\s+(.+)$',
        (run / 'pretrained_manifest.txt').read_text().strip(),
    )
    pretrain_path = Path(pretrain_match.group(2)).resolve() if pretrain_match else None
    pretrain_expected = pretrain_match.group(1) if pretrain_match else None
    pretrain_actual = (
        sha256_file(pretrain_path)
        if pretrain_path is not None and pretrain_path.is_file() else None
    )
    require(pretrain_actual == pretrain_expected, 'pretrained_hash')

    status = (
        'fail' if issues else (
            'pass_with_legacy_incomplete_provenance'
            if resolution['legacy_small_import'] else 'pass'
        )
    )
    audit = {
        'schema_version': 1,
        'scope': 'canonical completed MCTformer+-Small seed-0 run',
        'status': status,
        'passed': not issues,
        'run_dir': str(run),
        'checkpoint_sha256': checkpoint_hash,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'variant_resolution': resolution,
        'legacy_missing_checkpoint_metadata': [
            key for key in ('model_spec', 'pretrained', 'training_spec')
            if key not in checkpoint
        ],
        'metrics': metrics,
        'config': config,
        'training_git_state': git_state,
        'cam_count': len(cam_files),
        'expected_cam_count': len(cam_ids),
        'issues': issues,
        'source_immutability': {
            'mode': 'read_only',
            'checkpoint_copied': False,
            'cam_files_copied': False,
            'source_results_modified': False,
        },
        'command': shlex.join([sys.executable] + sys.argv),
    }
    listing = '\n'.join(
        f'{path.name}\t{path.stat().st_size}' for path in cam_files
    ).encode()
    pointer = {
        'schema_version': 1,
        'run_dir': str(run),
        'checkpoint': {
            'path': str(checkpoint_path), 'sha256': checkpoint_hash
        },
        'cam_dir': {
            'path': str((run / 'cam_train').resolve()),
            'file_count': len(cam_files),
            'filename_size_listing_sha256': hashlib.sha256(listing).hexdigest(),
        },
        'files': {
            name: {
                'path': str((run / name).resolve()),
                'sha256': sha256_file(run / name),
            }
            for name in REQUIRED_FILES
        },
        'dataset_inputs': {
            path.name: {'path': str(path), 'sha256': sha256_file(path)}
            for path in dataset_paths
        },
        'official_pretrained': {
            'path': str(pretrain_path), 'sha256': pretrain_actual
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths[0].write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + '\n'
    )
    output_paths[1].write_text(
        json.dumps(pointer, indent=2, sort_keys=True, allow_nan=False) + '\n'
    )
    print(json.dumps(audit, sort_keys=True))
    if issues:
        raise SystemExit(1)
    return audit, pointer


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
