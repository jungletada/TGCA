#!/usr/bin/env python3
"""Capture deterministic MCTformer+ logits/CAMs and compare a reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets_cam import build_transform  # noqa: E402
from models.mctformer_plus import (  # noqa: E402
    build_mctformerplus,
    model_spec_from_instance,
    resolve_mctformerplus_checkpoint_variant,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument(
        '--model', choices=(
            'mctformerplus_tiny', 'mctformerplus', 'mctformerplus_base'),
        required=True,
    )
    parser.add_argument('--voc-root', type=Path, required=True)
    parser.add_argument('--image-id', default='2007_000033')
    parser.add_argument('--input-size', type=int, default=448)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--output-stem', type=Path, required=True)
    parser.add_argument('--reference-npz', type=Path)
    parser.add_argument('--tolerance', type=float, default=1e-6)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array):
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order='C')
    ).hexdigest()


def capture(args):
    npz_path = args.output_stem.with_suffix('.npz')
    json_path = args.output_stem.with_suffix('.json')
    command_path = args.output_stem.parent / f'{args.output_stem.name}_command.txt'
    existing = [path for path in (npz_path, json_path, command_path) if path.exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite regression artifacts: {existing}')
    if args.tolerance < 0:
        raise ValueError('--tolerance must be non-negative')
    if args.input_size < 1 or args.input_size % 16:
        raise ValueError('--input-size must be a positive multiple of 16')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, args.model
    )
    model = build_mctformerplus(
        resolution['variant'],
        cam=True,
        num_classes=20,
        input_size=args.input_size,
        attention_normalization='vanilla',
        attention_gamma=1.0,
        bcss_variant='e0',
        psl_variant='baseline',
        cti_bgt=False,
    )
    state = checkpoint.get('model', checkpoint)
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f'Unexpected strict-load result: {incompatibility}')
    model.to(device).eval()

    image_path = args.voc_root / 'JPEGImages' / f'{args.image_id}.jpg'
    image = PIL.Image.open(image_path).convert('RGB')
    tensor = build_transform(
        False, False, argparse.Namespace(input_size=args.input_size)
    )(image).unsqueeze(0).contiguous()
    input_hash = sha256_array(tensor.numpy())
    with torch.inference_mode():
        x_cls, x_patch, attentions, _, auxiliary = model.forward_features(
            tensor.to(device), return_aux=True
        )
        batch, patch_count, channels = x_patch.shape
        side = int(patch_count ** 0.5)
        if side * side != patch_count:
            raise RuntimeError(f'Non-square patch count: {patch_count}')
        patch_grid = x_patch.reshape(
            batch, side, side, channels
        ).permute(0, 3, 1, 2).contiguous()
        raw_patch_cam = model.head(patch_grid)
        mean_attention = torch.stack(attentions).mean(dim=2)
        outputs = {
            'class_logits': x_cls.mean(dim=-1),
            'patch_logits': model.gwrp(
                raw_patch_cam[:, model._foreground_slice()]
            ),
            'patch_cam': torch.relu(
                raw_patch_cam[:, model._foreground_slice()]
            ),
            'class_to_patch': model.get_cls2pat(
                raw_patch_cam, mean_attention, auxiliary
            ),
            'final_cam': model.get_cam(
                raw_patch_cam, mean_attention, auxiliary
            ),
        }
    arrays = {
        key: value.detach().cpu().numpy() for key, value in outputs.items()
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Captured regression output contains NaN/Inf')

    comparison = None
    if args.reference_npz is not None:
        with np.load(args.reference_npz, allow_pickle=False) as reference:
            if set(reference.files) != set(arrays):
                raise AssertionError(
                    f'Regression keys differ: reference={reference.files}, '
                    f'observed={sorted(arrays)}'
                )
            details = {}
            maximum = 0.0
            for key, observed in arrays.items():
                expected = reference[key]
                if observed.shape != expected.shape or observed.dtype != expected.dtype:
                    raise AssertionError(
                        f'{key} metadata differs: {observed.shape}/{observed.dtype} '
                        f'!= {expected.shape}/{expected.dtype}'
                    )
                difference = float(np.max(np.abs(observed - expected)))
                details[key] = {
                    'max_abs_diff': difference,
                    'shape_equal': True,
                    'dtype_equal': True,
                }
                maximum = max(maximum, difference)
            comparison = {
                'reference_npz': str(args.reference_npz.resolve()),
                'reference_sha256': sha256_file(args.reference_npz),
                'tolerance': args.tolerance,
                'max_abs_diff': maximum,
                'outputs': details,
                'passed': maximum <= args.tolerance,
            }
            if not comparison['passed']:
                raise AssertionError(
                    f'Regression max_abs_diff={maximum} exceeds {args.tolerance}'
                )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    metadata = {
        'schema_version': 1,
        'artifact_role': 'postchange_numerical_regression',
        'repository': str(REPO_ROOT),
        'commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True
        ).strip(),
        'model_spec': model_spec_from_instance(model),
        'variant_resolution': resolution,
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': sha256_file(args.checkpoint),
            'state_dict_key_count': len(state),
            'strict_load_passed': True,
            'epoch': checkpoint.get('epoch'),
        },
        'input': {
            'image_id': args.image_id,
            'image_path': str(image_path.resolve()),
            'image_sha256': sha256_file(image_path),
            'original_pil_size_wh': list(image.size),
            'input_size': args.input_size,
            'tensor_shape': list(tensor.shape),
            'tensor_dtype': str(tensor.numpy().dtype),
            'tensor_sha256': input_hash,
        },
        'execution': {
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'device': str(device),
            'seed': args.seed,
            'eval': True,
            'amp': False,
            'dtype': 'float32',
            'cudnn_benchmark': False,
            'cudnn_deterministic': True,
            'deterministic_algorithms': True,
        },
        'outputs': {
            key: {
                'shape': list(value.shape),
                'dtype': str(value.dtype),
                'sha256': sha256_array(value),
                'minimum': float(value.min()),
                'maximum': float(value.max()),
                'mean': float(value.mean()),
                'finite': True,
            }
            for key, value in arrays.items()
        },
        'comparison': comparison,
    }
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    command_path.write_text(
        shlex.join([sys.executable] + sys.argv) + '\n', encoding='utf-8'
    )
    print(json.dumps(metadata, sort_keys=True))
    return metadata


def main():
    capture(parse_args())


if __name__ == '__main__':
    main()
