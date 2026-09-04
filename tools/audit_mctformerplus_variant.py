#!/usr/bin/env python3
"""Audit one frozen MCTformer+ width checkpoint without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import (  # noqa: E402
    adapt_deit_checkpoint_for_mctformerplus,
    build_mctformerplus,
    get_mctformerplus_spec,
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
    parser.add_argument('--official-pretrained', type=Path, required=True)
    parser.add_argument('--expected-checkpoint-sha256')
    parser.add_argument('--expected-pretrained-sha256')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--input-size', type=int, default=448)
    parser.add_argument('--expected-epochs', type=int, default=45)
    parser.add_argument('--expected-effective-batch', type=int, default=32)
    parser.add_argument('--expected-seed', type=int, default=0)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def execute(args):
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    for path in (args.checkpoint, args.official_pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_hash = sha256_file(args.checkpoint)
    pretrained_hash = sha256_file(args.official_pretrained)
    if (args.expected_checkpoint_sha256 is not None
            and checkpoint_hash != args.expected_checkpoint_sha256):
        raise ValueError('Checkpoint SHA-256 differs from expected value')
    if (args.expected_pretrained_sha256 is not None
            and pretrained_hash != args.expected_pretrained_sha256):
        raise ValueError('Official pretrained SHA-256 differs from expected value')

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    resolution = resolve_mctformerplus_checkpoint_variant(
        checkpoint, args.model
    )
    variant = resolution['variant']
    spec = get_mctformerplus_spec(variant)
    training_model = build_mctformerplus(
        variant, num_classes=20, input_size=args.input_size,
        attention_normalization='vanilla', bcss_variant='e0',
        psl_variant='baseline', cti_bgt=False,
    )
    cam_model = build_mctformerplus(
        variant, cam=True, num_classes=20, input_size=args.input_size,
        attention_normalization='vanilla', bcss_variant='e0',
        psl_variant='baseline', cti_bgt=False,
    )
    state = checkpoint.get('model', checkpoint)
    training_result = training_model.load_state_dict(state, strict=True)
    cam_result = cam_model.load_state_dict(state, strict=True)
    if any((training_result.missing_keys, training_result.unexpected_keys,
            cam_result.missing_keys, cam_result.unexpected_keys)):
        raise RuntimeError('Strict checkpoint load unexpectedly returned mismatches')
    nonfinite_state = [
        key for key, value in state.items()
        if value.is_floating_point() and not torch.isfinite(value).all()
    ]
    if nonfinite_state:
        raise ValueError(f'Non-finite checkpoint tensors: {nonfinite_state}')

    source = torch.load(args.official_pretrained, map_location='cpu')
    adapted, pretrained_report = adapt_deit_checkpoint_for_mctformerplus(
        source, training_model, num_classes=20
    )
    pretrained_strict = training_model.load_state_dict(adapted, strict=True)
    if pretrained_strict.missing_keys or pretrained_strict.unexpected_keys:
        raise RuntimeError('Adapted official pretrained strict load failed')
    pretrained_report.update({
        'cache_path': str(args.official_pretrained.resolve()),
        'source_sha256': pretrained_hash,
        'source_url': spec['pretrained_url'],
    })

    attention = checkpoint.get('attention_normalization', {})
    bcss = checkpoint.get('bcss', {'variant': 'e0'})
    psl = checkpoint.get('psl', {'variant': 'baseline'})
    cti = checkpoint.get('cti_bgt', {'enabled': False})
    method_checks = {
        'attention_vanilla': attention.get('mode', 'vanilla') == 'vanilla',
        'attention_gamma_one': float(attention.get('gamma', 1.0)) == 1.0,
        'attention_relation_bias_disabled': not bool(
            attention.get('relation_bias', False)
        ),
        'bcss_e0': bcss.get('variant', 'e0') == 'e0',
        'psl_baseline': psl.get('variant', 'baseline') == 'baseline',
        'cti_bgt_disabled': not bool(cti.get('enabled', False)),
    }
    modern = not resolution['legacy_small_import']
    checkpoint_pretrained = checkpoint.get('pretrained', {})
    checkpoint_training = checkpoint.get('training_spec', {})
    modern_checks = {
        'model_spec_present': isinstance(checkpoint.get('model_spec'), dict),
        'pretrained_metadata_present': isinstance(
            checkpoint.get('pretrained'), dict
        ),
        'training_spec_present': isinstance(
            checkpoint.get('training_spec'), dict
        ),
        'pretrained_sha_matches': (
            checkpoint_pretrained.get('sha256') == pretrained_hash
            if modern else True
        ),
        'pretrained_url_matches': (
            checkpoint_pretrained.get('url') == spec['pretrained_url']
            if modern else True
        ),
        'pretrained_filename_matches': (
            checkpoint_pretrained.get('filename')
            == Path(spec['pretrained_url']).name
            if modern else True
        ),
        'effective_batch_is_32': (
            checkpoint_training.get('effective_batch_size')
            == args.expected_effective_batch
            if modern else True
        ),
        'epochs_is_45': (
            checkpoint_training.get('epochs') == args.expected_epochs
            if modern else True
        ),
        'seed_is_0': (
            checkpoint_training.get('seed') == args.expected_seed
            if modern else True
        ),
        'final_epoch_matches': (
            checkpoint.get('epoch') == args.expected_epochs - 1
            if modern else True
        ),
        'nominal_lr_matches': (
            checkpoint_training.get('nominal_lr') == 0.0005
            if modern else True
        ),
        'optimizer_lr_matches': (
            checkpoint_training.get('optimizer_lr') == 0.00003125
            if modern else True
        ),
        'effective_batch_factors_match': (
            checkpoint_training.get('micro_batch_size', 0)
            * checkpoint_training.get('accum_iter', 0)
            * checkpoint_training.get('world_size', 0)
            == args.expected_effective_batch
            if modern else True
        ),
        'optimizer_updates_match_dataset': (
            checkpoint_training.get('optimizer_updates_per_epoch')
            == checkpoint_training.get('train_dataset_size', 0)
            // args.expected_effective_batch
            if modern else True
        ),
        'consumed_samples_match_updates': (
            checkpoint_training.get('consumed_samples_per_epoch')
            == checkpoint_training.get('optimizer_updates_per_epoch', 0)
            * args.expected_effective_batch
            if modern else True
        ),
    }
    passed = (
        all(method_checks.values())
        and (not modern or all(modern_checks.values()))
    )
    report = {
        'schema_version': 1,
        'status': (
            'pass_with_legacy_small_metadata_exception'
            if passed and not modern else ('pass' if passed else 'fail')
        ),
        'passed': passed,
        'checkpoint': {
            'path': str(args.checkpoint.resolve()),
            'sha256': checkpoint_hash,
            'size_bytes': args.checkpoint.stat().st_size,
            'epoch': checkpoint.get('epoch'),
            'state_key_count': len(state),
            'finite': not nonfinite_state,
            'strict_training_model_load': True,
            'strict_cam_model_load': True,
        },
        'variant_resolution': resolution,
        'model_spec': model_spec_from_instance(cam_model),
        'parameters': {
            'total': sum(value.numel() for value in cam_model.parameters()),
            'trainable': sum(
                value.numel() for value in cam_model.parameters()
                if value.requires_grad
            ),
        },
        'official_pretrained': pretrained_report,
        'method_configuration': {
            'attention_normalization': attention,
            'bcss': bcss,
            'psl': psl,
            'cti_bgt': cti,
            'checks': method_checks,
        },
        'checkpoint_metadata_checks': modern_checks,
        'training_spec': checkpoint_training,
        'source_immutability': {
            'checkpoint_opened_read_only': True,
            'pretrained_opened_read_only': True,
        },
        'command': shlex.join([sys.executable] + sys.argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    return report


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
