#!/usr/bin/env python3
"""Summarize baseline, joint FinalLN, and patch-only FinalLN results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path


MODELS = (
    ('MCTformer+', 'none'),
    ('MCTformer+-FinalLN', 'all_tokens'),
    ('MCTformer+-PatchFinalLN', 'patch_tokens'),
)
ATTENTION_STAGES = ('L10', 'L11', 'L12', 'native_last3')
ATTENTION_METRICS = (
    'c_pim', 'target_vs_bg_auroc', 'target_vs_other_fg_auroc',
    'positive_class_pair_top10_jaccard',
)
REPRESENTATION_METRICS = (
    'positive_class_token_pair_cosine',
    'shared_presence_projection_auroc',
    'fixed_all_ones_axis_energy',
    'learned_shared_direction_all_ones_alignment',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run-root', type=Path, required=True)
    parser.add_argument('--patch-run-root', type=Path, required=True)
    return parser.parse_args()


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def _read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    if path.exists():
        raise FileExistsError(path)
    if not rows:
        raise ValueError(f'cannot write empty table {path}')
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _index(rows, *keys):
    output = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in output:
            raise RuntimeError(f'duplicate table key {key}')
        output[key] = row
    return output


def _scope_fields(scope):
    return {
        'normalization_scope': scope,
        'final_norm': str(scope == 'all_tokens').lower(),
        'patch_final_norm': str(scope == 'patch_tokens').lower(),
    }


def _validate_source(source):
    for marker in ('PIPELINE_COMPLETE', 'EXPERIMENT_COMPLETE'):
        if not (source / marker).is_file():
            raise FileNotFoundError(source / marker)
    summary = _read_json(source / 'summary.json')
    if summary.get('status') != 'complete':
        raise RuntimeError('source matched FinalLN run is incomplete')
    expected = {
        'baseline': ('MCTformer+', False),
        'final_ln': ('MCTformer+-FinalLN', True),
    }
    for key, (_label, final_norm) in expected.items():
        audit = _read_json(source / 'audit' / f'{key}.json')
        if not audit.get('passed'):
            raise RuntimeError(f'source {key} audit failed')
        method = audit['method_configuration']
        if bool(method.get('final_norm')) is not final_norm:
            raise RuntimeError(f'source {key} FinalLN state mismatch')
        checkpoint_info = summary['source_checkpoints'][key]
        checkpoint = Path(checkpoint_info['path'])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if _sha256(checkpoint) != checkpoint_info['sha256']:
            raise RuntimeError(f'source {key} checkpoint hash mismatch')
    return summary


def _validate_patch(root, source):
    checkpoint = root / 'checkpoints/patch_final_ln/mctformerplus_final.pth'
    evaluation = root / 'evaluations/patch_final_ln'
    required = (
        checkpoint,
        root / 'audit/patch_final_ln.json',
        evaluation / 'classification/CLASSIFICATION_COMPLETE',
        evaluation / 'cam_train/CAM_COMPLETE',
        evaluation / 'cam_evaluation/THRESHOLD_EVALUATION_COMPLETE',
        evaluation / 'diagnostics/ATTENTION_REPRESENTATION_COMPLETE',
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = _read_json(root / 'audit/patch_final_ln.json')
    if not audit.get('passed'):
        raise RuntimeError('PatchFinalLN checkpoint audit failed')
    method = audit['method_configuration']
    if bool(method.get('final_norm')) or not bool(method.get('patch_final_norm')):
        raise RuntimeError('PatchFinalLN checkpoint scope mismatch')
    diagnostics = _read_json(evaluation / 'diagnostics/summary.json')
    if diagnostics.get('final_norm') or not diagnostics.get('patch_final_norm'):
        raise RuntimeError('PatchFinalLN diagnostic scope mismatch')

    source_audit = _read_json(source / 'audit/baseline.json')
    contract_keys = (
        'seed', 'epochs', 'micro_batch_size', 'accum_iter',
        'effective_batch_size', 'nominal_lr', 'optimizer_lr',
        'optimizer_updates_per_epoch', 'consumed_samples_per_epoch',
        'train_dataset_size', 'val_batch_size', 'world_size',
    )
    mismatches = {
        key: {
            'source_baseline': source_audit['training_spec'].get(key),
            'patch_final_ln': audit['training_spec'].get(key),
        }
        for key in contract_keys
        if source_audit['training_spec'].get(key) != audit['training_spec'].get(key)
    }
    if mismatches:
        raise RuntimeError(f'matched training contract mismatch: {mismatches}')
    if (
        source_audit['official_pretrained']['source_sha256']
        != audit['official_pretrained']['source_sha256']
    ):
        raise RuntimeError('pretrained initialization hash mismatch')
    if source_audit['model_spec'] != audit['model_spec']:
        raise RuntimeError('source and PatchFinalLN model specs differ')
    return checkpoint, evaluation, audit


def execute(args):
    source = args.source_run_root.resolve()
    root = args.patch_run_root.resolve()
    if source == root:
        raise ValueError('source and PatchFinalLN run roots must differ')
    if not source.is_dir() or not root.is_dir():
        raise FileNotFoundError(source if not source.is_dir() else root)
    targets = (
        'classification_results.csv', 'cam_results.csv',
        'attention_results.csv', 'representation_results.csv',
        'comparison_summary.csv', 'PATCH_FINAL_LN_EXPERIMENT_REPORT.md',
        'summary.json', 'EXPERIMENT_COMPLETE',
    )
    existing = [name for name in targets if (root / name).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite summary artifacts: {existing}')

    source_summary = _validate_source(source)
    patch_checkpoint, patch_evaluation, patch_audit = _validate_patch(root, source)

    source_class = _index(_read_csv(source / 'classification_results.csv'), 'model')
    source_cam = _index(_read_csv(source / 'cam_results.csv'), 'model')
    source_attention = _index(
        _read_csv(source / 'attention_results.csv'), 'model', 'stage', 'metric'
    )
    source_representation = _index(
        _read_csv(source / 'representation_results.csv'), 'model', 'metric'
    )

    patch_classification = _read_json(
        patch_evaluation / 'classification/classification_metrics.json'
    )
    patch_cam = _read_json(patch_evaluation / 'cam_evaluation/metrics.json')
    patch_attention = _index(
        _read_csv(patch_evaluation / 'diagnostics/attention_results.csv'),
        'stage', 'metric',
    )
    patch_representation = _index(
        _read_csv(patch_evaluation / 'diagnostics/representation_results.csv'),
        'metric',
    )

    classification_rows = []
    cam_rows = []
    attention_rows = []
    representation_rows = []
    for label, scope in MODELS[:2]:
        classification_rows.append({
            **_scope_fields(scope),
            'model': label,
            'class_token_map_percent': source_class[(label,)][
                'class_token_map_percent'
            ],
            'patch_head_map_percent': source_class[(label,)][
                'patch_head_map_percent'
            ],
        })
        cam_rows.append({
            **_scope_fields(scope), 'model': label,
            **{
                key: value for key, value in source_cam[(label,)].items()
                if key not in {'model', 'final_norm'}
            },
        })
        for stage in ATTENTION_STAGES:
            for metric in ATTENTION_METRICS:
                row = source_attention[(label, stage, metric)]
                attention_rows.append({
                    **_scope_fields(scope), 'model': label,
                    **{
                        key: value for key, value in row.items()
                        if key not in {'model', 'final_norm'}
                    },
                })
        for metric in REPRESENTATION_METRICS:
            row = source_representation[(label, metric)]
            representation_rows.append({
                **_scope_fields(scope), 'model': label,
                **{
                    key: value for key, value in row.items()
                    if key not in {'model', 'final_norm'}
                },
            })

    patch_label, patch_scope = MODELS[2]
    class_metrics = patch_classification['metrics_percent']
    classification_rows.append({
        **_scope_fields(patch_scope), 'model': patch_label,
        'class_token_map_percent': class_metrics['class_token']['macro_class_ap'],
        'patch_head_map_percent': class_metrics['patch_gwrp']['macro_class_ap'],
    })
    fixed = patch_cam['selected_metrics']['fixed_0.45']
    oracle = patch_cam['selected_metrics']['oracle']
    cam_rows.append({
        **_scope_fields(patch_scope), 'model': patch_label,
        'fixed_threshold': fixed['threshold'],
        'fixed_raw_cam_miou_percent': fixed['mean_iou_percent'],
        'fixed_semantic_fg_precision_percent': fixed[
            'semantic_foreground_precision_percent'
        ],
        'fixed_semantic_fg_recall_percent': fixed[
            'semantic_foreground_recall_percent'
        ],
        'best_threshold': oracle['threshold'],
        'best_raw_cam_miou_percent': oracle['mean_iou_percent'],
        'best_semantic_fg_precision_percent': oracle[
            'semantic_foreground_precision_percent'
        ],
        'best_semantic_fg_recall_percent': oracle[
            'semantic_foreground_recall_percent'
        ],
    })
    for stage in ATTENTION_STAGES:
        for metric in ATTENTION_METRICS:
            attention_rows.append({
                **_scope_fields(patch_scope), 'model': patch_label,
                **patch_attention[(stage, metric)],
            })
    for metric in REPRESENTATION_METRICS:
        representation_rows.append({
            **_scope_fields(patch_scope), 'model': patch_label,
            **patch_representation[(metric,)],
        })

    classification_index = _index(classification_rows, 'model')
    cam_index = _index(cam_rows, 'model')
    attention_index = _index(attention_rows, 'model', 'stage', 'metric')
    representation_index = _index(representation_rows, 'model', 'metric')
    comparison_rows = []

    def add(domain, metric, reference_label, reference, patch, stage=''):
        comparison_rows.append({
            'domain': domain, 'stage': stage, 'metric': metric,
            'reference_model': reference_label,
            'reference': reference, 'patch_final_ln': patch,
            'delta_patch_minus_reference': patch - reference,
        })

    for reference_label, _scope in MODELS[:2]:
        for metric in ('class_token_map_percent', 'patch_head_map_percent'):
            add(
                'classification', metric, reference_label,
                float(classification_index[(reference_label,)][metric]),
                float(classification_index[(patch_label,)][metric]),
            )
        for metric in (
            'fixed_raw_cam_miou_percent',
            'fixed_semantic_fg_precision_percent',
            'fixed_semantic_fg_recall_percent',
            'best_raw_cam_miou_percent', 'best_threshold',
        ):
            add(
                'raw_cam', metric, reference_label,
                float(cam_index[(reference_label,)][metric]),
                float(cam_index[(patch_label,)][metric]),
            )
        for stage in ATTENTION_STAGES:
            for metric in ATTENTION_METRICS:
                add(
                    'attention', metric, reference_label,
                    float(attention_index[(reference_label, stage, metric)][
                        'estimate'
                    ]),
                    float(attention_index[(patch_label, stage, metric)][
                        'estimate'
                    ]),
                    stage,
                )
        for metric in REPRESENTATION_METRICS:
            add(
                'representation', metric, reference_label,
                float(representation_index[(reference_label, metric)]['estimate']),
                float(representation_index[(patch_label, metric)]['estimate']),
            )

    _write_csv(root / 'classification_results.csv', classification_rows)
    _write_csv(root / 'cam_results.csv', cam_rows)
    _write_csv(root / 'attention_results.csv', attention_rows)
    _write_csv(root / 'representation_results.csv', representation_rows)
    _write_csv(root / 'comparison_summary.csv', comparison_rows)

    lines = [
        '# MCTformer+-PatchFinalLN Matched Experiment', '',
        '## Code and training contract', '',
        'PatchFinalLN keeps final class tokens and every per-block CCT token raw, while applying the existing final LayerNorm only to final patch tokens before the unchanged 3x3 patch head, GWRP, and native CAM path. Attention and CAM refinement are unchanged.', '',
        'PatchFinalLN was freshly trained from the same official DeiT-S initialization and seed-0 45-epoch configuration. The completed baseline and joint-FinalLN matched run is referenced read-only.', '',
        '## Classification', '',
        '| Model | Class-token mAP (%) | Patch-head mAP (%) |',
        '|---|---:|---:|',
    ]
    for label, _scope in MODELS:
        row = classification_index[(label,)]
        lines.append(
            f"| {label} | {float(row['class_token_map_percent']):.4f} | "
            f"{float(row['patch_head_map_percent']):.4f} |"
        )
    lines.extend([
        '', '## Raw CAM on VOC train', '',
        '| Model | mIoU@0.45 (%) | Precision@0.45 (%) | Recall@0.45 (%) | Best mIoU (%) | Best threshold |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for label, _scope in MODELS:
        row = cam_index[(label,)]
        lines.append(
            f"| {label} | {float(row['fixed_raw_cam_miou_percent']):.4f} | "
            f"{float(row['fixed_semantic_fg_precision_percent']):.4f} | "
            f"{float(row['fixed_semantic_fg_recall_percent']):.4f} | "
            f"{float(row['best_raw_cam_miou_percent']):.4f} | "
            f"{float(row['best_threshold']):.2f} |"
        )
    lines.extend([
        '', '## Class-to-patch attention on VOC val', '',
        '| Model | Stage | C-PiM | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Positive-pair top10 Jaccard |',
        '|---|---|---:|---:|---:|---:|',
    ])
    for label, _scope in MODELS:
        for stage in ATTENTION_STAGES:
            def value(metric):
                return float(attention_index[(label, stage, metric)]['estimate'])
            lines.append(
                f'| {label} | {stage} | {value("c_pim"):.6f} | '
                f'{value("target_vs_bg_auroc"):.6f} | '
                f'{value("target_vs_other_fg_auroc"):.6f} | '
                f'{value("positive_class_pair_top10_jaccard"):.6f} |'
            )
    lines.extend([
        '', '## Final class-token representation on VOC val', '',
        '| Model | Positive-pair cosine | Shared-presence AUROC | All-ones energy | Shared/all-ones alignment |',
        '|---|---:|---:|---:|---:|',
    ])
    for label, _scope in MODELS:
        def value(metric):
            return float(representation_index[(label, metric)]['estimate'])
        lines.append(
            f'| {label} | {value("positive_class_token_pair_cosine"):.6f} | '
            f'{value("shared_presence_projection_auroc"):.6f} | '
            f'{value("fixed_all_ones_axis_energy"):.6f} | '
            f'{value("learned_shared_direction_all_ones_alignment"):.6f} |'
        )
    lines.extend([
        '', '## Completion', '',
        '- Source baseline and joint-FinalLN results passed immutable hash and completion checks.',
        '- The PatchFinalLN checkpoint passed strict architecture, metadata, pretrained-source, finite-value, and matched-training audits.',
        '- Numerical deltas against both references are in `comparison_summary.csv`.',
        '',
    ])
    (root / 'PATCH_FINAL_LN_EXPERIMENT_REPORT.md').write_text(
        '\n'.join(lines), encoding='utf-8'
    )

    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'MCTformer+-PatchFinalLN',
        'source_run_root': str(source),
        'source_run_summary_sha256': _sha256(source / 'summary.json'),
        'source_checkpoints': source_summary['source_checkpoints'],
        'patch_checkpoint': {
            'path': str(patch_checkpoint),
            'sha256': _sha256(patch_checkpoint),
            'final_norm': False,
            'patch_final_norm': True,
        },
        'matched_training_spec': patch_audit['training_spec'],
        'classification_results': classification_rows,
        'cam_results': cam_rows,
        'attention_results': attention_rows,
        'representation_results': representation_rows,
        'comparison': comparison_rows,
        'command': shlex.join([sys.executable] + sys.argv),
    }
    (root / 'summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    (root / 'EXPERIMENT_COMPLETE').write_text('complete\n', encoding='utf-8')
    print(json.dumps(summary, sort_keys=True, allow_nan=False))
    return summary


def main():
    execute(parse_args())


if __name__ == '__main__':
    main()
