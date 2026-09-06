#!/usr/bin/env python3
"""Validate and summarize one matched baseline/FinalLN experiment root."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import sys
from pathlib import Path


VARIANTS = (
    ('baseline', 'MCTformer+', False),
    ('final_ln', 'MCTformer+-FinalLN', True),
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
    parser.add_argument('--run-root', type=Path, required=True)
    return parser.parse_args()


def _read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


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


def execute(args):
    root = args.run_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    targets = (
        'classification_results.csv', 'cam_results.csv',
        'attention_results.csv', 'representation_results.csv',
        'comparison_summary.csv', 'FINAL_LN_EXPERIMENT_REPORT.md',
        'summary.json', 'EXPERIMENT_COMPLETE',
    )
    existing = [name for name in targets if (root / name).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite summary artifacts: {existing}')

    bundles = {}
    for key, label, expected_final_norm in VARIANTS:
        evaluation = root / 'evaluations' / key
        checkpoint = root / 'checkpoints' / key / 'mctformerplus_final.pth'
        required = (
            checkpoint,
            root / 'audit' / f'{key}.json',
            evaluation / 'classification/CLASSIFICATION_COMPLETE',
            evaluation / 'cam_train/CAM_COMPLETE',
            evaluation / 'cam_evaluation/THRESHOLD_EVALUATION_COMPLETE',
            evaluation / 'diagnostics/ATTENTION_REPRESENTATION_COMPLETE',
        )
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        payload = _read_json(root / 'audit' / f'{key}.json')
        if not payload.get('passed'):
            raise RuntimeError(f'{key} checkpoint audit failed')
        checkpoint_payload = _read_json(
            evaluation / 'diagnostics/summary.json'
        )
        if checkpoint_payload.get('final_norm') is not expected_final_norm:
            raise RuntimeError(f'{key} FinalLN diagnostic state mismatch')
        classification = _read_json(
            evaluation / 'classification/classification_metrics.json'
        )
        cam = _read_json(evaluation / 'cam_evaluation/metrics.json')
        attention = _read_csv(evaluation / 'diagnostics/attention_results.csv')
        representation = _read_csv(
            evaluation / 'diagnostics/representation_results.csv'
        )
        bundles[key] = {
            'label': label,
            'final_norm': expected_final_norm,
            'checkpoint': checkpoint,
            'checkpoint_sha256': _sha256(checkpoint),
            'classification': classification,
            'cam': cam,
            'attention': _index(attention, 'stage', 'metric'),
            'representation': _index(representation, 'metric'),
        }

    classification_rows = []
    cam_rows = []
    attention_rows = []
    representation_rows = []
    for key, label, final_norm in VARIANTS:
        bundle = bundles[key]
        classification = bundle['classification']
        class_map = classification['metrics_percent']['class_token'][
            'macro_class_ap'
        ]
        patch_map = classification['metrics_percent']['patch_gwrp'][
            'macro_class_ap'
        ]
        classification_rows.append({
            'model': label,
            'final_norm': str(final_norm).lower(),
            'class_token_map_percent': class_map,
            'patch_head_map_percent': patch_map,
        })

        fixed = bundle['cam']['selected_metrics']['fixed_0.45']
        oracle = bundle['cam']['selected_metrics']['oracle']
        cam_rows.append({
            'model': label,
            'final_norm': str(final_norm).lower(),
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
                source = bundle['attention'][(stage, metric)]
                attention_rows.append({
                    'model': label,
                    'final_norm': str(final_norm).lower(),
                    **source,
                })
        for metric in REPRESENTATION_METRICS:
            source = bundle['representation'][(metric,)]
            representation_rows.append({
                'model': label,
                'final_norm': str(final_norm).lower(),
                **source,
            })

    comparison_rows = []

    def add(domain, metric, baseline, final_ln, stage=''):
        comparison_rows.append({
            'domain': domain,
            'stage': stage,
            'metric': metric,
            'baseline': baseline,
            'final_ln': final_ln,
            'delta_final_ln_minus_baseline': final_ln - baseline,
        })

    add(
        'classification', 'class_token_map_percent',
        float(classification_rows[0]['class_token_map_percent']),
        float(classification_rows[1]['class_token_map_percent']),
    )
    add(
        'classification', 'patch_head_map_percent',
        float(classification_rows[0]['patch_head_map_percent']),
        float(classification_rows[1]['patch_head_map_percent']),
    )
    for metric in (
        'fixed_raw_cam_miou_percent', 'fixed_semantic_fg_precision_percent',
        'fixed_semantic_fg_recall_percent', 'best_raw_cam_miou_percent',
        'best_threshold',
    ):
        add('raw_cam', metric, float(cam_rows[0][metric]), float(cam_rows[1][metric]))
    attention_index = _index(attention_rows, 'model', 'stage', 'metric')
    for stage in ATTENTION_STAGES:
        for metric in ATTENTION_METRICS:
            baseline = float(attention_index[('MCTformer+', stage, metric)]['estimate'])
            final_ln = float(attention_index[
                ('MCTformer+-FinalLN', stage, metric)
            ]['estimate'])
            add('attention', metric, baseline, final_ln, stage)
    representation_index = _index(representation_rows, 'model', 'metric')
    for metric in REPRESENTATION_METRICS:
        baseline = float(representation_index[('MCTformer+', metric)]['estimate'])
        final_ln = float(representation_index[
            ('MCTformer+-FinalLN', metric)
        ]['estimate'])
        add('representation', metric, baseline, final_ln)

    _write_csv(root / 'classification_results.csv', classification_rows)
    _write_csv(root / 'cam_results.csv', cam_rows)
    _write_csv(root / 'attention_results.csv', attention_rows)
    _write_csv(root / 'representation_results.csv', representation_rows)
    _write_csv(root / 'comparison_summary.csv', comparison_rows)

    class_by_model = {row['model']: row for row in classification_rows}
    cam_by_model = {row['model']: row for row in cam_rows}
    attention_by_key = _index(attention_rows, 'model', 'stage', 'metric')
    representation_by_key = _index(representation_rows, 'model', 'metric')
    lines = [
        '# MCTformer+-FinalLN Matched Experiment', '',
        '## Code change', '',
        'The only model change is the optional existing final LayerNorm after block 12 and before the final class/patch split. `all_x_cls`, attention, the 3x3 patch head, GWRP, native last-three class-to-patch attention, all-layer patch propagation, and square-root refinement are unchanged.', '',
        '## Matched training', '',
        'Both models were trained independently from the same official DeiT-S initialization with seed 0, 45 epochs, effective batch 32, AdamW, the same cosine schedule, VOC lists, augmentation, input size 448, vanilla attention, BCSS E0, PSL baseline, and CTI-BGT disabled.', '',
        '## Classification', '',
        '| Model | Class-token mAP (%) | Patch-head mAP (%) |',
        '|---|---:|---:|',
    ]
    for _, label, _ in VARIANTS:
        row = class_by_model[label]
        lines.append(
            f"| {label} | {float(row['class_token_map_percent']):.4f} | "
            f"{float(row['patch_head_map_percent']):.4f} |"
        )
    lines.extend([
        '', '## Raw CAM on VOC train', '',
        '| Model | mIoU@0.45 (%) | Precision@0.45 (%) | Recall@0.45 (%) | Best mIoU (%) | Best threshold |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for _, label, _ in VARIANTS:
        row = cam_by_model[label]
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
    for _, label, _ in VARIANTS:
        for stage in ATTENTION_STAGES:
            value = lambda metric: float(attention_by_key[(label, stage, metric)]['estimate'])
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
    for _, label, _ in VARIANTS:
        value = lambda metric: float(representation_by_key[(label, metric)]['estimate'])
        lines.append(
            f'| {label} | {value("positive_class_token_pair_cosine"):.6f} | '
            f'{value("shared_presence_projection_auroc"):.6f} | '
            f'{value("fixed_all_ones_axis_energy"):.6f} | '
            f'{value("learned_shared_direction_all_ones_alignment"):.6f} |'
        )
    lines.extend([
        '', '## Completion', '',
        '- Both final checkpoints passed strict architecture, metadata, finite-value, pretrained-source, and training-contract audits.',
        '- All requested classification, CAM, attention, and representation stages completed.',
        '- Numerical deltas are recorded in `comparison_summary.csv`.',
        '',
    ])
    (root / 'FINAL_LN_EXPERIMENT_REPORT.md').write_text(
        '\n'.join(lines), encoding='utf-8'
    )

    summary = {
        'schema_version': 1,
        'status': 'complete',
        'experiment': 'MCTformer+-FinalLN',
        'source_checkpoints': {
            key: {
                'path': str(bundle['checkpoint']),
                'sha256': bundle['checkpoint_sha256'],
                'final_norm': bundle['final_norm'],
            }
            for key, bundle in bundles.items()
        },
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
