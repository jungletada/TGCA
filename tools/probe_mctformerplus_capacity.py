#!/usr/bin/env python3
"""Isolated CUDA micro-batch probes for matched effective batch size 32."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import (  # noqa: E402
    adapt_deit_checkpoint_for_mctformerplus,
    build_mctformerplus,
)


RESULT_PREFIX = 'CAPACITY_PROBE_RESULT='


def parse_candidates(value):
    values = tuple(int(item) for item in value.split(','))
    if not values or len(set(values)) != len(values) or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError('candidates must be unique positive integers')
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', default='base', choices=('tiny', 'small', 'base'))
    parser.add_argument('--official-pretrained', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--candidates', type=parse_candidates, default=(32, 16, 8, 4, 2))
    parser.add_argument('--effective-batch-size', type=int, default=32)
    parser.add_argument('--input-size', type=int, default=448)
    parser.add_argument('--maximum-reserved-fraction', type=float, default=0.90)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--worker-micro', type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def _classification_regularizer(class_embeddings, targets):
    class_embeddings = F.normalize(class_embeddings[-12:], dim=-1)
    scores = class_embeddings @ class_embeddings.permute(0, 1, 3, 2)
    ground_truth = torch.arange(
        targets.size(-1), dtype=torch.long, device=targets.device
    ).view(1, 1, -1).expand(
        class_embeddings.shape[0], class_embeddings.shape[1], -1
    )
    regularizer = torch.nn.CrossEntropyLoss(reduction='none')(
        scores.permute(1, 2, 3, 0), ground_truth.permute(1, 2, 0)
    )
    return torch.mean(
        torch.mean(
            torch.sum(regularizer * targets.unsqueeze(-1), dim=-2), dim=-1
        ) / (torch.sum(targets, dim=-1) + 1e-8)
    )


def worker(args):
    micro = args.worker_micro
    if micro < 1 or args.effective_batch_size % micro:
        raise ValueError('worker micro batch must divide effective batch size')
    accum = args.effective_batch_size // micro
    result = {
        'variant': args.variant,
        'micro_batch_size': micro,
        'accum_iter': accum,
        'effective_batch_size': args.effective_batch_size,
        'input_size': args.input_size,
        'status': 'running',
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable')
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = build_mctformerplus(
            args.variant,
            num_classes=20,
            input_size=args.input_size,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attention_normalization='vanilla',
            bcss_variant='e0',
            psl_variant='baseline',
            cti_bgt=False,
        )
        source = torch.load(args.official_pretrained, map_location='cpu')
        adapted, _ = adapt_deit_checkpoint_for_mctformerplus(source, model, 20)
        model.load_state_dict(adapted, strict=True)
        model.to(device).train()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=3.125e-5, weight_decay=0.05
        )
        scaler = torch.cuda.amp.GradScaler()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for step in range(accum):
            images = torch.randn(
                micro, 3, args.input_size, args.input_size, device=device
            )
            targets = torch.zeros(micro, 20, device=device)
            targets[:, 0] = 1
            targets[:, 14] = 1
            with torch.cuda.amp.autocast():
                outputs = model(images, active_labels=targets)
                class_loss = F.multilabel_soft_margin_loss(outputs[0], targets)
                patch_loss = F.multilabel_soft_margin_loss(outputs[2], targets)
                regularizer = _classification_regularizer(outputs[1], targets)
                loss = (class_loss + patch_loss + regularizer) / accum
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        properties = torch.cuda.get_device_properties(device)
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        result.update({
            'status': 'pass',
            'device': properties.name,
            'total_memory_bytes': properties.total_memory,
            'peak_allocated_bytes': peak_allocated,
            'peak_reserved_bytes': peak_reserved,
            'peak_reserved_fraction': peak_reserved / properties.total_memory,
            'elapsed_seconds': elapsed,
            'images_per_second': args.effective_batch_size / elapsed,
            'optimizer_step_completed': True,
            'finite_loss': bool(torch.isfinite(loss).item()),
        })
    except torch.cuda.OutOfMemoryError as error:
        result.update({'status': 'oom', 'error': str(error)})
    except Exception as error:  # surfaced to the parent as a hard failure
        result.update({
            'status': 'error',
            'error': f'{type(error).__name__}: {error}',
            'traceback': traceback.format_exc(),
        })
    print(RESULT_PREFIX + json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['status'] in {'pass', 'oom'} else 1


def parent(args):
    if args.output is None:
        raise ValueError('--output is required for the parent probe')
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    if not 0 < args.maximum_reserved_fraction < 1:
        raise ValueError('--maximum-reserved-fraction must be in (0,1)')
    if args.effective_batch_size < 1:
        raise ValueError('--effective-batch-size must be positive')
    attempts = []
    selected = None
    for micro in args.candidates:
        if args.effective_batch_size % micro:
            raise ValueError(
                f'candidate {micro} does not divide {args.effective_batch_size}'
            )
        command = [
            sys.executable, str(Path(__file__).resolve()),
            '--variant', args.variant,
            '--official-pretrained', str(args.official_pretrained),
            '--effective-batch-size', str(args.effective_batch_size),
            '--input-size', str(args.input_size),
            '--device', args.device,
            '--seed', str(args.seed),
            '--worker-micro', str(micro),
        ]
        completed = subprocess.run(
            command, cwd=REPO_ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        lines = completed.stdout.splitlines()
        payload_lines = [
            line[len(RESULT_PREFIX):] for line in lines
            if line.startswith(RESULT_PREFIX)
        ]
        if len(payload_lines) != 1:
            raise RuntimeError(
                f'Capacity worker did not return one result: {completed.stdout}'
            )
        result = json.loads(payload_lines[0])
        result['command'] = shlex.join(command)
        result['exit_code'] = completed.returncode
        result['worker_log'] = completed.stdout
        if completed.returncode != 0 or result['status'] == 'error':
            raise RuntimeError(f'Capacity worker failed: {result}')
        attempts.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if (
            result['status'] == 'pass'
            and result['peak_reserved_fraction']
            <= args.maximum_reserved_fraction
        ):
            selected = {
                'micro_batch_size': result['micro_batch_size'],
                'accum_iter': result['accum_iter'],
                'effective_batch_size': result['effective_batch_size'],
                'peak_reserved_fraction': result['peak_reserved_fraction'],
            }
            break
    if selected is None:
        raise RuntimeError('No capacity candidate passed with required headroom')
    report = {
        'schema_version': 1,
        'status': 'pass',
        'variant': args.variant,
        'selection_policy': (
            'largest candidate completing forward/backward/AdamW step with '
            f'peak reserved fraction <= {args.maximum_reserved_fraction}'
        ),
        'selected': selected,
        'attempts': attempts,
        'source_pretrained': str(args.official_pretrained.resolve()),
        'command': shlex.join([sys.executable] + sys.argv),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(report, sort_keys=True))
    return report


def main():
    args = parse_args()
    if args.worker_micro:
        raise SystemExit(worker(args))
    parent(args)


if __name__ == '__main__':
    main()
