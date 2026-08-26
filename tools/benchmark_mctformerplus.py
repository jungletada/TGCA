#!/usr/bin/env python3
"""Benchmark MCTformer+ inference for one attention-normalization mode."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.mctformer_plus import MCTformerPlusCam
from models.tgca import SUPPORTED_MODES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def percentile(values, quantile):
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if args.input_size <= 0 or args.batch_size <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError("Invalid benchmark dimensions or iteration counts")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("attention_normalization", {})
    if config and config.get("mode") != args.mode:
        raise ValueError(
            f"Checkpoint mode {config.get('mode')!r} does not match {args.mode!r}"
        )
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = MCTformerPlusCam(
        num_classes=20,
        input_size=448,
        attention_normalization=args.mode,
        attention_gamma=1.0,
    )
    incompatibility = model.load_state_dict(state_dict, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(str(incompatibility))
    model.to(device).eval()
    inputs = torch.randn(
        args.batch_size, 3, args.input_size, args.input_size, device=device
    )
    torch.backends.cudnn.benchmark = True

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = model(inputs)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        baseline_allocated = torch.cuda.memory_allocated(device)
        durations_ms = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            output = model(inputs)
            torch.cuda.synchronize(device)
            durations_ms.append(1000.0 * (time.perf_counter() - start))
    if not torch.isfinite(output).all():
        raise RuntimeError("Non-finite benchmark output")

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    relation_bias_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("relation_bias")
    )
    metrics = {
        "host": "MCTformer+",
        "normalization": args.mode,
        "checkpoint": str(args.checkpoint.resolve()),
        "device": torch.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "batch_size": args.batch_size,
        "input_size": args.input_size,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "total_parameters": total_parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "relation_bias_parameters": relation_bias_parameters,
        "latency_ms_mean": statistics.mean(durations_ms),
        "latency_ms_median": statistics.median(durations_ms),
        "latency_ms_p95": percentile(durations_ms, 0.95),
        "throughput_images_per_second": (
            1000.0 * args.batch_size / statistics.mean(durations_ms)
        ),
        "baseline_allocated_memory_mb": baseline_allocated / (1024 ** 2),
        "peak_allocated_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
        "incremental_peak_allocated_memory_mb": (
            torch.cuda.max_memory_allocated(device) - baseline_allocated
        ) / (1024 ** 2),
        "output_shape": list(output.shape),
        "latency_samples_ms": durations_ms,
        "notes": "Current commit implementation; MACs/FLOPs not measured by this tool.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
