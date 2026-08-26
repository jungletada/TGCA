#!/usr/bin/env python3
"""Run the deterministic synthetic token-count and replication experiment."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.tgca import token_group_normalize


CLASS_COUNTS = (1, 20, 80)
PATCH_COUNTS = (49, 196, 400, 784, 1024)
HEAD_COUNTS = (1, 6)
REGIMES = ("equal", "iid_normal", "class_favored", "patch_favored")
REPLICATION_FACTORS = (1, 2, 4, 8)
MODES = ("vanilla", "tgca")


def build_logits(regime, heads, queries, class_count, patch_count, generator):
    shape = (1, heads, queries, class_count + patch_count)
    if regime == "equal":
        return torch.zeros(shape)
    logits = torch.randn(shape, generator=generator)
    if regime == "class_favored":
        logits[..., :class_count] += 1.0
    elif regime == "patch_favored":
        logits[..., class_count:] += 1.0
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    generator = torch.Generator().manual_seed(args.seed)
    fieldnames = [
        "seed", "mode", "regime", "class_count", "patch_count", "heads",
        "replication_factor", "class_group_mass", "patch_group_mass",
        "row_sum_max_error", "attention_output_max_error",
    ]
    rows = []
    for class_count in CLASS_COUNTS:
        for patch_count in PATCH_COUNTS:
            groups = torch.cat(
                (torch.zeros(class_count, dtype=torch.long), torch.ones(patch_count, dtype=torch.long))
            )
            for heads in HEAD_COUNTS:
                for regime in REGIMES:
                    logits = build_logits(
                        regime, heads, 4, class_count, patch_count, generator
                    )
                    values = torch.randn(
                        1, heads, class_count + patch_count, 8, generator=generator
                    )
                    for mode in MODES:
                        reference_attention = token_group_normalize(logits, groups, mode=mode)
                        reference_output = reference_attention @ values
                        for replication_factor in REPLICATION_FACTORS:
                            replicated_logits = torch.cat(
                                (
                                    logits[..., :class_count],
                                    logits[..., class_count:].repeat_interleave(replication_factor, dim=-1),
                                ),
                                dim=-1,
                            )
                            replicated_values = torch.cat(
                                (
                                    values[..., :class_count, :],
                                    values[..., class_count:, :].repeat_interleave(
                                        replication_factor, dim=-2
                                    ),
                                ),
                                dim=-2,
                            )
                            replicated_groups = torch.cat(
                                (
                                    groups[:class_count],
                                    groups[class_count:].repeat_interleave(replication_factor),
                                )
                            )
                            attention = token_group_normalize(
                                replicated_logits, replicated_groups, mode=mode
                            )
                            output = attention @ replicated_values
                            class_mass = attention[..., replicated_groups == 0].sum(-1)
                            patch_mass = attention[..., replicated_groups == 1].sum(-1)
                            rows.append(
                                {
                                    "seed": args.seed,
                                    "mode": mode,
                                    "regime": regime,
                                    "class_count": class_count,
                                    "patch_count": patch_count,
                                    "heads": heads,
                                    "replication_factor": replication_factor,
                                    "class_group_mass": float(class_mass.mean()),
                                    "patch_group_mass": float(patch_mass.mean()),
                                    "row_sum_max_error": float(
                                        (attention.sum(-1) - 1.0).abs().max()
                                    ),
                                    "attention_output_max_error": float(
                                        (output - reference_output).abs().max()
                                    ),
                                }
                            )

    csv_path = args.output_dir / "synthetic_replication.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tgca_rows = [row for row in rows if row["mode"] == "tgca"]
    vanilla_replicated = [
        row
        for row in rows
        if row["mode"] == "vanilla" and row["replication_factor"] > 1
    ]
    summary = {
        "seed": args.seed,
        "rows": len(rows),
        "tgca_max_row_sum_error": max(row["row_sum_max_error"] for row in tgca_rows),
        "tgca_max_replication_output_error": max(
            row["attention_output_max_error"] for row in tgca_rows
        ),
        "vanilla_max_replication_output_error": max(
            row["attention_output_max_error"] for row in vanilla_replicated
        ),
        "grid": {
            "class_counts": CLASS_COUNTS,
            "patch_counts": PATCH_COUNTS,
            "head_counts": HEAD_COUNTS,
            "regimes": REGIMES,
            "replication_factors": REPLICATION_FACTORS,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    figure_rows = [
        row for row in rows
        if row["regime"] == "equal"
        and row["class_count"] == 20
        and row["heads"] == 6
        and row["replication_factor"] == 1
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for mode in MODES:
        selected = sorted(
            (row for row in figure_rows if row["mode"] == mode),
            key=lambda row: row["patch_count"],
        )
        ax.plot(
            [row["patch_count"] for row in selected],
            [row["class_group_mass"] for row in selected],
            marker="o",
            label=mode,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Patch-token count")
    ax.set_ylabel("Class-group attention mass")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "group_mass_vs_patch_count.pdf")
    fig.savefig(args.output_dir / "group_mass_vs_patch_count.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
