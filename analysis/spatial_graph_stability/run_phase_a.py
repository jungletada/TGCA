#!/usr/bin/env python
"""Run frozen Phase A basis-dependence diagnostics on LaST and PatchFinalLN."""

from __future__ import annotations

import argparse
import gc
import math
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .basis import (
    cosine_affinity,
    generate_basis_transforms,
    last_channel_selector,
    transform_conv_input_basis,
    transform_linear_weight,
    transform_metadata,
)
from .basis_metrics import (
    channel_selection_agreement,
    compare_vote_maps,
    logit_comparison,
    semantic_vote_composition,
    summarize,
    vote_counts,
)
from .frozen import (
    VOCNaturalImageDataset,
    build_mct_dataset,
    load_official_last,
    load_patch_final_mctformer,
    official_last_tokens,
)
from .graph import PATCH_LABEL_MIXED, PATCH_LABEL_VOID, semantic_patch_labels
from .provenance import (
    EXPECTED_VOC_IMAGES,
    RunLog,
    command_line,
    create_output,
    csv_dump,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    text_dump,
    timestamp,
    write_environment_manifests,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--official-repo", type=Path, default=Path("hosts/LAST-ViT"))
    parser.add_argument(
        "--official-checkpoint",
        type=Path,
        default=Path("checkpoints/last_vit/ViT_190k.pth"),
    )
    parser.add_argument("--mct-checkpoint", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument(
        "--list-path",
        type=Path,
        default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--official-batch-size", type=int, default=32)
    parser.add_argument("--mct-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=20260901)
    parser.add_argument("--haar-count", type=int, default=10)
    parser.add_argument("--test-log", type=Path)
    return parser.parse_args()


def _resolve(repo_root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else repo_root / value).resolve()


def _summary_rows(
    transform_name: str,
    kind: str,
    metrics: dict[str, np.ndarray],
    *,
    section: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, values in metrics.items():
        row: dict[str, object] = {
            "section": section,
            "transform": transform_name,
            "kind": kind,
            "metric": metric,
        }
        row.update(summarize(values))
        rows.append(row)
    return rows


def _save_basis_outputs(
    raw_dir: Path,
    prefix: str,
    image_ids: list[str],
    transforms,
    indices: np.ndarray,
    votes: np.ndarray,
    pooled: np.ndarray,
    logits: np.ndarray | None,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for transform_index, transform in enumerate(transforms):
        path = raw_dir / f"{prefix}_{transform.name}.npz"
        payload = {
            "image_ids": np.asarray(image_ids),
            "selected_indices": indices[transform_index],
            "patch_vote_counts": votes[transform_index],
            "pooled_synthetic_token": pooled[transform_index],
        }
        if logits is not None:
            payload["last_logits"] = logits[transform_index]
        np.savez(path, **payload)
        records.append(
            {
                "transform": transform.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "arrays": {key: list(value.shape) for key, value in payload.items()},
            }
        )
    return records


def _standard_control_error(
    query: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    rotation: torch.Tensor,
) -> float:
    # Float64 isolates the mathematical reparameterization check from
    # float32 accumulation order in the selector sensitivity measurement.
    query = query.double()
    weight = weight.double()
    bias = bias.double()
    rotation = rotation.double()
    original = F.linear(query, weight, bias)
    transformed = F.linear(
        query @ rotation,
        transform_linear_weight(weight, rotation),
        bias,
    )
    return float((original - transformed).abs().max().item())


def _run_official(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    raw_dir: Path,
    plots_dir: Path,
    device: torch.device,
    log: RunLog,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    dataset = VOCNaturalImageDataset(args.voc_root, args.list_path, limit=args.limit)
    if not args.limit and len(dataset) != EXPECTED_VOC_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_VOC_IMAGES} VOC val images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.official_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    transforms = generate_basis_transforms(
        768, base_seed=args.base_seed, haar_count=args.haar_count
    )
    rotations = [
        torch.from_numpy(transform.matrix).to(device=device, dtype=torch.float32)
        for transform in transforms
    ]
    model, model_metadata = load_official_last(args.official_checkpoint, args.official_repo)
    model.to(device).eval()
    classifier = model.heads.head
    transformed_weights = [
        transform_linear_weight(classifier.weight, rotation)
        for rotation in rotations
    ]
    images = len(dataset)
    count = len(transforms)
    indices = np.empty((count, images, 1, 768), dtype=np.uint16)
    votes = np.empty((count, images, 196), dtype=np.uint16)
    pooled = np.empty((count, images, 768), dtype=np.float32)
    logits = np.empty((count, images, 1000), dtype=np.float32)
    invariance = {transform.name: 0.0 for transform in transforms}
    stability_nonfinite = {transform.name: 0 for transform in transforms}
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            batch_images = batch["image"].to(device, non_blocking=True)
            query, patch_tokens, _ = official_last_tokens(model, batch_images)
            batch_size = len(batch_images)
            for transform_index, (transform, rotation, transformed_weight) in enumerate(
                zip(transforms, rotations, transformed_weights)
            ):
                transformed_query = query @ rotation
                transformed_patches = patch_tokens @ rotation
                selected, selected_indices, stability, _ = last_channel_selector(
                    transformed_patches, topk=1, eps=None
                )
                transformed_logits = F.linear(
                    selected, transformed_weight, classifier.bias
                )
                stability_nonfinite[transform.name] += int(
                    (~torch.isfinite(stability)).sum().item()
                )
                # The official selector deliberately has no denominator
                # clamp. Its ranking is still defined when an isolated score
                # is +/-inf, provided the selected original token and logits
                # remain finite. Do not silently substitute a new selector.
                if not torch.isfinite(selected).all() or not torch.isfinite(transformed_logits).all():
                    raise RuntimeError(
                        f"official selector output produced non-finite values for {transform.name}"
                    )
                selected_votes = vote_counts(selected_indices, 196)
                sl = slice(offset, offset + batch_size)
                indices[transform_index, sl] = selected_indices.cpu().numpy().astype(np.uint16)
                votes[transform_index, sl] = selected_votes.cpu().numpy().astype(np.uint16)
                pooled[transform_index, sl] = selected.cpu().numpy().astype(np.float32)
                logits[transform_index, sl] = transformed_logits.cpu().numpy().astype(np.float32)
                invariance[transform.name] = max(
                    invariance[transform.name],
                    _standard_control_error(
                        query, classifier.weight, classifier.bias, rotation
                    ),
                )
            offset += batch_size
            if batch_number == 1 or batch_number % 10 == 0 or offset == images:
                log(f"A1 official LaST processed {offset}/{images} images")
    if offset != images:
        raise RuntimeError(f"A1 processed {offset} images, expected {images}")
    duration = time.perf_counter() - started
    raw_records = _save_basis_outputs(
        raw_dir, "official_last", dataset.image_ids, transforms, indices, votes, pooled, logits
    )

    vote_rows: list[dict[str, object]] = []
    logit_rows: list[dict[str, object]] = []
    identity_votes = votes[0]
    identity_logits = logits[0]
    for transform_index, transform in enumerate(transforms):
        comparison = compare_vote_maps(identity_votes, votes[transform_index])
        vote_rows.extend(
            _summary_rows(
                transform.name,
                transform.kind,
                comparison,
                section="official_last_vote_comparison",
            )
        )
        if transform.channel_correspondence is not None:
            agreement = channel_selection_agreement(
                indices[0], indices[transform_index], transform.channel_correspondence
            )
            vote_rows.extend(
                _summary_rows(
                    transform.name,
                    transform.kind,
                    {"per_channel_selected_index_agreement": agreement},
                    section="official_last_channel_correspondence",
                )
            )
        logit_rows.extend(
            _summary_rows(
                transform.name,
                transform.kind,
                logit_comparison(identity_logits, logits[transform_index]),
                section="official_last_logit_sensitivity",
            )
        )

    first_haar = next(index for index, item in enumerate(transforms) if item.kind == "haar")
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    axes[0].imshow(votes[0, 0].reshape(14, 14), cmap="magma")
    axes[0].set_title("Identity votes")
    axes[1].imshow(votes[first_haar, 0].reshape(14, 14), cmap="magma")
    axes[1].set_title(transforms[first_haar].name)
    axes[2].imshow(
        votes[first_haar, 0].reshape(14, 14).astype(float)
        - votes[0, 0].reshape(14, 14).astype(float),
        cmap="coolwarm",
    )
    axes[2].set_title("rotation − identity")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"Official LaST vote map: {dataset.image_ids[0]}")
    figure.tight_layout()
    figure.savefig(plots_dir / "official_identity_vs_rotation_vote_maps.png", dpi=180)
    plt.close(figure)

    metadata = {
        "model": model_metadata,
        "dataset": {
            "name": "PASCAL VOC 2012 val natural-image fallback",
            "image_count": images,
            "image_ids": dataset.image_ids,
            "imagenet_available": False,
            "classification_accuracy_reported": False,
            "transform": "Resize(256), CenterCrop(224), ToTensor, ImageNet normalize",
        },
        "selector": {
            "source": "official cls_pretrain/conf.py dense_vit.forward",
            "fft_dimension": "embedding",
            "sigma": math.sqrt(768),
            "topk": 1,
            "topk_dimension": "patch",
            "denominator_clamp": None,
        },
        "duration_seconds": duration,
        "raw_outputs": raw_records,
        "unclamped_stability_nonfinite_counts": stability_nonfinite,
    }
    del model, rotations, transformed_weights
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, vote_rows, logit_rows, invariance


def _conv_equivalence_error(
    patch_tokens: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    rotation: torch.Tensor,
) -> float:
    batch, patches, width = patch_tokens.shape
    side = int(math.isqrt(patches))
    if side * side != patches:
        raise ValueError("MCT patch-token grid is not square")
    values = patch_tokens.double()
    rotation = rotation.double()
    weight = weight.double()
    bias = bias.double()
    grid = values.reshape(batch, side, side, width).permute(0, 3, 1, 2)
    original = F.conv2d(grid, weight, bias, padding=1)
    transformed_values = values @ rotation
    transformed_grid = transformed_values.reshape(
        batch, side, side, width
    ).permute(0, 3, 1, 2)
    transformed_weight = transform_conv_input_basis(weight, rotation)
    transformed = F.conv2d(transformed_grid, transformed_weight, bias, padding=1)
    return float((original - transformed).abs().max().item())


def _run_mct(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    raw_dir: Path,
    plots_dir: Path,
    device: torch.device,
    log: RunLog,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, float], dict[str, float]]:
    dataset = build_mct_dataset(args.voc_root, args.list_path, limit=args.limit)
    if not args.limit and len(dataset) != EXPECTED_VOC_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_VOC_IMAGES} VOC val images, got {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.mct_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    transforms = generate_basis_transforms(
        384, base_seed=args.base_seed, haar_count=args.haar_count
    )
    rotations = [
        torch.from_numpy(transform.matrix).to(device=device, dtype=torch.float32)
        for transform in transforms
    ]
    model, model_metadata = load_patch_final_mctformer(args.mct_checkpoint)
    model.to(device).eval()
    images = len(dataset)
    count = len(transforms)
    indices = np.empty((count, images, 1, 384), dtype=np.uint16)
    votes = np.empty((count, images, 784), dtype=np.uint16)
    pooled = np.empty((count, images, 384), dtype=np.float32)
    semantic_labels = np.empty((images, 784), dtype=np.int8)
    conv_invariance = {transform.name: 0.0 for transform in transforms}
    affinity_invariance = {transform.name: 0.0 for transform in transforms}
    stability_nonfinite = {
        transform.name: {"total": 0, "nan": 0, "positive_infinity": 0, "negative_infinity": 0}
        for transform in transforms
    }
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            batch_images = batch["image"].to(device, non_blocking=True)
            _, patch_tokens, attentions, all_class, auxiliary = model.forward_features(
                batch_images, return_aux=True
            )
            if (
                patch_tokens.shape[1:] != (784, 384)
                or auxiliary.get("patch_final_norm") is not True
                or auxiliary.get("final_norm") is not False
                or len(attentions) != 12
                or len(all_class) != 12
            ):
                raise RuntimeError("PatchFinalLN runtime contract was not reproduced")
            batch_size = len(batch_images)
            if offset == 0:
                # Mandatory equivalence controls are evaluated in float64 on a
                # fixed first image to avoid conflating fp32 reduction order.
                base_affinity = cosine_affinity(patch_tokens[:1].double())
                for transform, rotation in zip(transforms, rotations):
                    conv_invariance[transform.name] = _conv_equivalence_error(
                        patch_tokens[:1], model.head.weight, model.head.bias, rotation
                    )
                    transformed_affinity = cosine_affinity(
                        patch_tokens[:1].double() @ rotation.double()
                    )
                    affinity_invariance[transform.name] = float(
                        (base_affinity - transformed_affinity).abs().max().item()
                    )
            for transform_index, rotation in enumerate(rotations):
                transformed_patches = patch_tokens @ rotation
                selected, selected_indices, stability, _ = last_channel_selector(
                    transformed_patches, topk=1, eps=None
                )
                name = transforms[transform_index].name
                # This remains the exact official, unclamped selector. A
                # zero low-pass residual can yield an infinite ranking score;
                # topk/gather still select a finite original patch token. The
                # count is an output diagnostic, not a reason to alter LaST.
                stability_nonfinite[name]["total"] += int((~torch.isfinite(stability)).sum().item())
                stability_nonfinite[name]["nan"] += int(torch.isnan(stability).sum().item())
                stability_nonfinite[name]["positive_infinity"] += int(torch.isposinf(stability).sum().item())
                stability_nonfinite[name]["negative_infinity"] += int(torch.isneginf(stability).sum().item())
                if not torch.isfinite(selected).all():
                    raise RuntimeError(f"MCT selected token is non-finite for {name}")
                selected_votes = vote_counts(selected_indices, 784)
                sl = slice(offset, offset + batch_size)
                indices[transform_index, sl] = selected_indices.cpu().numpy().astype(np.uint16)
                votes[transform_index, sl] = selected_votes.cpu().numpy().astype(np.uint16)
                pooled[transform_index, sl] = selected.cpu().numpy().astype(np.float32)
            masks = batch["mask"].numpy()
            for local_index in range(batch_size):
                semantic_labels[offset + local_index] = semantic_patch_labels(
                    masks[local_index], patch_size=16
                ).reshape(-1)
            offset += batch_size
            if batch_number == 1 or batch_number % 20 == 0 or offset == images:
                log(f"A2 PatchFinalLN processed {offset}/{images} images")
    if offset != images:
        raise RuntimeError(f"A2 processed {offset} images, expected {images}")
    duration = time.perf_counter() - started
    raw_records = _save_basis_outputs(
        raw_dir, "mct_patchfinal", dataset.image_ids, transforms, indices, votes, pooled, None
    )
    labels_path = raw_dir / "mct_patch_semantic_labels.npz"
    np.savez(
        labels_path,
        image_ids=np.asarray(dataset.image_ids),
        semantic_labels=semantic_labels,
        mixed_code=np.asarray(PATCH_LABEL_MIXED),
        void_code=np.asarray(PATCH_LABEL_VOID),
    )

    rows: list[dict[str, object]] = []
    for transform_index, transform in enumerate(transforms):
        comparison = compare_vote_maps(votes[0], votes[transform_index])
        rows.extend(
            _summary_rows(
                transform.name,
                transform.kind,
                comparison,
                section="mct_vote_comparison",
            )
        )
        if transform.channel_correspondence is not None:
            agreement = channel_selection_agreement(
                indices[0], indices[transform_index], transform.channel_correspondence
            )
            rows.extend(
                _summary_rows(
                    transform.name,
                    transform.kind,
                    {"per_channel_selected_index_agreement": agreement},
                    section="mct_channel_correspondence",
                )
            )
        composition: dict[str, list[float]] = {}
        for image_index in range(images):
            result = semantic_vote_composition(
                votes[transform_index, image_index], semantic_labels[image_index]
            )
            for metric, value in result.items():
                composition.setdefault(metric, []).append(value)
        rows.extend(
            _summary_rows(
                transform.name,
                transform.kind,
                {key: np.asarray(value) for key, value in composition.items()},
                section="mct_union_gt_vote_composition",
            )
        )

    first_haar = next(index for index, item in enumerate(transforms) if item.kind == "haar")
    figure, axes = plt.subplots(1, 4, figsize=(12.5, 3.2))
    axes[0].imshow(votes[0, 0].reshape(28, 28), cmap="magma")
    axes[0].set_title("Identity votes")
    axes[1].imshow(votes[first_haar, 0].reshape(28, 28), cmap="magma")
    axes[1].set_title(transforms[first_haar].name)
    axes[2].imshow(
        votes[first_haar, 0].reshape(28, 28).astype(float)
        - votes[0, 0].reshape(28, 28).astype(float),
        cmap="coolwarm",
    )
    axes[2].set_title("rotation − identity")
    shown_labels = semantic_labels[0].reshape(28, 28).copy()
    shown_labels[shown_labels < 0] = 21
    axes[3].imshow(shown_labels, cmap="tab20b", vmin=0, vmax=21)
    axes[3].set_title("Patch-level GT")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(f"PatchFinalLN LaST-selector vote map: {dataset.image_ids[0]}")
    figure.tight_layout()
    figure.savefig(plots_dir / "mct_identity_vs_rotation_vote_maps.png", dpi=180)
    plt.close(figure)

    metadata = {
        "model": model_metadata,
        "dataset": {
            "name": "PASCAL VOC 2012 val",
            "image_count": images,
            "input_size": 448,
            "patch_grid": [28, 28],
            "gt_geometry": {
                "valid_pixels_minimum_fraction": 0.5,
                "semantic_majority_fraction": 0.5,
                "mixed_and_void_excluded_from_valid_composition": True,
            },
        },
        "selector": {
            "source": "official LaST channel selector applied without training",
            "fft_dimension": "embedding",
            "sigma": math.sqrt(384),
            "topk": 1,
            "topk_dimension": "patch",
            "denominator_clamp": None,
        },
        "duration_seconds": duration,
        "raw_outputs": raw_records,
        "unclamped_stability_nonfinite_counts": stability_nonfinite,
        "semantic_labels": {
            "path": str(labels_path),
            "sha256": sha256_file(labels_path),
        },
    }
    del model, rotations
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, rows, conv_invariance, affinity_invariance


def _metric_lookup(rows: list[dict[str, object]], transform: str, metric: str) -> float | None:
    matches = [
        row for row in rows
        if row["transform"] == transform and row["metric"] == metric
    ]
    if not matches:
        return None
    value = matches[0].get("mean")
    return float(value) if value is not None else None


def _write_reports(
    output_dir: Path,
    official_metadata: dict[str, object],
    official_vote_rows: list[dict[str, object]],
    logit_rows: list[dict[str, object]],
    mct_metadata: dict[str, object],
    mct_rows: list[dict[str, object]],
    checks: dict[str, object],
) -> None:
    haar_names = [f"haar_{index:02d}" for index in range(10)]
    official_table = []
    for name in ("permutation", "signed_permutation", *haar_names):
        official_table.append(
            (
                name,
                _metric_lookup(official_vote_rows, name, "vote_spearman"),
                _metric_lookup(official_vote_rows, name, "top10_jaccard"),
                _metric_lookup(logit_rows, name, "logit_cosine"),
                _metric_lookup(logit_rows, name, "top1_agreement"),
            )
        )
    lines = [
        "# Official LaST Basis-Dependence Report",
        "",
        "## Frozen inputs and scope",
        "",
        f"- Official repository commit: `{official_metadata['model']['repository_commit']}`.",
        f"- Official checkpoint SHA256: `{official_metadata['model']['checkpoint_sha256']}`.",
        f"- Natural-image fallback: full VOC val (`{official_metadata['dataset']['image_count']}` images) with the official 224 validation transform.",
        "- ImageNet val was unavailable; no ImageNet classification accuracy is reported.",
        "- Selector is the official unclamped FFT/Gaussian/Top-1 implementation; no training occurred.",
        "",
        "## Equivalent-basis results",
        "",
        "| Basis | Vote Spearman | Top-10% vote Jaccard | LaST-logit cosine | Top-1 agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, spearman, jaccard, cosine, top1 in official_table:
        lines.append(
            f"| {name} | {spearman:.6f} | {jaccard:.6f} | {cosine:.6f} | {top1:.6f} |"
        )
    standard_max = max(checks["official_standard_classifier_max_abs_logit_error"].values())
    lines.extend(
        [
            "",
            "## Equivalence control and interpretation",
            "",
            f"The ordinary final CLS classifier remained equivalent with maximum float64 absolute logit error `{standard_max:.3e}` (required `<1e-5`). The table therefore measures selector/output sensitivity under representations of the same ordinary classifier function.",
            "",
            "These are representation/selector diagnostics only. They do not establish accuracy changes, semantic leakage, or a causal mechanism.",
            "",
        ]
    )
    text_dump(output_dir / "OFFICIAL_LAST_BASIS_REPORT.md", "\n".join(lines))

    mct_table = []
    for name in ("permutation", "signed_permutation", *haar_names):
        mct_table.append(
            (
                name,
                _metric_lookup(mct_rows, name, "vote_spearman"),
                _metric_lookup(mct_rows, name, "top10_jaccard"),
                _metric_lookup(mct_rows, name, "foreground_vote_fraction_valid"),
                _metric_lookup(mct_rows, name, "foreground_vote_enrichment"),
            )
        )
    lines = [
        "# PatchFinalLN MCTformer+ Basis-Control Report",
        "",
        "## Frozen inputs and scope",
        "",
        f"- PatchFinalLN checkpoint: `{mct_metadata['model']['checkpoint']}`.",
        f"- Checkpoint SHA256: `{mct_metadata['model']['checkpoint_sha256']}`.",
        f"- Full VOC val: `{mct_metadata['dataset']['image_count']}` images, deterministic 448 transform, 28×28 patches.",
        "- The semantic 3×3 map is held invariant by transforming every input-channel kernel slice; GT is used only after selection for union-FG/BG composition.",
        "",
        "## Selector results",
        "",
        "| Basis | Vote Spearman | Top-10% Jaccard | FG vote mass (valid) | FG vote enrichment |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, spearman, jaccard, fg_mass, enrichment in mct_table:
        lines.append(
            f"| {name} | {spearman:.6f} | {jaccard:.6f} | {fg_mass:.6f} | {enrichment:.6f} |"
        )
    conv_max = max(checks["mct_conv_max_abs_cam_logit_error"].values())
    affinity_max = max(checks["mct_cosine_affinity_max_abs_error"].values())
    lines.extend(
        [
            "",
            "## Invariance controls and interpretation",
            "",
            f"The reparameterized 3×3 classifier reproduced its semantic map with maximum float64 absolute error `{conv_max:.3e}`; pairwise cosine affinity remained invariant with maximum error `{affinity_max:.3e}` (both required `<1e-5`).",
            "",
            "Any vote-map differences therefore belong to the channel-index Fourier selector, while the spatial semantic map and cosine graph remain equivalent. This is a frozen representation-level diagnosis, not a trained-method or localization claim.",
            "",
            f"The official unclamped selector produced {sum(value['total'] for value in mct_metadata['unclamped_stability_nonfinite_counts'].values())} non-finite ranking entries on this input set. Selected original patch tokens remained finite; the counts are preserved in run metadata rather than being hidden by a denominator clamp.",
            "",
        ]
    )
    text_dump(output_dir / "MCT_BASIS_REPORT.md", "\n".join(lines))


def _plot_summary(
    plots_dir: Path,
    official_vote_rows: list[dict[str, object]],
    logit_rows: list[dict[str, object]],
    mct_rows: list[dict[str, object]],
) -> None:
    haar = [f"haar_{index:02d}" for index in range(10)]
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(
        range(10),
        [_metric_lookup(official_vote_rows, name, "vote_spearman") for name in haar],
        marker="o",
        label="Official LaST",
    )
    axes[0].plot(
        range(10),
        [_metric_lookup(mct_rows, name, "vote_spearman") for name in haar],
        marker="s",
        label="PatchFinalLN",
    )
    axes[0].set_title("Vote-map Spearman")
    axes[0].set_xlabel("Haar rotation")
    axes[0].legend(fontsize=8)
    axes[1].plot(
        range(10),
        [_metric_lookup(official_vote_rows, name, "top10_jaccard") for name in haar],
        marker="o",
        label="Official LaST",
    )
    axes[1].plot(
        range(10),
        [_metric_lookup(mct_rows, name, "top10_jaccard") for name in haar],
        marker="s",
        label="PatchFinalLN",
    )
    axes[1].set_title("Top-10% vote Jaccard")
    axes[1].set_xlabel("Haar rotation")
    axes[2].plot(
        range(10),
        [_metric_lookup(logit_rows, name, "logit_cosine") for name in haar],
        marker="o",
    )
    axes[2].set_title("Official LaST logit cosine")
    axes[2].set_xlabel("Haar rotation")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "basis_sensitivity_summary.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    require_environment()
    args.repo_root = args.repo_root.expanduser().resolve()
    args.official_repo = _resolve(args.repo_root, args.official_repo)
    args.official_checkpoint = _resolve(args.repo_root, args.official_checkpoint)
    args.mct_checkpoint = _resolve(args.repo_root, args.mct_checkpoint)
    args.voc_root = _resolve(args.repo_root, args.voc_root)
    args.list_path = _resolve(args.repo_root, args.list_path)
    if args.test_log:
        args.test_log = _resolve(args.repo_root, args.test_log)
    if min(args.official_batch_size, args.mct_batch_size) < 1:
        raise ValueError("batch sizes must be positive")
    if args.num_workers < 0 or args.limit < 0 or args.haar_count != 10:
        raise ValueError("workers/limit invalid or Phase A does not have exactly 10 Haar rotations")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full Phase A is fixed to an available CUDA device")
    git = require_clean_tracked(args.repo_root)
    output_dir = create_output(args.output_dir)
    raw_dir = output_dir / "raw_outputs"
    plots_dir = output_dir / "selected_plots"
    raw_dir.mkdir()
    plots_dir.mkdir()
    log = RunLog(output_dir / "run.log")
    log(f"Phase A started: {command_line()}")
    before_hashes = {
        "official_last_checkpoint": sha256_file(args.official_checkpoint),
        "patchfinal_checkpoint": sha256_file(args.mct_checkpoint),
        "voc_val_list": sha256_file(args.list_path),
    }
    manifests = write_environment_manifests(output_dir)

    all_transforms = {}
    for dimension in (768, 384):
        transforms = generate_basis_transforms(
            dimension, base_seed=args.base_seed, haar_count=args.haar_count
        )
        matrix_path = raw_dir / f"basis_matrices_d{dimension}.npz"
        np.savez(matrix_path, **{item.name: item.matrix for item in transforms})
        all_transforms[str(dimension)] = {
            "transforms": transform_metadata(transforms),
            "matrix_archive": str(matrix_path),
            "matrix_archive_sha256": sha256_file(matrix_path),
        }
    json_dump(
        output_dir / "basis_transforms.json",
        {
            "schema_version": 1,
            "base_seed": args.base_seed,
            "haar_count": args.haar_count,
            "generator": "NumPy default_rng Gaussian QR with positive R diagonal",
            "transforms_by_dimension": all_transforms,
        },
    )

    official_metadata, official_vote_rows, logit_rows, official_invariance = _run_official(
        args=args,
        output_dir=output_dir,
        raw_dir=raw_dir,
        plots_dir=plots_dir,
        device=device,
        log=log,
    )
    csv_dump(output_dir / "last_vote_metrics.csv", official_vote_rows)
    csv_dump(output_dir / "logit_sensitivity.csv", logit_rows)
    mct_metadata, mct_rows, conv_invariance, affinity_invariance = _run_mct(
        args=args,
        output_dir=output_dir,
        raw_dir=raw_dir,
        plots_dir=plots_dir,
        device=device,
        log=log,
    )
    csv_dump(output_dir / "mct_vote_metrics.csv", mct_rows)
    checks = {
        "tolerance": 1e-5,
        "official_standard_classifier_max_abs_logit_error": official_invariance,
        "mct_conv_max_abs_cam_logit_error": conv_invariance,
        "mct_cosine_affinity_max_abs_error": affinity_invariance,
    }
    checks["official_standard_classifier_pass"] = max(official_invariance.values()) < 1e-5
    checks["mct_conv_pass"] = max(conv_invariance.values()) < 1e-5
    checks["mct_cosine_affinity_pass"] = max(affinity_invariance.values()) < 1e-5
    checks["passed"] = bool(
        checks["official_standard_classifier_pass"]
        and checks["mct_conv_pass"]
        and checks["mct_cosine_affinity_pass"]
    )
    json_dump(output_dir / "invariance_checks.json", checks)
    if not checks["passed"]:
        raise RuntimeError(f"mandatory Phase A invariance check failed: {checks}")
    _plot_summary(plots_dir, official_vote_rows, logit_rows, mct_rows)
    _write_reports(
        output_dir,
        official_metadata,
        official_vote_rows,
        logit_rows,
        mct_metadata,
        mct_rows,
        checks,
    )
    exact_commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Official source/checkpoint acquisition",
        "git clone --depth 1 https://github.com/ChengShiest/LAST-ViT.git hosts/LAST-ViT",
        "curl -L --fail --output checkpoints/last_vit/ViT_190k.pth https://github.com/ChengShiest/LAST-ViT/releases/download/weights2/ViT_190k.pth",
        "",
        "# Minimal added dependency; existing torch/torchvision/timm were not upgraded",
        f"{shutil.which('python') or 'python'} -m pip install omegaconf==2.3.0",
        "",
        "# Analysis",
        command_line(),
    ]
    if args.test_log:
        exact_commands.extend(
            ["", f"# Test record: {args.test_log}", f"# SHA256: {sha256_file(args.test_log)}"]
        )
    text_dump(output_dir / "exact_commands.sh", "\n".join(exact_commands) + "\n")
    (output_dir / "exact_commands.sh").chmod(0o755)
    after_hashes = {
        "official_last_checkpoint": sha256_file(args.official_checkpoint),
        "patchfinal_checkpoint": sha256_file(args.mct_checkpoint),
        "voc_val_list": sha256_file(args.list_path),
    }
    if before_hashes != after_hashes:
        raise RuntimeError("an immutable Phase A input changed during analysis")
    metadata = {
        "schema_version": 1,
        "phase": "A",
        "status": "complete",
        "started_or_recorded_at": timestamp(),
        "command": command_line(),
        "git": git,
        "environment": manifests,
        "test_log": (
            {"path": str(args.test_log), "sha256": sha256_file(args.test_log)}
            if args.test_log
            else None
        ),
        "input_hashes_before": before_hashes,
        "input_hashes_after": after_hashes,
        "source_immutable": before_hashes == after_hashes,
        "official_last": official_metadata,
        "patchfinal_mctformerplus": mct_metadata,
        "invariance_checks": checks,
    }
    json_dump(output_dir / "run_metadata.json", metadata)
    json_dump(
        output_dir / "completion.json",
        {
            "status": "complete",
            "phase": "A",
            "finished_at": timestamp(),
            "invariance_passed": True,
            "image_count_official": official_metadata["dataset"]["image_count"],
            "image_count_mct": mct_metadata["dataset"]["image_count"],
        },
    )
    log("Phase A complete")


if __name__ == "__main__":
    main()
