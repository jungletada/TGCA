#!/usr/bin/env python
"""Frozen E1/E2 diagnostics for native MCTformer+ final class/patch tokens.

This entry point intentionally observes only raw post-Block-12 tokens.  It
does not consume layer-wise attention, alter a model output, or save a token
cache.  VOC semantic masks enter only after every relation has been computed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from analysis.lazy_assignment.experiment2.bootstrap_experiment2 import (
    iter_all_and_label_strata,
    summarize_clustered,
)
from analysis.lazy_assignment.experiment2.common import VOC_CLASS_NAMES
from analysis.lazy_assignment.experiment2.metrics_region import (
    TOPK_RATIOS,
    map_overlap_metrics,
    region_map_metrics,
    spatial_spearman,
    stable_topk_mask,
)
from analysis.lazy_assignment.experiment2.metrics_shared_ownership import (
    shared_support_metrics,
)
from analysis.lazy_assignment.experiment2.patch_regions import (
    assign_pair_patch_regions,
    assign_patch_regions,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import VOCSemanticDataset
from analysis.spatial_graph_stability.basis import generate_basis_transforms
from analysis.spatial_graph_stability.provenance import (
    BOOTSTRAP_REPEATS,
    BOOTSTRAP_SEED,
    EXPECTED_VOC_IMAGES,
    RunLog,
    command_line,
    create_output,
    csv_dump,
    git_metadata,
    json_dump,
    require_clean_tracked,
    require_environment,
    sha256_file,
    text_dump,
    timestamp,
    write_environment_manifests,
)
from models.mctformer_plus import (
    build_mctformerplus,
    resolve_mctformerplus_checkpoint_variant,
    validate_mctformerplus_final_norm_checkpoint,
)

from .relations import (
    EPSILON,
    apply_shared_basis,
    classifier_relevance,
    final_token_relations,
    positive_channel_statistics,
    positive_coordinate_patch_statistics,
    spatial_probability,
    transformed_mean_readout,
)


NATIVE_CHECKPOINT_SHA256 = "aced3d3bd69c57782c8e85f1d10abd7ff7ab02504df92c2f2f3898defcf7a65a"
EMBED_DIM = 384
NUM_CLASSES = 20
PATCH_COUNT = 784
PATCH_GRID = (28, 28)
BASIS_SEED = 20260901

SEMANTIC_RELATIONS_E1 = ("s_last", "s_cosine", "patch_norm", "r_native_classifier")
SEMANTIC_RELATIONS_E2 = ("s_pos", "s_negmag")
SEMANTIC_METRICS = (
    "target_hit",
    "other_fg_hit",
    "background_hit",
    "target_top05_fraction",
    "other_fg_top05_fraction",
    "bg_top05_fraction",
    "target_tail_enrich_05",
    "other_fg_tail_enrich_05",
    "bg_tail_enrich_05",
    "target_top10_fraction",
    "other_fg_top10_fraction",
    "bg_top10_fraction",
    "target_tail_enrich_10",
    "other_fg_tail_enrich_10",
    "bg_tail_enrich_10",
    "target_top20_fraction",
    "other_fg_top20_fraction",
    "bg_top20_fraction",
    "target_tail_enrich_20",
    "other_fg_tail_enrich_20",
    "bg_tail_enrich_20",
    "auc_target_bg",
    "ap_target_bg",
    "auc_target_other",
    "ap_target_other",
    "target_mean",
    "other_fg_mean",
    "bg_mean",
    "target_median",
    "other_fg_median",
    "bg_median",
    "spatial_mass_target",
    "spatial_mass_other_fg",
    "spatial_mass_background",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "results/final_ln_ablation/20260906-mctformerplus-final-ln-matched-s0-0ef6bd8/"
            "checkpoints/baseline/mctformerplus_final.pth"
        ),
    )
    parser.add_argument("--voc-root", type=Path, default=Path("data/VOCdevkit/VOC2012"))
    parser.add_argument(
        "--list-path",
        type=Path,
        default=Path("data/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=BOOTSTRAP_REPEATS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--limit", type=int, default=0, help="deterministic smoke prefix")
    parser.add_argument("--test-log", type=Path)
    return parser.parse_args()


def _resolve(root: Path, value: Path) -> Path:
    return value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()


def _label_stratum(count: int) -> str:
    if count == 1:
        return "single_label"
    if count == 2:
        return "exactly_2_labels"
    if count >= 3:
        return "3plus_labels"
    raise ValueError("positive map requires at least one positive label")


def load_native_mctformer(checkpoint_path: Path) -> tuple[torch.nn.Module, dict[str, object]]:
    """Strictly load only the audited native MCTformer+-Small baseline."""

    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != NATIVE_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Final-token diagnostics require the fixed native MCTformer+ checkpoint; "
            f"got {checkpoint_hash}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    resolution = resolve_mctformerplus_checkpoint_variant(checkpoint, "mctformerplus")
    if resolution.get("variant") != "small":
        raise RuntimeError(f"expected MCTformer+-Small, got {resolution.get('variant')}")
    validate_mctformerplus_final_norm_checkpoint(
        checkpoint,
        expected=False,
        expected_patch=False,
        expected_last_mct=False,
        expected_class_stable_last=False,
    )
    attention = checkpoint.get("attention_normalization", {})
    bcss = checkpoint.get("bcss", {"variant": "e0"})
    psl = checkpoint.get("psl", {"variant": "baseline"})
    cti = checkpoint.get("cti_bgt", {"enabled": False})
    if (
        attention.get("mode", "vanilla") != "vanilla"
        or float(attention.get("gamma", 1.0)) != 1.0
        or bool(attention.get("relation_bias", False))
        or bcss.get("variant", "e0") != "e0"
        or psl.get("variant", "baseline") != "baseline"
        or bool(cti.get("enabled", False))
    ):
        raise RuntimeError("checkpoint violates native vanilla/E0/PSL-baseline contract")
    model = build_mctformerplus(
        "small",
        cam=True,
        num_classes=NUM_CLASSES,
        input_size=448,
        attention_normalization="vanilla",
        attention_gamma=1.0,
        bcss_variant="e0",
        psl_variant="baseline",
        cti_bgt=False,
        final_norm=False,
        patch_final_norm=False,
        last_mct=False,
        class_stable_last=False,
    )
    state = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"native checkpoint strict load failed: {incompatible}")
    if len(model.blocks) != 12 or model.embed_dim != EMBED_DIM or model.head.kernel_size != (3, 3):
        raise RuntimeError("unexpected native MCTformer+ architecture")
    metadata: dict[str, object] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "epoch": checkpoint.get("epoch"),
        "strict_state_load": True,
        "variant_resolution": resolution,
        "flags": {
            "final_norm": False,
            "patch_final_norm": False,
            "last_mct": False,
            "class_stable_last": False,
        },
        "method_contract": {
            "attention": attention,
            "bcss": bcss,
            "psl": psl,
            "cti_bgt": cti,
        },
        "training_spec": checkpoint.get("training_spec", {}),
        "head": {"kernel_size": list(model.head.kernel_size), "in_channels": int(model.head.in_channels)},
    }
    del checkpoint, state
    return model, metadata


def _raw_classifier_map(model: torch.nn.Module, patches: torch.Tensor) -> torch.Tensor:
    if patches.shape[1:] != (PATCH_COUNT, EMBED_DIM):
        raise RuntimeError(f"raw patch token shape must be [B,784,384], got {tuple(patches.shape)}")
    batch = patches.shape[0]
    grid = patches.reshape(batch, *PATCH_GRID, EMBED_DIM).permute(0, 3, 1, 2).contiguous()
    raw = model.head(grid)[:, :NUM_CLASSES].flatten(2)
    if raw.shape[1:] != (NUM_CLASSES, PATCH_COUNT) or not torch.isfinite(raw).all():
        raise RuntimeError("native raw 3x3 classifier map contract failed")
    return raw


def _soft_mass(scores: np.ndarray, regions: np.ndarray) -> dict[str, float]:
    probability = spatial_probability(torch.from_numpy(np.asarray(scores)[None, None]))[0, 0].numpy()
    result: dict[str, float] = {}
    for name, code in (("target", 0), ("other_fg", 1), ("background", 2)):
        result[f"spatial_mass_{name}"] = float(probability[np.asarray(regions).reshape(-1) == code].sum())
    return result


def _semantic_rows(
    *, image_id: str, image_index: int, class_id: int, label_count: int,
    region_codes: np.ndarray, maps: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relation, map_values in maps.items():
        values = np.asarray(map_values, dtype=np.float64).reshape(-1)
        metric = region_map_metrics(values, region_codes, grid_h=28, grid_w=28)
        metric.update(_soft_mass(values, region_codes))
        rows.append(
            {
                "image_id": image_id,
                "image_index": image_index,
                "class_id": class_id,
                "class_name": VOC_CLASS_NAMES[class_id],
                "num_positive_classes": label_count,
                "label_stratum": _label_stratum(label_count),
                "relation": relation,
                **metric,
            }
        )
    return rows


def _summarize_semantic(rows: Sequence[Mapping[str, object]], repeats: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    result: list[dict[str, object]] = []
    if frame.empty:
        return result
    for relation, subset in frame.groupby("relation", sort=True):
        for stratum, scoped in iter_all_and_label_strata(subset):
            result.extend(summarize_clustered(
                scoped, value_cols=SEMANTIC_METRICS,
                identity={"relation": relation, "scope": "all_positive_classes", "stratum": stratum},
                repeats=repeats, seed=seed,
            ))
        for class_id, class_rows in subset.groupby("class_id", sort=True):
            result.extend(summarize_clustered(
                class_rows, value_cols=SEMANTIC_METRICS,
                identity={"relation": relation, "scope": "per_class", "stratum": "all", "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)]},
                repeats=repeats, seed=seed, include_macro_class=False,
            ))
    return result


def _summary_by_columns(
    rows: Sequence[Mapping[str, object]], *, group_columns: Sequence[str], value_columns: Sequence[str],
    repeats: int, seed: int, strata: bool = False, per_class: bool = False,
) -> list[dict[str, object]]:
    frame = pd.DataFrame(rows)
    result: list[dict[str, object]] = []
    if frame.empty:
        return result
    for keys, subset in frame.groupby(list(group_columns), sort=True):
        identity = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        scopes = iter_all_and_label_strata(subset) if strata and "label_stratum" in subset else (("all", subset),)
        for stratum, scoped in scopes:
            result.extend(summarize_clustered(
                scoped, value_cols=value_columns,
                identity={**identity, "scope": "all_rows", "stratum": stratum},
                repeats=repeats, seed=seed,
            ))
        if per_class and "class_id" in subset:
            for class_id, class_rows in subset.groupby("class_id", sort=True):
                result.extend(summarize_clustered(
                    class_rows, value_cols=value_columns,
                    identity={**identity, "scope": "per_class", "stratum": "all", "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)]},
                    repeats=repeats, seed=seed, include_macro_class=False,
                ))
    return result


def _multilabel_rows(
    *, image_id: str, image_index: int, class_ids: Sequence[int], mask: torch.Tensor,
    label_count: int, maps: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    if len(class_ids) < 2:
        return []
    result: list[dict[str, object]] = []
    for first, second in itertools.combinations(class_ids, 2):
        pair = assign_pair_patch_regions(mask, int(first), int(second))["region_codes"].reshape(-1)
        eligible = pair != 5
        for relation, values in maps.items():
            a, b = np.asarray(values[first]), np.asarray(values[second])
            record: dict[str, object] = {
                "image_id": image_id,
                "image_index": image_index,
                "class_id": int(first),
                "class_name": VOC_CLASS_NAMES[int(first)],
                "other_class_id": int(second),
                "other_class_name": VOC_CLASS_NAMES[int(second)],
                "num_positive_classes": label_count,
                "label_stratum": _label_stratum(label_count),
                "relation": relation,
                "map_spearman": spatial_spearman(a[eligible], b[eligible]),
            }
            for ratio in TOPK_RATIOS:
                suffix = int(round(100 * ratio))
                record[f"top{suffix:02d}_jaccard"] = map_overlap_metrics(a, b, ratio=ratio, eligible=eligible)["topk_jaccard"]
            shared = shared_support_metrics(a, b, pair, ratio=0.10)
            for key, value in shared.items():
                if key not in ("topk_ratio",):
                    record[key] = value
            record["dominant_object_capture"] = max(
                float(record.get("shared_target_a_fraction", float("nan"))),
                float(record.get("shared_target_b_fraction", float("nan"))),
            )
            result.append(record)
    return result


def _summarize_multilabel(rows: Sequence[Mapping[str, object]], repeats: int, seed: int) -> list[dict[str, object]]:
    metrics = (
        "map_spearman", "top05_jaccard", "top10_jaccard", "top20_jaccard",
        "shared_target_a_fraction", "shared_target_b_fraction", "shared_other_fg_fraction",
        "shared_background_fraction", "dominant_object_capture",
    )
    return _summary_by_columns(rows, group_columns=("relation",), value_columns=metrics, repeats=repeats, seed=seed, strata=True)


def _present_absent_row(
    *, image_id: str, image_index: int, class_id: int, label_count: int, presence: str,
    relation: str, values: np.ndarray, foreground: np.ndarray, background: np.ndarray, valid: np.ndarray,
) -> dict[str, object]:
    score = np.asarray(values, dtype=np.float64).reshape(-1)
    selected = score[valid]
    probability = spatial_probability(torch.from_numpy(score[None, None]))[0, 0].numpy()
    entropy = -float((probability[valid] * np.log(np.maximum(probability[valid], 1e-300))).sum() / math.log(int(valid.sum())))
    top = stable_topk_mask(score, 0.10, valid)
    return {
        "image_id": image_id, "image_index": image_index, "class_id": class_id,
        "class_name": VOC_CLASS_NAMES[class_id], "num_positive_classes": label_count,
        "presence": presence, "relation": relation,
        "max_score": float(selected.max()), "top10_mean_score": float(score[top].mean()),
        "spatial_entropy": entropy, "foreground_mass": float(probability[foreground].sum()),
        "background_mass": float(probability[background].sum()),
    }


def _channel_patch_rows(
    *, image_id: str, image_index: int, class_id: int, label_count: int,
    regions: np.ndarray, u_pos: np.ndarray, c_sign: np.ndarray, s_pos: np.ndarray,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for signal, values in (("u_pos", u_pos), ("c_sign", c_sign), ("s_pos", s_pos)):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        means: dict[str, float] = {}
        for name, code in (("target", 0), ("other_fg", 1), ("background", 2), ("mixed", 3)):
            chosen = values[np.asarray(regions).reshape(-1) == code]
            means[name] = float(chosen.mean()) if chosen.size else float("nan")
            result.append({
                "image_id": image_id, "image_index": image_index, "class_id": class_id,
                "class_name": VOC_CLASS_NAMES[class_id], "num_positive_classes": label_count,
                "label_stratum": _label_stratum(label_count), "signal": signal, "region": name,
                "mean": means[name], "median": float(np.median(chosen)) if chosen.size else float("nan"),
                "q25": float(np.quantile(chosen, .25)) if chosen.size else float("nan"),
                "q75": float(np.quantile(chosen, .75)) if chosen.size else float("nan"),
            })
        for comparison, right in (("target_minus_background", "background"), ("target_minus_other_fg", "other_fg")):
            result.append({
                "image_id": image_id, "image_index": image_index, "class_id": class_id,
                "class_name": VOC_CLASS_NAMES[class_id], "num_positive_classes": label_count,
                "label_stratum": _label_stratum(label_count), "signal": signal, "region": comparison,
                "mean": means["target"] - means[right] if np.isfinite(means["target"]) and np.isfinite(means[right]) else float("nan"),
                "median": float("nan"), "q25": float("nan"), "q75": float("nan"),
            })
    return result


def _basis_rows(
    *, class_tokens: torch.Tensor, patch_tokens: torch.Tensor, original_s_last: torch.Tensor,
    original_s_pos: torch.Tensor, positive_class_ids: Sequence[Sequence[int]], image_ids: Sequence[str],
    image_offset: int, region_codes: Sequence[Mapping[int, np.ndarray]], label_counts: Sequence[int], transforms: Sequence[object],
) -> tuple[list[dict[str, object]], dict[str, float], list[dict[str, object]]]:
    """Compute exact E1 invariance and E2 coordinate-basis diagnostics in float64."""

    rows: list[dict[str, object]] = []
    semantic_rows: list[dict[str, object]] = []
    maximum = {"s_last": 0.0, "readout": 0.0, "permutation_s_pos": 0.0}
    classes64, patches64 = class_tokens.double(), patch_tokens.double()
    # Run both sides of the algebraic checks in float64.  The ordinary analysis
    # deliberately uses the model's float32 representation, but using its
    # already-rounded map as the reference here would turn an identity basis
    # change into a reduction-order test rather than a basis-invariance test.
    original_relation64 = final_token_relations(classes64, patches64)
    original_last64 = original_relation64["s_last"]
    original_pos64 = original_relation64["s_pos"]
    original_pos = original_s_pos.detach().cpu().numpy()
    for transform in transforms:
        matrix = torch.from_numpy(np.asarray(transform.matrix)).to(device=classes64.device, dtype=torch.float64)
        c_prime, p_prime = apply_shared_basis(classes64, patches64, matrix)
        relation = final_token_relations(c_prime, p_prime)
        last_error = float((relation["s_last"] - original_last64).abs().max().item())
        original_logits, transformed_logits = transformed_mean_readout(classes64, matrix)
        readout_error = float((original_logits - transformed_logits).abs().max().item())
        if transform.kind == "permutation":
            maximum["permutation_s_pos"] = max(
                maximum["permutation_s_pos"],
                float((relation["s_pos"] - original_pos64).abs().max().item()),
            )
        maximum["s_last"] = max(maximum["s_last"], last_error)
        maximum["readout"] = max(maximum["readout"], readout_error)
        transformed_pos = relation["s_pos"].detach().cpu().numpy()
        for local_index, positive_ids in enumerate(positive_class_ids):
            for class_id in positive_ids:
                base = original_pos[local_index, class_id]
                changed = transformed_pos[local_index, class_id]
                valid_regions = np.asarray(region_codes[local_index][int(class_id)]).reshape(-1)
                valid = valid_regions != 4
                overlap = map_overlap_metrics(base, changed, ratio=.10, eligible=valid)
                metric = region_map_metrics(changed, valid_regions, grid_h=28, grid_w=28)
                rows.append({
                    "image_id": image_ids[local_index], "image_index": image_offset + local_index,
                    "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                    "num_positive_classes": int(label_counts[local_index]), "label_stratum": _label_stratum(int(label_counts[local_index])),
                    "transform": transform.name, "transform_kind": transform.kind,
                    "s_last_max_abs_error": last_error, "readout_max_abs_error": readout_error,
                    "s_pos_map_spearman": overlap["spearman"], "s_pos_top10_jaccard": overlap["topk_jaccard"],
                    "s_pos_normalized_l1": float(np.abs(changed - base).sum() / np.abs(base).sum().clip(min=EPSILON)),
                    "s_pos_auc_target_bg": metric["auc_target_bg"], "s_pos_auc_target_other": metric["auc_target_other"],
                })
                semantic_rows.append({
                    "image_id": image_ids[local_index], "image_index": image_offset + local_index,
                    "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                    "num_positive_classes": int(label_counts[local_index]), "label_stratum": _label_stratum(int(label_counts[local_index])),
                    "transform": transform.name, "transform_kind": transform.kind,
                    "auc_target_bg": metric["auc_target_bg"], "auc_target_other": metric["auc_target_other"],
                })
    return rows, maximum, semantic_rows


def _cross_image_rows(records: Sequence[Mapping[str, object]], device: torch.device) -> list[dict[str, object]]:
    """Compare all positive-token anchors to same/different-class other images.

    Pair observations are reduced to one mean per anchor image/class before the
    bootstrap.  Thus neither individual token pairs nor patches are bootstrap
    units.
    """

    if not records:
        return []
    masks = torch.from_numpy(np.stack([np.asarray(row["positive_mask"], dtype=np.float32) for row in records])).to(device)
    tokens = torch.from_numpy(np.stack([np.asarray(row["positive_token"], dtype=np.float32) for row in records])).to(device)
    classes = torch.tensor([int(row["class_id"]) for row in records], device=device)
    images = torch.tensor([int(row["image_index"]) for row in records], device=device)
    counts = masks.sum(dim=1)
    intersection = masks @ masks.transpose(0, 1)
    union = counts[:, None] + counts[None, :] - intersection
    jaccard = intersection / union.clamp_min(1.0)
    binary_cosine = intersection / (counts[:, None] * counts[None, :]).sqrt().clamp_min(1.0)
    unit = torch.nn.functional.normalize(tokens, p=2, dim=-1, eps=EPSILON)
    token_cosine = unit @ unit.transpose(0, 1)
    different_image = images[:, None] != images[None, :]
    same = different_image & (classes[:, None] == classes[None, :])
    different = different_image & (classes[:, None] != classes[None, :])

    def reduce(values: torch.Tensor, eligible: torch.Tensor) -> np.ndarray:
        denom = eligible.sum(dim=1)
        output = (values * eligible).sum(dim=1) / denom.clamp_min(1)
        return output.masked_fill(denom == 0, float("nan")).detach().cpu().numpy()

    same_values = [reduce(values, same) for values in (jaccard, binary_cosine, token_cosine)]
    different_values = [reduce(values, different) for values in (jaccard, binary_cosine, token_cosine)]
    result: list[dict[str, object]] = []
    for index, record in enumerate(records):
        result.append({
            "image_id": record["image_id"], "image_index": record["image_index"],
            "class_id": record["class_id"], "class_name": record["class_name"],
            "comparison": "same_class_other_images", "binary_mask_jaccard": float(same_values[0][index]),
            "binary_mask_cosine": float(same_values[1][index]), "relu_token_cosine": float(same_values[2][index]),
        })
        result.append({
            "image_id": record["image_id"], "image_index": record["image_index"],
            "class_id": record["class_id"], "class_name": record["class_name"],
            "comparison": "different_class_other_images", "binary_mask_jaccard": float(different_values[0][index]),
            "binary_mask_cosine": float(different_values[1][index]), "relu_token_cosine": float(different_values[2][index]),
        })
    return result


def _paired_delta_rows(
    semantic_rows: Sequence[Mapping[str, object]], multilabel_rows: Sequence[Mapping[str, object]],
    repeats: int, seed: int,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    semantic = pd.DataFrame(semantic_rows)
    keys = ["image_id", "image_index", "class_id", "class_name", "num_positive_classes", "label_stratum"]
    delta_metrics = ("auc_target_bg", "auc_target_other", "target_tail_enrich_10", "bg_tail_enrich_10")
    left = semantic[semantic["relation"] == "s_last"][keys + list(delta_metrics)]
    right = semantic[semantic["relation"] == "s_pos"][keys + list(delta_metrics)]
    merged = left.merge(right, on=keys, suffixes=("_last", "_pos"), validate="one_to_one")
    for metric in delta_metrics:
        merged[metric] = pd.to_numeric(merged[f"{metric}_pos"], errors="coerce") - pd.to_numeric(merged[f"{metric}_last"], errors="coerce")
    for stratum, scoped in iter_all_and_label_strata(merged):
        result.extend(summarize_clustered(scoped, value_cols=delta_metrics, identity={"comparison": "s_pos_minus_s_last", "family": "semantic", "stratum": stratum}, repeats=repeats, seed=seed))
    overlap = pd.DataFrame(multilabel_rows)
    if not overlap.empty:
        pair_keys = keys + ["other_class_id", "other_class_name"]
        left_pair = overlap[overlap["relation"] == "s_last"][pair_keys + ["top10_jaccard"]]
        right_pair = overlap[overlap["relation"] == "s_pos"][pair_keys + ["top10_jaccard"]]
        paired = left_pair.merge(right_pair, on=pair_keys, suffixes=("_last", "_pos"), validate="one_to_one")
        paired["top10_jaccard"] = paired["top10_jaccard_pos"] - paired["top10_jaccard_last"]
        for stratum, scoped in iter_all_and_label_strata(paired):
            result.extend(summarize_clustered(scoped, value_cols=("top10_jaccard",), identity={"comparison": "s_pos_minus_s_last", "family": "multilabel", "stratum": stratum}, repeats=repeats, seed=seed))
    return result


def _confidence_rows(semantic_rows: Sequence[Mapping[str, object]], logits: Mapping[tuple[int, int], float], repeats: int, seed: int) -> list[dict[str, object]]:
    frame = pd.DataFrame([row for row in semantic_rows if row["relation"] == "s_pos"])
    frame["native_class_logit"] = [logits[(int(row.image_index), int(row.class_id))] for row in frame.itertuples()]
    order = np.argsort(frame["native_class_logit"].to_numpy(dtype=np.float64), kind="stable")
    quantile = np.empty(len(frame), dtype=np.int8)
    quantile[order] = np.minimum(4, 5 * np.arange(len(frame)) // max(1, len(frame)))
    frame["confidence_quintile"] = quantile
    output: list[dict[str, object]] = []
    for quintile, subset in frame.groupby("confidence_quintile", sort=True):
        output.extend(summarize_clustered(
            subset, value_cols=("auc_target_bg", "auc_target_other", "target_tail_enrich_10"),
            identity={"relation": "s_pos", "confidence_quintile": int(quintile)},
            repeats=repeats, seed=seed,
        ))
    return output


def _find_metric(rows: Sequence[Mapping[str, object]], relation: str, metric: str) -> float:
    selected = [row for row in rows if row.get("relation") == relation and row.get("scope") == "all_positive_classes" and row.get("stratum") == "all" and row.get("aggregation") == "micro" and row.get("metric") == metric]
    return float(selected[0]["estimate"]) if len(selected) == 1 else float("nan")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    """Write a valid header-only smoke artifact when a registered stratum is empty."""

    if rows:
        columns = list(fields)
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        csv_dump(path, rows, fields=columns)
    else:
        text_dump(path, ",".join(fields) + "\n")


def _format(value: object) -> str:
    return "NA" if value is None or not np.isfinite(float(value)) else f"{float(value):.6f}"


def _write_reports(
    output_dir: Path, *, model: Mapping[str, object], semantic_summary: Sequence[Mapping[str, object]],
    multilabel_summary: Sequence[Mapping[str, object]], channel_summary: Sequence[Mapping[str, object]],
    class_specificity: Sequence[Mapping[str, object]], patch_distribution: Sequence[Mapping[str, object]],
    basis: Mapping[str, object], deltas: Sequence[Mapping[str, object]], confidence: Sequence[Mapping[str, object]],
) -> None:
    e1 = [
        "# Final-Token Affinity Report", "", "## Frozen contract", "",
        f"- Native MCTformer+-Small checkpoint SHA256 `{model['checkpoint_sha256']}`; raw post-Block-12 class and patch tokens only.",
        "- No final LayerNorm, L2 normalization in S_last, projection, Q/K projection, attention-map analysis, CAM modification, selector, loss, or training was used.",
        "- `S_last = c^T p / sqrt(D)` is final-token compatibility, not Transformer attention. GT is used only after relation construction; intervals resample whole images.",
        "", "## E1 primary semantic diagnostics (micro)", "",
        "| Relation | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% | BG tail enrich@10% | C-PiM target hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for relation in SEMANTIC_RELATIONS_E1:
        values = [_find_metric(semantic_summary, relation, key) for key in ("auc_target_bg", "auc_target_other", "target_tail_enrich_10", "bg_tail_enrich_10", "target_hit")]
        e1.append(f"| {relation} | " + " | ".join(_format(value) for value in values) + " |")
    e1.extend(["", "## Final positive-class map overlap (micro)", "", "| Relation | Map Spearman | Top-10% Jaccard | Shared BG fraction | Dominant-object capture |", "|---|---:|---:|---:|---:|"])
    for relation in ("s_last", "s_pos"):
        def pick(metric: str) -> float:
            found = [row for row in multilabel_summary if row.get("relation") == relation and row.get("scope") == "all_rows" and row.get("stratum") == "all" and row.get("aggregation") == "micro" and row.get("metric") == metric]
            return float(found[0]["estimate"]) if len(found) == 1 else float("nan")
        e1.append(f"| {relation} | " + " | ".join(_format(pick(key)) for key in ("map_spearman", "top10_jaccard", "shared_background_fraction", "dominant_object_capture")) + " |")
    e1.extend([
        "", "## Basis regression", "",
        f"- Shared permutation/signed-permutation/five-Haar maximum `S_last` error: `{basis['maximum_s_last_error']:.3e}` (required <1e-5).",
        f"- Equivalent transformed mean-readout maximum logit error: `{basis['maximum_readout_error']:.3e}`.",
        "", "## Interpretation boundary", "",
        "These are frozen representation-level diagnostics. They establish neither attention behavior, CAM behavior, semantic leakage, causal shortcut use, nor an intervention effect.", "",
    ])
    text_dump(output_dir / "FINAL_TOKEN_AFFINITY_REPORT.md", "\n".join(e1))

    e2 = [
        "# Positive-Channel Relation Report", "", "## Frozen contract", "",
        "- `S_pos = ReLU(c)^T p / sqrt(D)` uses coordinate signs defined by the native mean class-token readout. It is a coordinate-dependent diagnostic, not an intrinsic rotation-invariant relation or a proposed method.",
        "- `S_last = S_pos - S_negmag` was numerically verified per forward batch; GT enters only after all maps are fixed.",
        "", "## E2 semantic diagnostics (micro)", "",
        "| Relation | Target-vs-BG AUROC | Target-vs-other-FG AUROC | Target tail enrich@10% | BG tail enrich@10% | C-PiM target hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for relation in ("s_last", *SEMANTIC_RELATIONS_E2):
        values = [_find_metric(semantic_summary, relation, key) for key in ("auc_target_bg", "auc_target_other", "target_tail_enrich_10", "bg_tail_enrich_10", "target_hit")]
        e2.append(f"| {relation} | " + " | ".join(_format(value) for value in values) + " |")
    e2.extend(["", "## Paired S_pos − S_last effects", "", "| Family | Metric | Δ | 95% CI |", "|---|---|---:|---:|"])
    for row in deltas:
        if row.get("aggregation") == "micro" and row.get("stratum") == "all":
            e2.append(f"| {row.get('family')} | {row.get('metric')} | {_format(row.get('estimate'))} | [{_format(row.get('ci_low'))}, {_format(row.get('ci_high'))}] |")
    e2.extend(["", "## Coordinate-basis dependence", "", "| Transform | S_pos map Spearman | Top-10% Jaccard | Normalized L1 |", "|---|---:|---:|---:|"])
    basis_frame = pd.DataFrame(channel_summary)
    for transform, subset in basis_frame.groupby("transform", sort=True):
        values = []
        for metric in ("s_pos_map_spearman", "s_pos_top10_jaccard", "s_pos_normalized_l1"):
            chosen = subset[(subset["metric"] == metric) & (subset["aggregation"] == "micro") & (subset["scope"] == "all_rows") & (subset["stratum"] == "all")]
            values.append(float(chosen.iloc[0]["estimate"]) if len(chosen) == 1 else float("nan"))
        e2.append(f"| {transform} | " + " | ".join(_format(value) for value in values) + " |")
    e2.extend(["", "## Interpretation boundary", "", "Positive-coordinate utility, if observed, is only a classifier-coordinate mechanism because signed permutations and rotations may change ReLU-coordinate selection. No selector, aggregation, loss, CAM modification, or training is implied by this analysis.", ""])
    text_dump(output_dir / "POSITIVE_CHANNEL_RELATION_REPORT.md", "\n".join(e2))


def _render_examples(
    *, output_dir: Path, dataset: VOCSemanticDataset, model: torch.nn.Module, device: torch.device,
    semantic_rows: Sequence[Mapping[str, object]], multilabel_rows: Sequence[Mapping[str, object]], logits: Mapping[tuple[int, int], float],
) -> list[dict[str, object]]:
    frame = pd.DataFrame([row for row in semantic_rows if row["relation"] == "s_last"])
    multi = pd.DataFrame([row for row in multilabel_rows if row["relation"] == "s_last"])
    candidates: list[tuple[str, pd.Series]] = []
    for stratum in ("single_label", "exactly_2_labels", "3plus_labels"):
        chosen = frame[frame["label_stratum"] == stratum].sort_values(["image_index", "class_id"], kind="stable")
        if not chosen.empty:
            candidates.append((stratum, chosen.iloc[0]))
    if not frame.empty:
        candidates.append(("small_target", frame.sort_values(["num_target", "image_index", "class_id"], kind="stable").iloc[0]))
        candidates.append(("large_target", frame.sort_values(["num_target", "image_index", "class_id"], ascending=[False, True, True], kind="stable").iloc[0]))
        success = frame.assign(logit=[logits[(int(row.image_index), int(row.class_id))] for row in frame.itertuples()])
        candidates.append(("classification_success", success.sort_values("logit", ascending=False, kind="stable").iloc[0]))
        candidates.append(("classification_failure", success.sort_values("logit", ascending=True, kind="stable").iloc[0]))
    if not multi.empty:
        pair = multi.sort_values("top10_jaccard", ascending=False, kind="stable").iloc[0]
        candidates.append(("late_overlap", frame[(frame["image_index"] == pair["image_index"]) & (frame["class_id"] == pair["class_id"])].iloc[0]))
    selected: list[tuple[str, pd.Series]] = []
    seen: set[tuple[int, int]] = set()
    for reason, row in candidates:
        key = (int(row["image_index"]), int(row["class_id"]))
        if key not in seen:
            seen.add(key); selected.append((reason, row))
    manifest: list[dict[str, object]] = []
    mean = np.asarray((0.485, 0.456, 0.406)); std = np.asarray((0.229, 0.224, 0.225))
    for reason, row in selected[:8]:
        index, class_id = int(row["image_index"]), int(row["class_id"])
        sample = dataset[index]; image = sample["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            classes, patches, _, _, auxiliary = model.forward_features(image, return_aux=True)
            if auxiliary.get("final_norm") or auxiliary.get("patch_final_norm"):
                raise RuntimeError("example renderer observed a non-native final normalization")
            relation = final_token_relations(classes, patches)
            raw = _raw_classifier_map(model, patches)
            relevance = classifier_relevance(raw)
            channel = positive_coordinate_patch_statistics(classes, patches)
        regions = assign_patch_regions(sample["mask"], class_id)["region_codes"].reshape(*PATCH_GRID)
        rgb = np.clip(sample["image"].permute(1, 2, 0).numpy() * std + mean, 0, 1)
        panels = (
            ("RGB", rgb, None), ("GT regions", regions, "tab10"),
            ("S_last", relation["s_last"][0, class_id].cpu().numpy().reshape(*PATCH_GRID), "magma"),
            ("S_pos", relation["s_pos"][0, class_id].cpu().numpy().reshape(*PATCH_GRID), "magma"),
            ("Native R_M", relevance[0, class_id].cpu().numpy().reshape(*PATCH_GRID), "magma"),
            ("U_pos", channel["u_pos"][0, class_id].cpu().numpy().reshape(*PATCH_GRID), "magma"),
            ("C_sign", channel["c_sign"][0, class_id].cpu().numpy().reshape(*PATCH_GRID), "viridis"),
        )
        figure, axes = plt.subplots(2, 4, figsize=(15, 7))
        for axis, (title, values, cmap) in zip(axes.flat, panels):
            axis.imshow(values, cmap=cmap); axis.set_title(title); axis.axis("off")
        axes.flat[-1].axis("off")
        figure.suptitle(f"{sample['name']} · {VOC_CLASS_NAMES[class_id]} · {reason}")
        figure.tight_layout()
        path = output_dir / f"{sample['name']}_{VOC_CLASS_NAMES[class_id]}_{reason}.png"
        figure.savefig(path, dpi=160, bbox_inches="tight"); plt.close(figure)
        manifest.append({"image_id": str(sample["name"]), "image_index": index, "class_id": class_id, "class_name": VOC_CLASS_NAMES[class_id], "selection_rule": reason, "path": str(path), "sha256": sha256_file(path)})
    return manifest


def main() -> None:
    args = parse_args(); require_environment()
    args.repo_root = args.repo_root.expanduser().resolve()
    for name in ("output_dir", "checkpoint", "voc_root", "list_path"):
        setattr(args, name, _resolve(args.repo_root, getattr(args, name)))
    if args.test_log:
        args.test_log = _resolve(args.repo_root, args.test_log)
    if args.limit < 0 or args.batch_size < 1 or args.num_workers < 0 or args.bootstrap_repeats < 1:
        raise ValueError("invalid batch/worker/limit/bootstrap argument")
    if not args.limit and (args.bootstrap_repeats != BOOTSTRAP_REPEATS or args.bootstrap_seed != BOOTSTRAP_SEED):
        raise ValueError("full analysis requires exactly 5,000 bootstrap resamples and seed 20260901")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("full final-token analysis requires CUDA")
    git = require_clean_tracked(args.repo_root)
    output_dir = create_output(args.output_dir); visual_dir = output_dir / "selected_visualizations"; visual_dir.mkdir()
    log = RunLog(output_dir / "run.log"); log(f"final-token E1/E2 started: {command_line()}")
    before = {"checkpoint": sha256_file(args.checkpoint), "voc_val_list": sha256_file(args.list_path)}
    environment = write_environment_manifests(output_dir)
    # Establish the exact native readout contract on CPU before moving the
    # immutable model to CUDA.  Two independent CUDA forwards can differ by a
    # few ulps because of reduction scheduling; CPU makes this a deterministic
    # check of the production ``x_cls.mean(dim=-1)`` path, rather than silently
    # loosening the pre-registered 1e-6 tolerance.
    model, model_metadata = load_native_mctformer(args.checkpoint); model.eval()
    dataset = VOCSemanticDataset(args.voc_root, args.list_path, input_size=448, limit=args.limit)
    if not args.limit and len(dataset) != EXPECTED_VOC_IMAGES:
        raise RuntimeError(f"expected all {EXPECTED_VOC_IMAGES} VOC val images, got {len(dataset)}")
    cpu_image = dataset[0]["image"].unsqueeze(0)
    with torch.inference_mode():
        cpu_classes, _, _, _, _ = model.forward_features(cpu_image, return_aux=True)
        cpu_native_logits = model(cpu_image, return_diagnostics=True)["class_logits"]
    native_readout_cpu_error = float((cpu_classes.mean(dim=-1) - cpu_native_logits).abs().max().item())
    if native_readout_cpu_error >= 1e-6:
        raise RuntimeError(
            "deterministic extracted mean logits differ from native logits: "
            f"{native_readout_cpu_error}"
        )
    log(f"deterministic CPU native mean-readout audit max_abs_error={native_readout_cpu_error:.9g}")
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    transforms = generate_basis_transforms(EMBED_DIM, base_seed=BASIS_SEED, haar_count=5)
    semantic_raw: list[dict[str, object]] = []; multilabel_raw: list[dict[str, object]] = []
    present_absent_raw: list[dict[str, object]] = []; positive_statistics_raw: list[dict[str, object]] = []
    channel_patch_raw: list[dict[str, object]] = []; decomposition_raw: list[dict[str, object]] = []
    same_image_specificity_raw: list[dict[str, object]] = []; positive_token_records: list[dict[str, object]] = []
    basis_raw: list[dict[str, object]] = []; basis_semantic_raw: list[dict[str, object]] = []
    logits: dict[tuple[int, int], float] = {}; max_logit_identity_error = 0.0; max_native_mean_equivalence_error = native_readout_cpu_error; max_decomposition_error = 0.0
    basis_max = {"s_last": 0.0, "readout": 0.0, "permutation_s_pos": 0.0}
    offset = 0
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            images = batch["image"].to(device, non_blocking=True)
            classes, patches, _, _, auxiliary = model.forward_features(images, return_aux=True)
            if classes.shape[1:] != (NUM_CLASSES, EMBED_DIM) or patches.shape[1:] != (PATCH_COUNT, EMBED_DIM):
                raise RuntimeError("raw final-token shape contract failed")
            if auxiliary.get("final_norm") or auxiliary.get("patch_final_norm") or auxiliary.get("last_mct") or auxiliary.get("class_stable_last"):
                raise RuntimeError("native baseline extraction observed an excluded variant")
            native_logits = classes.mean(dim=-1)
            relations = final_token_relations(classes, patches)
            decomposition = float((relations["s_last"] - (relations["s_pos"] - relations["s_negmag"])).abs().max().item())
            max_decomposition_error = max(max_decomposition_error, decomposition)
            if decomposition >= 1e-6:
                raise RuntimeError(f"positive/negative decomposition identity failed: {decomposition}")
            statistics = positive_channel_statistics(classes)
            identity_error = float(statistics["logit_identity_error"].abs().max().item())
            max_logit_identity_error = max(max_logit_identity_error, identity_error)
            if identity_error >= 1e-6:
                raise RuntimeError(f"positive/negative logit identity failed: {identity_error}")
            raw_classifier = _raw_classifier_map(model, patches)
            all_maps = {
                "s_last": relations["s_last"].detach().cpu().numpy(),
                "s_pos": relations["s_pos"].detach().cpu().numpy(),
                "s_negmag": relations["s_negmag"].detach().cpu().numpy(),
                "s_cosine": relations["s_cosine"].detach().cpu().numpy(),
                "patch_norm": relations["patch_norm"].detach().cpu().numpy(),
                "r_native_classifier": classifier_relevance(raw_classifier).detach().cpu().numpy(),
            }
            coordinate = positive_coordinate_patch_statistics(classes, patches)
            u_pos = coordinate["u_pos"].detach().cpu().numpy(); c_sign = coordinate["c_sign"].detach().cpu().numpy()
            labels = batch["label"].numpy() > 0
            native_np = native_logits.detach().cpu().numpy()
            cpos_np = relations["positive_channels"].detach().cpu().numpy()
            class_np = classes.detach().cpu().numpy(); masks = [batch["mask"][i] for i in range(images.shape[0])]
            positive_ids = [np.flatnonzero(labels[i]).tolist() for i in range(images.shape[0])]
            counts = [len(values) for values in positive_ids]
            per_class_regions = [
                {int(class_id): assign_patch_regions(masks[local_index], int(class_id))["region_codes"].reshape(-1)
                 for class_id in positive_ids[local_index]}
                for local_index in range(images.shape[0])
            ]
            local_basis, local_max, local_basis_semantic = _basis_rows(
                class_tokens=classes, patch_tokens=patches, original_s_last=relations["s_last"], original_s_pos=relations["s_pos"],
                positive_class_ids=positive_ids, image_ids=[str(value) for value in batch["name"]], image_offset=offset,
                region_codes=per_class_regions, label_counts=counts, transforms=transforms,
            )
            basis_raw.extend(local_basis); basis_semantic_raw.extend(local_basis_semantic)
            for key, value in local_max.items(): basis_max[key] = max(basis_max[key], value)
            for local_index, image_id in enumerate(batch["name"]):
                image_index = offset + local_index; label_count = counts[local_index]
                if label_count < 1:
                    raise RuntimeError(f"VOC image {image_id} has no image-level positive label")
                global_regions = assign_patch_regions(masks[local_index], 0)["region_codes"].reshape(-1)
                foreground = (global_regions == 0) | (global_regions == 1); background = global_regions == 2; valid = global_regions != 4
                for class_id in range(NUM_CLASSES):
                    presence = "positive" if labels[local_index, class_id] else "absent"
                    logits[(image_index, class_id)] = float(native_np[local_index, class_id])
                    positive_statistics_raw.append({
                        "image_id": str(image_id), "image_index": image_index, "class_id": class_id, "class_name": VOC_CLASS_NAMES[class_id],
                        "presence": presence, "num_positive_classes": label_count,
                        "positive_channel_fraction": float(statistics["positive_fraction"][local_index, class_id].item()),
                        "positive_contribution_mass": float(statistics["positive_mass"][local_index, class_id].item()),
                        "negative_contribution_mass": float(statistics["negative_mass"][local_index, class_id].item()),
                        "positive_negative_mass_ratio": float(statistics["positive_negative_ratio"][local_index, class_id].item()),
                        "native_class_logit": float(native_np[local_index, class_id]),
                        "D_logit_minus_pos_minus_neg_abs_error": float(statistics["logit_identity_error"][local_index, class_id].item()),
                    })
                    for relation in ("s_last", "s_pos"):
                        present_absent_raw.append(_present_absent_row(
                            image_id=str(image_id), image_index=image_index, class_id=class_id, label_count=label_count, presence=presence,
                            relation=relation, values=all_maps[relation][local_index, class_id], foreground=foreground, background=background, valid=valid,
                        ))
                for class_id in positive_ids[local_index]:
                    regions = per_class_regions[local_index][int(class_id)]
                    semantic_raw.extend(_semantic_rows(
                        image_id=str(image_id), image_index=image_index, class_id=int(class_id), label_count=label_count, region_codes=regions,
                        maps={name: all_maps[name][local_index, class_id] for name in (*SEMANTIC_RELATIONS_E1, *SEMANTIC_RELATIONS_E2)},
                    ))
                    channel_patch_raw.extend(_channel_patch_rows(
                        image_id=str(image_id), image_index=image_index, class_id=int(class_id), label_count=label_count, regions=regions,
                        u_pos=u_pos[local_index, class_id], c_sign=c_sign[local_index, class_id], s_pos=all_maps["s_pos"][local_index, class_id],
                    ))
                    for term in ("s_pos", "s_negmag", "s_last"):
                        metric = region_map_metrics(all_maps[term][local_index, class_id], regions, grid_h=28, grid_w=28)
                        decomposition_raw.append({
                            "image_id": str(image_id), "image_index": image_index, "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                            "num_positive_classes": label_count, "label_stratum": _label_stratum(label_count), "term": term,
                            "target_mean": metric["target_mean"], "other_fg_mean": metric["other_fg_mean"], "bg_mean": metric["bg_mean"],
                            "auc_target_bg": metric["auc_target_bg"], "auc_target_other": metric["auc_target_other"],
                        })
                    positive_token_records.append({
                        "image_id": str(image_id), "image_index": image_index, "class_id": int(class_id), "class_name": VOC_CLASS_NAMES[int(class_id)],
                        "positive_mask": class_np[local_index, class_id] > 0, "positive_token": cpos_np[local_index, class_id],
                    })
                for first, second in itertools.combinations(positive_ids[local_index], 2):
                    mask_first = class_np[local_index, first] > 0; mask_second = class_np[local_index, second] > 0
                    intersection = int(np.logical_and(mask_first, mask_second).sum()); union = int(np.logical_or(mask_first, mask_second).sum())
                    same_image_specificity_raw.append({
                        "image_id": str(image_id), "image_index": image_index, "class_id": int(first), "class_name": VOC_CLASS_NAMES[int(first)],
                        "other_class_id": int(second), "other_class_name": VOC_CLASS_NAMES[int(second)], "comparison": "positive_classes_same_image",
                        "binary_mask_jaccard": intersection / union if union else 1.0,
                        "binary_mask_cosine": intersection / math.sqrt(max(1, int(mask_first.sum()) * int(mask_second.sum()))),
                        "relu_token_cosine": float(np.dot(cpos_np[local_index, first], cpos_np[local_index, second]) / (np.linalg.norm(cpos_np[local_index, first]) * np.linalg.norm(cpos_np[local_index, second]) + EPSILON)),
                    })
                multilabel_raw.extend(_multilabel_rows(
                    image_id=str(image_id), image_index=image_index, class_ids=positive_ids[local_index], mask=masks[local_index], label_count=label_count,
                    maps={"s_last": all_maps["s_last"][local_index], "s_pos": all_maps["s_pos"][local_index]},
                ))
            offset += images.shape[0]
            if batch_number == 1 or batch_number % 10 == 0 or offset == len(dataset):
                log(f"extracted raw final-token E1/E2 diagnostics for {offset}/{len(dataset)} images")
    if offset != len(dataset):
        raise RuntimeError("frozen extraction did not cover the requested VOC images")
    if basis_max["s_last"] >= 1e-5:
        raise RuntimeError(f"shared-basis S_last invariance failed: {basis_max['s_last']}")
    if basis_max["permutation_s_pos"] >= 1e-5:
        raise RuntimeError(f"ordinary permutation should preserve S_pos: {basis_max['permutation_s_pos']}")
    log("frozen extraction complete; computing whole-image clustered bootstrap summaries")
    semantic_summary = _summarize_semantic(semantic_raw, args.bootstrap_repeats, args.bootstrap_seed)
    multilabel_summary = _summarize_multilabel(multilabel_raw, args.bootstrap_repeats, args.bootstrap_seed)
    positive_statistics_summary = _summary_by_columns(
        positive_statistics_raw, group_columns=("presence",),
        value_columns=("positive_channel_fraction", "positive_contribution_mass", "negative_contribution_mass", "positive_negative_mass_ratio", "native_class_logit", "D_logit_minus_pos_minus_neg_abs_error"),
        repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, per_class=True,
    )
    present_absent_summary = _summary_by_columns(
        present_absent_raw, group_columns=("relation", "presence"),
        value_columns=("max_score", "top10_mean_score", "spatial_entropy", "foreground_mass", "background_mass"),
        repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, per_class=True,
    )
    patch_distribution_summary = _summary_by_columns(
        channel_patch_raw, group_columns=("signal", "region"), value_columns=("mean", "median", "q25", "q75"),
        repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, strata=True, per_class=True,
    )
    decomposition_summary = _summary_by_columns(
        decomposition_raw, group_columns=("term",), value_columns=("target_mean", "other_fg_mean", "bg_mean", "auc_target_bg", "auc_target_other"),
        repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, strata=True, per_class=True,
    )
    class_specificity_raw = same_image_specificity_raw + _cross_image_rows(positive_token_records, device)
    class_specificity_summary = _summary_by_columns(
        class_specificity_raw, group_columns=("comparison",), value_columns=("binary_mask_jaccard", "binary_mask_cosine", "relu_token_cosine"),
        repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, per_class=True,
    )
    basis_summary = _summary_by_columns(
        basis_raw, group_columns=("transform", "transform_kind"), value_columns=("s_last_max_abs_error", "readout_max_abs_error", "s_pos_map_spearman", "s_pos_top10_jaccard", "s_pos_normalized_l1", "s_pos_auc_target_bg", "s_pos_auc_target_other"),
        repeats=args.bootstrap_repeats, seed=args.bootstrap_seed, strata=True,
    )
    confidence_summary = _confidence_rows(semantic_raw, logits, args.bootstrap_repeats, args.bootstrap_seed)
    deltas = _paired_delta_rows(semantic_raw, multilabel_raw, args.bootstrap_repeats, args.bootstrap_seed)
    basis_payload = {
        "schema_version": 1, "shared_basis_seed": BASIS_SEED,
        "transforms": [transform.metadata() for transform in transforms],
        "maximum_s_last_error": basis_max["s_last"], "maximum_readout_error": basis_max["readout"],
        "maximum_permutation_s_pos_error": basis_max["permutation_s_pos"],
        "s_last_invariance_passed": basis_max["s_last"] < 1e-5,
        "permutation_s_pos_passed": basis_max["permutation_s_pos"] < 1e-5,
        "positive_channel_rotation_note": "Signed permutations and Haar rotations are recorded as basis-dependence diagnostics, not failures.",
    }
    json_dump(output_dir / "basis_invariance_checks.json", basis_payload)
    semantic_fields = ("relation", "scope", "stratum", "class_id", "class_name", "aggregation", "metric", "estimate", "ci_low", "ci_high", "num_images", "num_images_total", "num_rows", "num_rows_total", "num_classes", "bootstrap_repeats", "bootstrap_valid_repeats", "bootstrap_valid_fraction", "bootstrap_seed", "bootstrap_base_seed")
    generic_fields = tuple(dict.fromkeys([*(semantic_summary[0].keys() if semantic_summary else ()), "relation", "scope", "stratum", "class_id", "class_name", "aggregation", "metric", "estimate", "ci_low", "ci_high", "num_images", "num_images_total", "num_rows", "num_rows_total", "num_classes", "bootstrap_repeats", "bootstrap_valid_repeats", "bootstrap_valid_fraction", "bootstrap_seed", "bootstrap_base_seed"]))
    _write_csv(output_dir / "final_token_semantic_metrics.csv", semantic_summary, semantic_fields)
    _write_csv(output_dir / "final_token_multilabel_overlap.csv", multilabel_summary, generic_fields)
    _write_csv(output_dir / "final_token_norm_cosine_decomposition.csv", [row for row in semantic_summary if row.get("relation") in ("s_last", "s_cosine", "patch_norm", "r_native_classifier")], semantic_fields)
    _write_csv(output_dir / "final_token_present_absent.csv", present_absent_summary, generic_fields)
    _write_csv(output_dir / "positive_channel_statistics.csv", positive_statistics_summary, generic_fields)
    _write_csv(output_dir / "positive_channel_class_specificity.csv", class_specificity_summary, generic_fields)
    _write_csv(output_dir / "positive_channel_patch_distributions.csv", patch_distribution_summary, generic_fields)
    _write_csv(output_dir / "positive_channel_semantic_metrics.csv", [row for row in semantic_summary if row.get("relation") in SEMANTIC_RELATIONS_E2], semantic_fields)
    _write_csv(output_dir / "positive_channel_decomposition.csv", decomposition_summary, generic_fields)
    _write_csv(output_dir / "positive_channel_basis_dependence.csv", basis_summary, generic_fields)
    _write_csv(output_dir / "class_confidence_stratification.csv", confidence_summary, generic_fields)
    _write_csv(output_dir / "bootstrap_deltas.csv", deltas, generic_fields)
    visual_manifest = _render_examples(output_dir=visual_dir, dataset=dataset, model=model, device=device, semantic_rows=semantic_raw, multilabel_rows=multilabel_raw, logits=logits)
    json_dump(visual_dir / "selection_manifest.json", {"examples": visual_manifest})
    _write_reports(output_dir, model=model_metadata, semantic_summary=semantic_summary, multilabel_summary=multilabel_summary, channel_summary=basis_summary, class_specificity=class_specificity_summary, patch_distribution=patch_distribution_summary, basis=basis_payload, deltas=deltas, confidence=confidence_summary)
    after = {"checkpoint": sha256_file(args.checkpoint), "voc_val_list": sha256_file(args.list_path)}
    if before != after:
        raise RuntimeError("immutable final-token analysis input changed during execution")
    config = {
        "schema_version": 1, "experiment": "native MCTformer+ final-token and positive-channel diagnostics",
        "token_source": "raw post-Block-12 residual class/patch tokens", "input_size": 448,
        "num_classes": NUM_CLASSES, "embed_dim": EMBED_DIM, "patch_grid": list(PATCH_GRID), "patch_count": PATCH_COUNT,
        "s_last": "c^T p / sqrt(D)", "s_pos": "ReLU(c)^T p / sqrt(D)", "s_negmag": "ReLU(-c)^T p / sqrt(D)",
        "native_classifier_reference": "ReLU(H_3x3(P)) / max_spatial ReLU(H_3x3(P))", "gt_use": "diagnostic evaluation only after relation construction",
        "prohibited": ["training", "FinalLN", "PatchFinalLN", "LaST pooling", "selector", "aggregation", "loss change", "CAM modification", "per-layer A_c2p analysis"],
        "checkpoint": model_metadata, "bootstrap": {"repeats": args.bootstrap_repeats, "seed": args.bootstrap_seed, "unit": "whole image"},
        "dataset": {"name": "PASCAL VOC 2012 val", "image_count": len(dataset), "transform": "Experiment 2 deterministic 448 resize/center crop with nearest mask geometry"},
    }
    json_dump(output_dir / "config.json", config)
    metadata = {
        "schema_version": 1, "status": "complete", "finished_at": timestamp(), "command": command_line(), "git": git,
        "environment": environment, "input_hashes_before": before, "input_hashes_after": after, "source_immutable": before == after,
        "checkpoint_audit": model_metadata, "numeric_checks": {"native_mean_logit_max_abs_error": max_native_mean_equivalence_error, "D_logit_positive_negative_identity_max_abs_error": max_logit_identity_error, "s_last_positive_negative_identity_max_abs_error": max_decomposition_error, **basis_payload},
        "statistics": config["bootstrap"], "test_log": {"path": str(args.test_log), "sha256": sha256_file(args.test_log)} if args.test_log else None,
        "handoff_note": "docs/CHAT_HANDOFF.md and docs/RESEARCH_PLAN_FULL.md were absent at task start; live Git/tmux/result state was used.",
        "raw_token_cache_saved": False,
    }
    json_dump(output_dir / "git_metadata.json", metadata)
    text_dump(output_dir / "exact_commands.sh", "#!/usr/bin/env bash\nset -euo pipefail\n\n" + command_line() + "\n")
    (output_dir / "exact_commands.sh").chmod(0o755)
    log("final-token E1/E2 complete")


if __name__ == "__main__":
    main()
