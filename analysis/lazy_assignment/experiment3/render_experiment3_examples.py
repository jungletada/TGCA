#!/usr/bin/env python3
"""Render deterministic, rule-selected examples for Experiment 3.

Selections are made from the pre-registered primary paired contrasts before any
image is opened: MCTformer+ A ``both_removed - raw`` positive-map overlap,
B ``B1 - B0`` per-image raw-CAM mIoU, and C ``C4 - C0`` per-image raw-CAM
mIoU.  Each validation receives the maximum, lower median, and minimum image,
with lexical ``image_id`` tie-breaking.  These panels are diagnostic examples,
not a best-case gallery and not a proposed method.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.evaluation_metrics import (  # noqa: E402
    iou_from_confusion,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    patch_label_counts,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (  # noqa: E402
    build_joint_transform,
    resolve_semantic_mask_path,
)
from analysis.lazy_assignment.experiment3.common import (  # noqa: E402
    EXPECTED_PATCHES,
    json_dump,
    read_json,
    require_tgca_repro,
    sha256_file,
    timestamp,
)
from analysis.lazy_assignment.experiment3.generate_experiment3_report import (  # noqa: E402
    ValidatedAnalyses,
    validate_analysis_roots,
)
from analysis.lazy_assignment.visualize_patch_score import (  # noqa: E402
    VOC_CLASS_NAMES,
    input_tensor_to_rgb,
)


SELECTION_RULE = (
    "maximum, lower median, minimum of the validation's primary per-image "
    "paired delta; lexical image_id tie-break"
)
RANK_ROLES = ("maximum", "lower_median", "minimum")
GT_COLORMAP = ListedColormap(("#d9d9d9", "#22aa66", "#dd8844", "#222222"))
GT_NORM = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), GT_COLORMAP.N)


@dataclass(frozen=True)
class ManifestArtifact:
    path: Path
    sha256: str
    positive_class_ids: tuple[int, ...]
    source_signal_sha256: str = ""


@dataclass(frozen=True)
class Selection:
    validation: str
    rank_role: str
    image_id: str
    delta: float
    focal_class_id: int
    companion_class_id: Optional[int]


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if not result:
        raise ValueError("empty safe filename")
    return result


def select_extreme_image_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Select max/lower-median/min with an explicit stable tie-break."""

    required = {"image_id", "delta"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"selection frame lacks {sorted(required - set(frame.columns))}"
        )
    values = frame.loc[:, ["image_id", "delta"]].copy()
    values["image_id"] = values["image_id"].astype(str)
    values["delta"] = pd.to_numeric(values["delta"], errors="raise")
    if values["image_id"].duplicated().any() or not np.isfinite(values["delta"]).all():
        raise ValueError("selection values must be unique by image and finite")
    if len(values) < 3:
        raise RuntimeError("rule-selected examples require at least three images")
    ordered = values.sort_values(["delta", "image_id"], kind="mergesort").reset_index(
        drop=True
    )
    minimum_value = float(ordered.iloc[0]["delta"])
    maximum_value = float(ordered.iloc[-1]["delta"])
    minimum = ordered[ordered["delta"] == minimum_value].iloc[0]
    maximum = ordered[ordered["delta"] == maximum_value].iloc[0]
    median = ordered.iloc[(len(ordered) - 1) // 2]
    result = pd.DataFrame(
        [
            {**maximum.to_dict(), "rank_role": "maximum"},
            {**median.to_dict(), "rank_role": "lower_median"},
            {**minimum.to_dict(), "rank_role": "minimum"},
        ]
    )
    if result["image_id"].duplicated().any():
        raise RuntimeError("extreme selection produced duplicate images")
    return result.loc[:, ["rank_role", "image_id", "delta"]]


def _decode_confusion(value: object) -> np.ndarray:
    blob = bytes(value)
    if len(blob) != 21 * 21 * 8:
        raise ValueError("invalid encoded 21x21 confusion")
    return np.frombuffer(blob, dtype="<i8").reshape(21, 21).copy()


def per_image_confusion_delta(
    frame: pd.DataFrame, *, baseline: str, comparison: str
) -> pd.DataFrame:
    """Return comparison-minus-baseline single-image VOC mIoU."""

    required = {"image_id", "variant_code", "confusion"}
    if not required.issubset(frame.columns):
        raise ValueError(
            f"confusion frame lacks {sorted(required - set(frame.columns))}"
        )
    selected = frame[frame["variant_code"].isin((baseline, comparison))]
    records = []
    for image_id, group in selected.groupby("image_id", sort=True):
        if set(group["variant_code"]) != {baseline, comparison} or len(group) != 2:
            raise RuntimeError(f"unpaired confusion rows for image {image_id}")
        metrics = {}
        for row in group.itertuples():
            _, metrics[row.variant_code] = iou_from_confusion(
                _decode_confusion(row.confusion)
            )
        records.append(
            {
                "image_id": str(image_id),
                "delta": float(metrics[comparison] - metrics[baseline]),
            }
        )
    return pd.DataFrame.from_records(records)


def _a_image_deltas(root: Path) -> pd.DataFrame:
    frame = pd.read_parquet(root / "canonical" / "positive_map_overlap.parquet")
    selected = frame[
        (frame["model"] == "mctformer_plus")
        & frame["layer"].isin((10, 11, 12))
        & np.isclose(frame["topk_ratio"], 0.10)
        & frame["variant"].isin(("raw", "both_removed"))
    ]
    index = ["image_id", "class_a", "class_b", "layer", "topk_ratio"]
    pivot = selected.pivot(index=index, columns="variant", values="topk_jaccard")
    if not {"raw", "both_removed"}.issubset(pivot.columns):
        raise RuntimeError("Validation A primary overlap contrast is incomplete")
    delta = (pivot["both_removed"] - pivot["raw"]).rename("delta").reset_index()
    return delta.groupby("image_id", as_index=False, sort=True)["delta"].mean()


def _bc_image_deltas(root: Path, *, baseline: str, comparison: str) -> pd.DataFrame:
    frame = pd.read_parquet(root / "canonical_image_cam_t045.parquet")
    selected = frame[
        (frame["model"] == "mctformer_plus") & np.isclose(frame["threshold"], 0.45)
    ]
    return per_image_confusion_delta(selected, baseline=baseline, comparison=comparison)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"non-object JSONL row {path}:{number}")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"empty manifest: {path}")
    return rows


def _resolved_child(root: Path, relative: object) -> Path:
    root = root.resolve()
    path = (root / str(relative)).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"manifest path escapes {root}: {relative}")
    return path


def _manifest_index(
    root: Path, *, manifest: Optional[Path] = None, path_field: str
) -> Mapping[str, ManifestArtifact]:
    root = root.expanduser().resolve()
    rows = _read_jsonl(manifest or (root / "manifest.jsonl"))
    result = {}
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if not image_id or image_id in result:
            raise RuntimeError(f"duplicate/empty image in {root} manifest")
        positive = tuple(int(value) for value in row.get("positive_class_ids", ()))
        path = _resolved_child(root, row.get(path_field, ""))
        result[image_id] = ManifestArtifact(
            path=path,
            sha256=str(row.get("artifact_sha256", "")),
            positive_class_ids=positive,
            source_signal_sha256=str(row.get("source_signal_sha256", "")),
        )
    return result


def _analysis_run_root(
    analyses: ValidatedAnalyses, validation: str, model: str = "mctformer_plus"
) -> Path:
    metadata = analyses.metadata[validation]
    field = "input_runs" if validation == "A" else "run_roots"
    roots = metadata.get(field)
    if not isinstance(roots, Mapping) or not isinstance(roots.get(model), str):
        raise RuntimeError(f"Validation {validation} metadata lacks {field}/{model}")
    root = Path(str(roots[model])).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _build_selections(analyses: ValidatedAnalyses) -> Mapping[str, list[Selection]]:
    deltas = {
        "A": _a_image_deltas(analyses.roots["A"]),
        "B": _bc_image_deltas(analyses.roots["B"], baseline="B0", comparison="B1"),
        "C": _bc_image_deltas(analyses.roots["C"], baseline="C0", comparison="C4"),
    }
    run_indices = {
        name: _manifest_index(
            _analysis_run_root(analyses, name),
            path_field="signal_path" if name == "A" else "artifact_path",
        )
        for name in ("A", "B")
    }
    c_root = _analysis_run_root(analyses, "C")
    run_indices["C"] = _manifest_index(
        c_root,
        manifest=c_root / "signals" / "C0" / "manifest.jsonl",
        path_field="artifact_path",
    )
    output: dict[str, list[Selection]] = {}
    for validation, frame in deltas.items():
        chosen = select_extreme_image_deltas(frame)
        values = []
        for row in chosen.itertuples():
            artifact = run_indices[validation].get(str(row.image_id))
            if artifact is None or not artifact.positive_class_ids:
                raise RuntimeError(
                    f"selected {validation} image lacks positive-class provenance: {row.image_id}"
                )
            positives = sorted(artifact.positive_class_ids)
            companion = positives[1] if validation == "A" else None
            if validation == "A" and len(positives) < 2:
                raise RuntimeError("Validation A overlap example must be multi-label")
            values.append(
                Selection(
                    validation=validation,
                    rank_role=str(row.rank_role),
                    image_id=str(row.image_id),
                    delta=float(row.delta),
                    focal_class_id=positives[0],
                    companion_class_id=companion,
                )
            )
        output[validation] = values
    return output


def _load_transformed_input(
    voc_root: Path, image_id: str
) -> tuple[np.ndarray, np.ndarray, Mapping[str, object], Path, Path]:
    jpeg = (voc_root / "JPEGImages" / f"{image_id}.jpg").resolve()
    mask_path = resolve_semantic_mask_path(voc_root, image_id).resolve()
    if not jpeg.is_file():
        raise FileNotFoundError(jpeg)
    with Image.open(jpeg) as source:
        image = source.convert("RGB")
    with Image.open(mask_path) as source:
        mask = source.copy()
    image_tensor, mask_tensor, geometry = build_joint_transform(448)(image, mask)
    return (
        input_tensor_to_rgb(image_tensor),
        mask_tensor.numpy(),
        geometry,
        jpeg,
        mask_path,
    )


def _target_gt(mask: np.ndarray, class_id: int) -> np.ndarray:
    output = np.zeros(mask.shape, dtype=np.uint8)
    output[(mask >= 1) & (mask <= 20)] = 2
    output[mask == class_id + 1] = 1
    output[mask == 255] = 3
    return output


def _load_npz_checked(artifact: ManifestArtifact) -> Mapping[str, np.ndarray]:
    if not artifact.path.is_file() or sha256_file(artifact.path) != artifact.sha256:
        raise RuntimeError(f"manifested artifact hash mismatch: {artifact.path}")
    with np.load(artifact.path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _maps_for_selection(
    selection: Selection,
    analyses: ValidatedAnalyses,
    exp2_index: Mapping[str, ManifestArtifact],
    mask: np.ndarray,
) -> tuple[list[tuple[str, np.ndarray, str]], list[Path]]:
    validation = selection.validation
    class_id = selection.focal_class_id
    controls: list[Path] = []
    exp2 = exp2_index[selection.image_id]
    source = _load_npz_checked(exp2)
    controls.append(exp2.path)
    source_positive = set(np.asarray(source["positive_class_ids"], dtype=np.int64))
    if class_id not in source_positive:
        raise RuntimeError(f"selected class is not positive in source: {class_id}")
    observed_counts = patch_label_counts(mask, 16)
    if not np.array_equal(observed_counts, np.asarray(source["patch_label_counts"])):
        raise RuntimeError(f"matched GT geometry differs for {selection.image_id}")

    if validation == "A":
        run_root = _analysis_run_root(analyses, "A")
        artifact = _manifest_index(run_root, path_field="signal_path")[
            selection.image_id
        ]
        arrays = _load_npz_checked(artifact)
        controls.append(artifact.path)
        if str(np.asarray(arrays["source_signal_sha256"]).item()) != exp2.sha256:
            raise RuntimeError("Validation A/Experiment 2 source hash linkage failed")
        layers = np.asarray(arrays["control_layer_ids"], dtype=np.int64)
        indices = np.flatnonzero(layers == 12)
        if len(indices) != 1:
            raise RuntimeError("Validation A control arrays lack a unique L12")
        layer = int(indices[0])
        maps = [
            ("L12 raw", arrays["raw_control_all"][layer, class_id], "signed"),
            (
                "L12 both removed",
                arrays["both_removed_control_all"][layer, class_id],
                "signed",
            ),
            (
                "L12 pre-attn norm",
                arrays["feature_norm_control_all"][layer, class_id],
                "signed",
            ),
            ("L12 QK energy", arrays["qk_control_all"][layer, class_id], "signed"),
            (
                "L12 A_c2p",
                arrays["attention_conditional_control_all"][layer, class_id],
                "positive",
            ),
        ]
    elif validation == "B":
        run_root = _analysis_run_root(analyses, "B")
        artifact = _manifest_index(run_root, path_field="artifact_path")[
            selection.image_id
        ]
        arrays = _load_npz_checked(artifact)
        controls.append(artifact.path)
        if str(np.asarray(arrays["source_signal_sha256"]).item()) != exp2.sha256:
            raise RuntimeError("Validation B/Experiment 2 source hash linkage failed")
        positive = np.asarray(arrays["positive_class_ids"], dtype=np.int64)
        local = _class_offset(positive, class_id)
        variants = [str(value) for value in arrays["variant_codes"]]
        lookup = {code: index for index, code in enumerate(variants)}
        b0 = arrays["final_cam"][lookup["B0"], local]
        b1 = arrays["final_cam"][lookup["B1"], local]
        b4 = arrays["final_cam"][lookup["B4"], local]
        maps = [
            ("B0 native final", b0, "positive"),
            ("B1 L10 final", b1, "positive"),
            ("B4 L10-L11 final", b4, "positive"),
            ("B1 - B0", b1 - b0, "difference"),
            ("B4 - B0", b4 - b0, "difference"),
        ]
    else:
        run_root = _analysis_run_root(analyses, "C")
        variant_arrays = {}
        for variant in ("C0", "C4"):
            variant_root = run_root / "signals" / variant
            artifact = _manifest_index(
                run_root,
                manifest=variant_root / "manifest.jsonl",
                path_field="artifact_path",
            )[selection.image_id]
            variant_arrays[variant] = _load_npz_checked(artifact)
            controls.append(artifact.path)
            if (
                str(np.asarray(variant_arrays[variant]["source_signal_sha256"]).item())
                != exp2.sha256
            ):
                raise RuntimeError(
                    f"Validation C {variant}/Experiment 2 source hash linkage failed"
                )
        positive = np.asarray(
            variant_arrays["C0"]["positive_class_ids"], dtype=np.int64
        )
        local = _class_offset(positive, class_id)
        c0_attn = variant_arrays["C0"]["attention_c2p_conditional_l10_l12"][2, local]
        c4_attn = variant_arrays["C4"]["attention_c2p_conditional_l10_l12"][2, local]
        c0_cam = variant_arrays["C0"]["final_cam"][local]
        c4_cam = variant_arrays["C4"]["final_cam"][local]
        maps = [
            ("C0 L12 A_c2p", c0_attn, "positive"),
            ("C4 L12 A_c2p", c4_attn, "positive"),
            ("C0 final CAM", c0_cam, "positive"),
            ("C4 final CAM", c4_cam, "positive"),
            ("C4 - C0 final", c4_cam - c0_cam, "difference"),
        ]
    for name, values, _ in maps:
        values = np.asarray(values)
        if values.size != EXPECTED_PATCHES or not np.isfinite(values).all():
            raise RuntimeError(f"invalid {validation} map {name}: {values.shape}")
    return maps, controls


def _class_offset(positive: np.ndarray, class_id: int) -> int:
    indices = np.flatnonzero(positive == class_id)
    if len(indices) != 1:
        raise RuntimeError(f"focal class {class_id} is not uniquely positive")
    return int(indices[0])


def render_compact_panel(
    *,
    rgb: np.ndarray,
    mask: np.ndarray,
    focal_class_id: int,
    maps: Sequence[tuple[str, np.ndarray, str]],
    title: str,
    destination: Path,
    dpi: int,
) -> None:
    """Render one seven-column scientific panel with deterministic scaling."""

    if len(maps) != 5:
        raise ValueError("a compact panel requires exactly five signal maps")
    figure, axes = plt.subplots(1, 7, figsize=(16.8, 2.8))
    axes[0].imshow(rgb, interpolation="nearest")
    axes[0].set_title("Input", fontsize=9)
    axes[1].imshow(
        _target_gt(mask, focal_class_id),
        cmap=GT_COLORMAP,
        norm=GT_NORM,
        interpolation="nearest",
    )
    axes[1].set_title("GT target", fontsize=9)
    for axis, (name, raw, kind) in zip(axes[2:], maps):
        values = np.asarray(raw, dtype=np.float64).reshape(28, 28)
        if kind in {"signed", "difference"}:
            limit = max(float(np.abs(values).max()), np.finfo(np.float32).eps)
            cmap, vmin, vmax = "coolwarm", -limit, limit
        elif kind == "positive":
            cmap, vmin = "viridis", 0.0
            vmax = max(float(values.max()), np.finfo(np.float32).eps)
        else:
            raise ValueError(f"unknown map scaling kind: {kind}")
        axis.imshow(
            values,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(name, fontsize=8)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(
        destination,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "TGCA Experiment 3 deterministic renderer"},
    )
    plt.close(figure)


def _exp2_index(analyses: ValidatedAnalyses) -> Mapping[str, ManifestArtifact]:
    metadata = analyses.metadata["B"]
    roots = metadata.get("source_signal_roots")
    if not isinstance(roots, Mapping) or not isinstance(
        roots.get("mctformer_plus"), str
    ):
        raise RuntimeError("Validation B metadata lacks MCTformer+ Experiment 2 root")
    root = Path(str(roots["mctformer_plus"])).expanduser().resolve()
    return _manifest_index(root, path_field="signal_path")


def _render_validation(
    *,
    validation: str,
    selections: Sequence[Selection],
    output: Path,
    final_output: Path,
    analyses: ValidatedAnalyses,
    exp2_index: Mapping[str, ManifestArtifact],
    voc_root: Path,
    dpi: int,
    command: str,
    common_controls: Mapping[Path, str],
) -> Mapping[Path, str]:
    rows = []
    controls: dict[Path, str] = {}
    source_table = {
        "A": analyses.roots["A"] / "canonical" / "positive_map_overlap.parquet",
        "B": analyses.roots["B"] / "canonical_image_cam_t045.parquet",
        "C": analyses.roots["C"] / "canonical_image_cam_t045.parquet",
    }[validation]
    selection_metric = {
        "A": "mean L10-L12 positive-pair top10 Jaccard: both_removed - raw",
        "B": "single-image raw-CAM mIoU at 0.45: B1 - B0",
        "C": "single-image raw-CAM mIoU at 0.45: C4 - C0",
    }[validation]
    for selection in selections:
        rgb, mask, geometry, jpeg, mask_path = _load_transformed_input(
            voc_root, selection.image_id
        )
        maps, signal_controls = _maps_for_selection(
            selection, analyses, exp2_index, mask
        )
        for path in [jpeg, mask_path, *signal_controls]:
            controls[path] = sha256_file(path)
        filename = _safe_name(
            f"{validation}_{selection.rank_role}_{selection.image_id}_c{selection.focal_class_id}.png"
        )
        temporary_panel = output / filename
        final_panel = final_output / filename
        render_compact_panel(
            rgb=rgb,
            mask=mask,
            focal_class_id=selection.focal_class_id,
            maps=maps,
            title=(
                f"Validation {validation} · {selection.rank_role} · "
                f"{selection.image_id} · class {selection.focal_class_id} "
                f"({VOC_CLASS_NAMES[selection.focal_class_id]})"
                + (
                    ""
                    if selection.companion_class_id is None
                    else (
                        f" with class {selection.companion_class_id} "
                        f"({VOC_CLASS_NAMES[selection.companion_class_id]})"
                    )
                )
                + f" · Δ={selection.delta:+.5f}"
            ),
            destination=temporary_panel,
            dpi=dpi,
        )
        rows.append(
            {
                "validation": validation,
                "rank_role": selection.rank_role,
                "selection_rule": SELECTION_RULE,
                "selection_metric": selection_metric,
                "selection_source_table": str(source_table),
                "tie_break": "lexical image_id ascending",
                "selection_was_manual": False,
                "image_id": selection.image_id,
                "focal_class_id": selection.focal_class_id,
                "focal_class_name": VOC_CLASS_NAMES[selection.focal_class_id],
                "companion_class_id": (
                    ""
                    if selection.companion_class_id is None
                    else selection.companion_class_id
                ),
                "companion_class_name": (
                    ""
                    if selection.companion_class_id is None
                    else VOC_CLASS_NAMES[selection.companion_class_id]
                ),
                "primary_per_image_delta": selection.delta,
                "panel_path": str(final_panel),
                "panel_sha256": sha256_file(temporary_panel),
                "matched_geometry_json": json.dumps(geometry, sort_keys=True),
            }
        )
    selection_path = output / "selection.csv"
    pd.DataFrame.from_records(rows).to_csv(selection_path, index=False)
    metadata = {
        "status": "complete",
        "run_kind": "full",
        "validation": validation,
        "selected_examples": len(rows),
        "selection_rule": SELECTION_RULE,
        "selection_was_manual": False,
        "dpi": dpi,
        "selection_sha256": sha256_file(selection_path),
        "input_hashes": {
            str(path): digest
            for path, digest in {**common_controls, **controls}.items()
        },
        "command": command,
        "completed_at": timestamp(),
    }
    json_dump(output / "render_metadata.json", metadata)
    return controls


def render_examples(
    *,
    run_root: Path,
    validation_a_root: Path,
    validation_b_root: Path,
    validation_c_root: Path,
    source_metadata: Path,
    dpi: int = 120,
    command: str = "",
) -> Mapping[str, object]:
    require_tgca_repro()
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    analyses = validate_analysis_roots(
        run_root, validation_a_root, validation_b_root, validation_c_root
    )
    source_path = source_metadata.expanduser().resolve()
    source = read_json(source_path)
    if source.get("status") != "complete" or source.get("integrity_passed") is not True:
        raise RuntimeError("renderer requires a completed Experiment 3 source audit")
    dataset = source.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("voc_root"), str):
        raise RuntimeError("source metadata lacks VOC root")
    voc_root = Path(str(dataset["voc_root"])).expanduser().resolve()
    outputs = {
        "A": analyses.run_root / "presence_axis" / "examples",
        "B": analyses.run_root / "cam_layer_intervention" / "examples",
        "C": analyses.run_root / "c2c_intervention" / "examples",
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    selection_sources = (
        analyses.roots["A"] / "canonical" / "positive_map_overlap.parquet",
        analyses.roots["B"] / "canonical_image_cam_t045.parquet",
        analyses.roots["C"] / "canonical_image_cam_t045.parquet",
        _analysis_run_root(analyses, "A") / "manifest.jsonl",
        _analysis_run_root(analyses, "B") / "manifest.jsonl",
        _analysis_run_root(analyses, "C") / "signals" / "C0" / "manifest.jsonl",
        _analysis_run_root(analyses, "C") / "signals" / "C4" / "manifest.jsonl",
    )
    source_controls = {source_path: sha256_file(source_path)}
    source_controls.update(analyses.control_hashes)
    source_controls.update({path: sha256_file(path) for path in selection_sources})
    selections = _build_selections(analyses)
    exp2_index = _exp2_index(analyses)
    temporary = {
        name: Path(tempfile.mkdtemp(prefix=".examples.tmp-", dir=str(path.parent)))
        for name, path in outputs.items()
    }
    published: list[Path] = []
    try:
        rendered_controls: dict[Path, str] = {}
        for name in ("A", "B", "C"):
            rendered_controls.update(
                _render_validation(
                    validation=name,
                    selections=selections[name],
                    output=temporary[name],
                    final_output=outputs[name],
                    analyses=analyses,
                    exp2_index=exp2_index,
                    voc_root=voc_root,
                    dpi=dpi,
                    command=command,
                    common_controls=source_controls,
                )
            )
        for path, digest in {**source_controls, **rendered_controls}.items():
            if sha256_file(path) != digest:
                raise RuntimeError(f"renderer input changed: {path}")
        for name in ("A", "B", "C"):
            if outputs[name].exists():
                raise FileExistsError(outputs[name])
            temporary[name].replace(outputs[name])
            published.append(outputs[name])
    except Exception:
        for path in temporary.values():
            shutil.rmtree(path, ignore_errors=True)
        # Only directories published by this invocation are rolled back.  All
        # three destinations were proven absent before rendering, so this
        # cannot remove a pre-existing user result.
        for path in published:
            shutil.rmtree(path, ignore_errors=True)
        raise
    return {
        "status": "complete",
        "selection_rule": SELECTION_RULE,
        "outputs": {name: str(path) for name, path in outputs.items()},
        "selected_examples_per_validation": 3,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--validation-a-root", type=Path, required=True)
    parser.add_argument("--validation-b-root", type=Path, required=True)
    parser.add_argument("--validation-c-root", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def main() -> None:
    args = parse_args()
    command = " ".join([sys.executable, *sys.argv])
    result = render_examples(
        run_root=args.run_root,
        validation_a_root=args.validation_a_root,
        validation_b_root=args.validation_b_root,
        validation_c_root=args.validation_c_root,
        source_metadata=args.source_metadata,
        dpi=args.dpi,
        command=command,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
