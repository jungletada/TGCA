#!/usr/bin/env python3
"""Render rule-selected Experiment 2 examples from immutable signal artifacts.

The renderer is deliberately model-free.  It reads the existing per-image NPZ
files named by each signal-root manifest and applies the already-tested joint
VOC transform to RGB and semantic GT.  Missing inputs are errors: no map, class,
or panel is synthesized as a substitute.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from analysis.lazy_assignment.experiment2.common import (  # noqa: E402
    VOC_CLASS_NAMES,
    read_json,
    sha256_file,
)
from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    patch_label_counts,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (  # noqa: E402
    build_joint_transform,
    resolve_semantic_mask_path,
)
from analysis.lazy_assignment.visualize_patch_score import (  # noqa: E402
    input_tensor_to_rgb,
)


PANEL_COLUMNS = (
    "Input",
    "GT",
    "L12 Feature",
    "L12 conditional A_c2p",
    "Patch CAM",
    "C2P CAM",
    "Final CAM",
)
REQUIRED_NPZ_KEYS = (
    "image_id",
    "positive_class_ids",
    "grid_h",
    "grid_w",
    "patch_label_counts",
    "feature_post_scores",
    "attn_c2p_conditional",
    "patch_cam",
    "c2p_cam",
    "final_cam",
)
MANIFEST_FIELDS = (
    "case_id",
    "category",
    "selection_kind",
    "render_status",
    "selection_row_index",
    "model",
    "image_id",
    "class_ids_json",
    "class_names_json",
    "num_class_rows",
    "panel_path",
    "panel_sha256",
    "signal_root",
    "signal_npz_path",
    "signal_npz_sha256",
    "jpeg_path",
    "semantic_mask_path",
    "matched_geometry_json",
    "experiment1_case_id",
    "experiment1_raw_figure",
    "experiment1_raw_sha256",
    "experiment1_minmax_figure",
    "experiment1_minmax_sha256",
)
GT_COLORMAP = ListedColormap(("#d9d9d9", "#22aa66", "#dd8844", "#222222"))
GT_NORM = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), GT_COLORMAP.N)


@dataclass(frozen=True)
class SignalArtifact:
    model: str
    image_id: str
    root: Path
    path: Path
    expected_sha256: str


@dataclass(frozen=True)
class RenderCase:
    selection_row_index: int
    case_id: str
    category: str
    model: str
    image_id: str
    class_ids: tuple[int, ...]
    artifact: SignalArtifact
    jpeg_path: Path
    mask_path: Path
    selection_metric: str
    selection_value: str


@dataclass(frozen=True)
class FixedExperiment1Case:
    selection_row_index: int
    case_id: str
    category: str
    model: str
    image_id: str
    experiment1_case_id: str
    raw_path: Path
    minmax_path: Path


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example-selection", type=Path, required=True)
    parser.add_argument("--canonical-metadata", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args(argv)
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    return args


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-._")
    if not cleaned:
        raise ValueError("panel filename would be empty")
    return cleaned


def _required_text(row: pd.Series, name: str) -> str:
    if name not in row:
        raise ValueError(f"example selection lacks required column {name!r}")
    value = str(row[name]).strip()
    if not value:
        raise ValueError(f"selection row {row.name} has empty {name!r}")
    return value


def _optional_int(row: pd.Series, name: str) -> Optional[int]:
    if name not in row:
        return None
    value = str(row[name]).strip()
    if not value:
        return None
    try:
        numeric = float(value)
    except ValueError as error:
        raise ValueError(
            f"selection row {row.name} has invalid integer {name}={value!r}"
        ) from error
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(
            f"selection row {row.name} has invalid integer {name}={value!r}"
        )
    result = int(numeric)
    if not 0 <= result < len(VOC_CLASS_NAMES):
        raise ValueError(
            f"selection row {row.name} has class outside [0,19]: {name}={result}"
        )
    return result


def _selected_classes(row: pd.Series) -> tuple[int, ...]:
    class_a = _optional_int(row, "class_a")
    class_b = _optional_int(row, "class_b")
    if (class_a is None) != (class_b is None):
        raise ValueError(
            f"selection row {row.name} must provide both class_a and class_b"
        )
    companion = _optional_int(row, "companion_class_id")
    if class_a is not None and class_b is not None:
        if class_a == class_b:
            raise ValueError(f"selection row {row.name} repeats one pair class")
        if companion is not None:
            raise ValueError(
                f"selection row {row.name} cannot combine pair and companion classes"
            )
        selected = (class_a, class_b)
    else:
        class_id = _optional_int(row, "class_id")
        if class_id is None:
            raise ValueError(
                f"selection row {row.name} has neither a class pair nor class_id"
            )
        if companion == class_id:
            raise ValueError(f"selection row {row.name} repeats its focal class")
        selected = (class_id,) if companion is None else (class_id, companion)

    if "positive_class_ids_json" not in row or "num_positive_classes" not in row:
        raise ValueError(
            f"selection row {row.name} lacks positive-class panel provenance"
        )
    try:
        positive_raw = json.loads(str(row["positive_class_ids_json"]))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"selection row {row.name} has invalid positive_class_ids_json"
        ) from error
    if not isinstance(positive_raw, list):
        raise ValueError(
            f"selection row {row.name} positive_class_ids_json is not a list"
        )
    positive = tuple(int(value) for value in positive_raw)
    if (
        not positive
        or len(set(positive)) != len(positive)
        or any(not 0 <= value < len(VOC_CLASS_NAMES) for value in positive)
    ):
        raise ValueError(f"selection row {row.name} has invalid positive classes")
    try:
        recorded_count = int(str(row["num_positive_classes"]))
    except ValueError as error:
        raise ValueError(
            f"selection row {row.name} has invalid num_positive_classes"
        ) from error
    if recorded_count != len(positive) or any(
        value not in positive for value in selected
    ):
        raise ValueError(
            f"selection row {row.name} positive-class provenance is inconsistent"
        )
    if len(positive) >= 2 and len(selected) < 2:
        raise ValueError(
            f"multi-label selection row {row.name} must display at least two positives"
        )
    return selected


def _is_fixed_experiment1(row: pd.Series) -> bool:
    category = str(row.get("category", ""))
    case_id = str(row.get("experiment1_case_id", "")).strip()
    return category.startswith("experiment1_fixed::") or bool(case_id)


def _inside_root(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes immutable signal root {root}: {candidate}")
    return candidate


def _load_signal_index(
    canonical_metadata: dict, models: Sequence[str]
) -> dict[tuple[str, str], SignalArtifact]:
    source_roots = canonical_metadata.get("source_roots")
    if not isinstance(source_roots, dict):
        raise ValueError("canonical metadata lacks source_roots")
    result: dict[tuple[str, str], SignalArtifact] = {}
    for model in sorted(set(models)):
        root_value = source_roots.get(model)
        if not isinstance(root_value, str) or not root_value:
            raise ValueError(f"canonical metadata lacks signal root for {model}")
        root = Path(root_value).expanduser().resolve()
        completion = read_json(root / "completion.json")
        if completion.get("status") != "complete":
            raise RuntimeError(f"signal root is not complete: {root}")
        manifest_path = root / "manifest.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                image_id = str(record.get("image_id", ""))
                relative = record.get("signal_path")
                expected = str(record.get("artifact_sha256", ""))
                if not image_id or not isinstance(relative, str) or not relative:
                    raise ValueError(
                        f"invalid signal manifest row {manifest_path}:{line_number}"
                    )
                key = (model, image_id)
                if key in result:
                    raise ValueError(f"duplicate signal manifest image: {key}")
                path = _inside_root(root, root / relative)
                result[key] = SignalArtifact(model, image_id, root, path, expected)
    return result


def _resolve_existing_link(row: pd.Series, column: str) -> Path:
    value = _required_text(row, column)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        source_table = Path(_required_text(row, "source_table")).expanduser()
        if not source_table.is_absolute():
            source_table = source_table.resolve()
        candidate = source_table.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(
            f"fixed Experiment 1 figure does not exist ({column}): {candidate}"
        )
    return candidate


def _build_case_plans(
    selection: pd.DataFrame,
    canonical_metadata: dict,
    voc_root: Path,
) -> tuple[list[RenderCase], list[FixedExperiment1Case]]:
    for column in ("case_id", "category", "model", "image_id"):
        if column not in selection:
            raise ValueError(f"example selection lacks required column {column!r}")
    if selection.empty:
        raise ValueError("example selection contains no rows")
    if selection["case_id"].astype(str).duplicated().any():
        raise ValueError("example selection case_id values must be unique")

    fixed_rows = [row for _, row in selection.iterrows() if _is_fixed_experiment1(row)]
    new_rows = [
        row for _, row in selection.iterrows() if not _is_fixed_experiment1(row)
    ]
    models = [_required_text(row, "model") for row in new_rows]
    signal_index = _load_signal_index(canonical_metadata, models) if models else {}

    render_cases: list[RenderCase] = []
    fixed_cases: list[FixedExperiment1Case] = []
    for row in new_rows:
        model = _required_text(row, "model")
        image_id = _required_text(row, "image_id")
        artifact = signal_index.get((model, image_id))
        if artifact is None:
            raise FileNotFoundError(
                f"no manifested signal artifact for model/image {(model, image_id)}"
            )
        jpeg_path = voc_root / "JPEGImages" / f"{image_id}.jpg"
        if not jpeg_path.is_file():
            raise FileNotFoundError(jpeg_path)
        mask_path = resolve_semantic_mask_path(voc_root, image_id)
        render_cases.append(
            RenderCase(
                selection_row_index=int(row.name),
                case_id=_required_text(row, "case_id"),
                category=_required_text(row, "category"),
                model=model,
                image_id=image_id,
                class_ids=_selected_classes(row),
                artifact=artifact,
                jpeg_path=jpeg_path.resolve(),
                mask_path=mask_path.resolve(),
                selection_metric=str(row.get("selection_metric", "")).strip(),
                selection_value=str(row.get("selection_value", "")).strip(),
            )
        )
    for row in fixed_rows:
        fixed_cases.append(
            FixedExperiment1Case(
                selection_row_index=int(row.name),
                case_id=_required_text(row, "case_id"),
                category=_required_text(row, "category"),
                model=_required_text(row, "model"),
                image_id=_required_text(row, "image_id"),
                experiment1_case_id=_required_text(row, "experiment1_case_id"),
                raw_path=_resolve_existing_link(row, "experiment1_raw_figure"),
                minmax_path=_resolve_existing_link(row, "experiment1_minmax_figure"),
            )
        )
    return render_cases, fixed_cases


def _validate_and_load_maps(
    case: RenderCase,
    expected_input_size: int,
    expected_patch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[int, dict[str, np.ndarray]]]:
    with Image.open(case.jpeg_path) as source_image:
        image = source_image.convert("RGB")
    with Image.open(case.mask_path) as source_mask:
        mask = source_mask.copy()
    transform = build_joint_transform(expected_input_size)
    image_tensor, mask_tensor, geometry = transform(image, mask)
    rgb = input_tensor_to_rgb(image_tensor)
    mask_array = mask_tensor.numpy()
    observed_counts = patch_label_counts(mask_array, expected_patch_size)

    with np.load(case.artifact.path, allow_pickle=False) as artifact:
        missing = sorted(set(REQUIRED_NPZ_KEYS).difference(artifact.files))
        if missing:
            raise KeyError(
                f"signal NPZ {case.artifact.path} lacks required keys {missing}"
            )
        artifact_image_id = str(np.asarray(artifact["image_id"]).item())
        if artifact_image_id != case.image_id:
            raise ValueError(
                f"signal image ID mismatch: {artifact_image_id} != {case.image_id}"
            )
        positive = np.asarray(artifact["positive_class_ids"], dtype=np.int64)
        if positive.ndim != 1 or len(set(positive.tolist())) != positive.size:
            raise ValueError("positive_class_ids must be a unique rank-one array")
        grid_h = int(np.asarray(artifact["grid_h"]).item())
        grid_w = int(np.asarray(artifact["grid_w"]).item())
        expected_grid = expected_input_size // expected_patch_size
        if (grid_h, grid_w) != (expected_grid, expected_grid):
            raise ValueError(
                f"signal grid {(grid_h, grid_w)} != matched grid {(expected_grid, expected_grid)}"
            )
        stored_counts = np.asarray(artifact["patch_label_counts"])
        if not np.array_equal(stored_counts, observed_counts):
            difference = int(
                np.abs(stored_counts.astype(np.int32) - observed_counts).max()
            )
            raise ValueError(
                f"matched GT patch counts differ from NPZ for {case.image_id}; max-|Δ|={difference}"
            )

        feature = np.asarray(artifact["feature_post_scores"], dtype=np.float32)
        attention = np.asarray(artifact["attn_c2p_conditional"], dtype=np.float32)
        patch_cam = np.asarray(artifact["patch_cam"], dtype=np.float32)
        c2p_cam = np.asarray(artifact["c2p_cam"], dtype=np.float32)
        final_cam = np.asarray(artifact["final_cam"], dtype=np.float32)
        num_classes = positive.size
        num_patches = grid_h * grid_w
        if feature.shape != (12, num_classes, num_patches):
            raise ValueError(f"invalid feature_post_scores shape: {feature.shape}")
        if attention.shape != (12, num_classes, num_patches):
            raise ValueError(f"invalid attn_c2p_conditional shape: {attention.shape}")
        for name, values in (
            ("patch_cam", patch_cam),
            ("c2p_cam", c2p_cam),
            ("final_cam", final_cam),
        ):
            if values.shape != (num_classes, num_patches):
                raise ValueError(f"invalid {name} shape: {values.shape}")
        arrays = (feature, attention, patch_cam, c2p_cam, final_cam)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError(f"signal NPZ has NaN/Inf: {case.artifact.path}")
        if (attention < -1e-8).any() or not np.allclose(
            attention.sum(axis=-1), 1.0, rtol=0.0, atol=2e-5
        ):
            raise ValueError(
                f"conditional attention is not nonnegative/unit-mass: {case.artifact.path}"
            )

        offset = {int(class_id): index for index, class_id in enumerate(positive)}
        result: dict[int, dict[str, np.ndarray]] = {}
        for class_id in case.class_ids:
            if class_id not in offset:
                raise ValueError(
                    f"selected class {class_id} is absent from positive_class_ids "
                    f"for {case.model}/{case.image_id}: {positive.tolist()}"
                )
            local = offset[class_id]
            result[class_id] = {
                "feature": feature[11, local].reshape(grid_h, grid_w).copy(),
                "attention": attention[11, local].reshape(grid_h, grid_w).copy(),
                "patch_cam": patch_cam[local].reshape(grid_h, grid_w).copy(),
                "c2p_cam": c2p_cam[local].reshape(grid_h, grid_w).copy(),
                "final_cam": final_cam[local].reshape(grid_h, grid_w).copy(),
            }
    return rgb, mask_array, geometry, result


def _target_gt(mask: np.ndarray, class_id: int) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=np.uint8)
    result[(mask >= 1) & (mask <= 20)] = 2
    result[mask == class_id + 1] = 1
    result[mask == 255] = 3
    return result


def _patch_grid(axis: plt.Axes, height: int, width: int, step: int = 1) -> None:
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_xticks(np.arange(-0.5, width, step), minor=True)
    axis.set_yticks(np.arange(-0.5, height, step), minor=True)
    axis.grid(which="minor", color="white", alpha=0.18, linewidth=0.18)
    axis.tick_params(
        which="both", bottom=False, left=False, labelbottom=False, labelleft=False
    )


def _map_scale(
    maps: dict[int, dict[str, np.ndarray]], class_ids: Sequence[int], key: str
) -> tuple[float, float]:
    values = np.concatenate([maps[class_id][key].reshape(-1) for class_id in class_ids])
    if key == "feature":
        return -1.0, 1.0
    minimum = min(0.0, float(values.min()))
    maximum = float(values.max())
    if maximum <= minimum:
        maximum = float(np.nextafter(np.float32(minimum), np.float32(np.inf)))
    return minimum, maximum


def _render_case_panel(
    case: RenderCase,
    rgb: np.ndarray,
    mask: np.ndarray,
    maps: dict[int, dict[str, np.ndarray]],
    destination: Path,
    dpi: int,
    patch_size: int,
) -> None:
    rows = len(case.class_ids)
    figure = plt.figure(figsize=(22.0, 3.5 * rows + 0.55))
    grid = figure.add_gridspec(
        rows + 1,
        len(PANEL_COLUMNS),
        height_ratios=([1.0] * rows) + [0.065],
        hspace=0.24,
        wspace=0.18,
    )
    axes = np.asarray(
        [
            [
                figure.add_subplot(grid[row, column])
                for column in range(len(PANEL_COLUMNS))
            ]
            for row in range(rows)
        ],
        dtype=object,
    )
    colorbar_axes = [
        figure.add_subplot(grid[rows, column]) for column in range(len(PANEL_COLUMNS))
    ]
    colorbar_axes[0].axis("off")
    colorbar_axes[1].axis("off")
    signal_columns = {
        2: ("feature", "coolwarm", "cosine"),
        3: ("attention", "viridis", "conditional mass"),
        4: ("patch_cam", "viridis", "native value"),
        5: ("c2p_cam", "viridis", "native value"),
        6: ("final_cam", "viridis", "native value"),
    }
    scales = {
        key: _map_scale(maps, case.class_ids, key)
        for key, _, _ in signal_columns.values()
    }
    handles: dict[int, object] = {}
    for row_index, class_id in enumerate(case.class_ids):
        class_name = VOC_CLASS_NAMES[class_id]
        axes[row_index, 0].imshow(rgb, interpolation="nearest")
        _patch_grid(axes[row_index, 0], rgb.shape[0], rgb.shape[1], step=patch_size)
        axes[row_index, 0].set_ylabel(
            f"class {class_id}: {class_name}", fontsize=10, fontweight="bold"
        )

        axes[row_index, 1].imshow(
            _target_gt(mask, class_id),
            cmap=GT_COLORMAP,
            norm=GT_NORM,
            interpolation="nearest",
        )
        _patch_grid(axes[row_index, 1], mask.shape[0], mask.shape[1], step=patch_size)
        axes[row_index, 1].text(
            0.02,
            0.02,
            "target / other-FG / BG / void",
            transform=axes[row_index, 1].transAxes,
            fontsize=7,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

        for column, (key, cmap, _) in signal_columns.items():
            values = maps[class_id][key]
            handle = axes[row_index, column].imshow(
                values,
                cmap=cmap,
                vmin=scales[key][0],
                vmax=scales[key][1],
                interpolation="nearest",
            )
            _patch_grid(axes[row_index, column], values.shape[0], values.shape[1])
            handles[column] = handle
        if row_index == 0:
            for column, title in enumerate(PANEL_COLUMNS):
                axes[row_index, column].set_title(title, fontsize=10)

    for column, (_, _, colorbar_label) in signal_columns.items():
        colorbar = figure.colorbar(
            handles[column],
            cax=colorbar_axes[column],
            orientation="horizontal",
        )
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(colorbar_label, fontsize=8)
    selection_note = case.selection_metric
    if case.selection_value:
        selection_note += f"={case.selection_value}"
    figure.suptitle(
        f"{case.case_id} · {case.category} · {case.model} · {case.image_id}"
        + (f" · {selection_note}" if selection_note else ""),
        fontsize=12,
    )
    figure.subplots_adjust(
        left=0.035,
        right=0.985,
        bottom=0.06,
        top=0.86 if rows > 1 else 0.80,
    )
    figure.savefig(
        destination,
        dpi=dpi,
        bbox_inches="tight",
        metadata={"Software": "TGCA Experiment 2 model-free renderer"},
    )
    plt.close(figure)


def _fixed_manifest_row(case: FixedExperiment1Case) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "selection_kind": "fixed_experiment1",
        "render_status": "linked_existing_not_redrawn",
        "selection_row_index": case.selection_row_index,
        "model": case.model,
        "image_id": case.image_id,
        "class_ids_json": "[]",
        "class_names_json": "[]",
        "num_class_rows": 0,
        "panel_path": "",
        "panel_sha256": "",
        "signal_root": "",
        "signal_npz_path": "",
        "signal_npz_sha256": "",
        "jpeg_path": "",
        "semantic_mask_path": "",
        "matched_geometry_json": "",
        "experiment1_case_id": case.experiment1_case_id,
        "experiment1_raw_figure": str(case.raw_path),
        "experiment1_raw_sha256": sha256_file(case.raw_path),
        "experiment1_minmax_figure": str(case.minmax_path),
        "experiment1_minmax_sha256": sha256_file(case.minmax_path),
    }


def render_examples(
    example_selection: Path,
    canonical_metadata_path: Path,
    source_metadata_path: Path,
    output_dir: Path,
    *,
    dpi: int = 150,
    command: Optional[str] = None,
) -> dict[str, object]:
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    example_selection = example_selection.expanduser().resolve()
    canonical_metadata_path = canonical_metadata_path.expanduser().resolve()
    source_metadata_path = source_metadata_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite render directory: {output_dir}")
    canonical_metadata = read_json(canonical_metadata_path)
    source_metadata = read_json(source_metadata_path)
    if canonical_metadata.get("status") != "complete":
        raise RuntimeError("example rendering requires complete canonical metadata")
    if source_metadata.get("integrity_passed") is not True:
        raise RuntimeError(
            "example rendering requires a passing Experiment 2 input audit"
        )
    dataset = source_metadata.get("dataset", {})
    input_size = int(dataset.get("input_size", -1))
    patch_size = int(dataset.get("patch_size", -1))
    if (input_size, patch_size) != (448, 16):
        raise ValueError(
            f"Experiment 2 rendering is fixed to input/patch=448/16, got {input_size}/{patch_size}"
        )
    voc_root_value = dataset.get("voc_root")
    if not isinstance(voc_root_value, str) or not voc_root_value:
        raise ValueError("source metadata lacks dataset.voc_root")
    voc_root = Path(voc_root_value).expanduser().resolve()
    if not example_selection.is_file():
        raise FileNotFoundError(example_selection)
    selection = pd.read_csv(example_selection, dtype=str, keep_default_na=False)
    selection.index = np.arange(len(selection), dtype=np.int64)
    render_cases, fixed_cases = _build_case_plans(
        selection, canonical_metadata, voc_root
    )

    unique_artifacts = {case.artifact.path: case.artifact for case in render_cases}
    source_hashes_before: dict[Path, str] = {}
    for path, artifact in unique_artifacts.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256_file(path)
        if artifact.expected_sha256 and observed != artifact.expected_sha256:
            raise RuntimeError(
                f"signal artifact hash mismatch for {artifact.model}/{artifact.image_id}: "
                f"manifest={artifact.expected_sha256}, observed={observed}"
            )
        source_hashes_before[path] = observed

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        panel_dir = staging / "panels"
        panel_dir.mkdir()
        generated_rows: dict[int, dict[str, object]] = {}
        for case in render_cases:
            rgb, mask, geometry, maps = _validate_and_load_maps(
                case, input_size, patch_size
            )
            classes = "-".join(f"{value:02d}" for value in case.class_ids)
            filename = (
                _safe_filename(
                    f"{case.case_id}_{case.category}_{case.model}_{case.image_id}_classes-{classes}"
                )
                + ".png"
            )
            staged_panel = panel_dir / filename
            _render_case_panel(
                case, rgb, mask, maps, staged_panel, dpi=dpi, patch_size=patch_size
            )
            target_panel = output_dir / "panels" / filename
            generated_rows[case.selection_row_index] = {
                "case_id": case.case_id,
                "category": case.category,
                "selection_kind": "experiment2_gt_rule",
                "render_status": "rendered_from_existing_npz",
                "selection_row_index": case.selection_row_index,
                "model": case.model,
                "image_id": case.image_id,
                "class_ids_json": json.dumps(list(case.class_ids)),
                "class_names_json": json.dumps(
                    [VOC_CLASS_NAMES[value] for value in case.class_ids]
                ),
                "num_class_rows": len(case.class_ids),
                "panel_path": str(target_panel),
                "panel_sha256": sha256_file(staged_panel),
                "signal_root": str(case.artifact.root),
                "signal_npz_path": str(case.artifact.path),
                "signal_npz_sha256": source_hashes_before[case.artifact.path],
                "jpeg_path": str(case.jpeg_path),
                "semantic_mask_path": str(case.mask_path),
                "matched_geometry_json": json.dumps(geometry, sort_keys=True),
                "experiment1_case_id": "",
                "experiment1_raw_figure": "",
                "experiment1_raw_sha256": "",
                "experiment1_minmax_figure": "",
                "experiment1_minmax_sha256": "",
            }
        fixed_rows = {
            case.selection_row_index: _fixed_manifest_row(case) for case in fixed_cases
        }
        manifest_rows = [
            (generated_rows | fixed_rows)[index] for index in range(len(selection))
        ]
        manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_FIELDS)
        manifest_path = staging / "render_manifest.csv"
        manifest.to_csv(manifest_path, index=False)

        unchanged = all(
            sha256_file(path) == before for path, before in source_hashes_before.items()
        )
        if not unchanged:
            raise RuntimeError(
                "one or more immutable signal NPZ files changed during rendering"
            )
        canonical_count = canonical_metadata.get("num_manifest_images_per_model")
        dataset_count = dataset.get("num_images")
        canonical_is_full = (
            canonical_count is not None
            and dataset_count is not None
            and int(canonical_count) == int(dataset_count)
        )
        metadata = {
            "status": "complete",
            "command": command or "programmatic render_examples invocation",
            "example_selection": str(example_selection),
            "example_selection_sha256": sha256_file(example_selection),
            "canonical_metadata": str(canonical_metadata_path),
            "canonical_metadata_sha256": sha256_file(canonical_metadata_path),
            "source_metadata": str(source_metadata_path),
            "source_metadata_sha256": sha256_file(source_metadata_path),
            "output_dir": str(output_dir),
            "panel_columns": list(PANEL_COLUMNS),
            "input_size": input_size,
            "patch_size": patch_size,
            "manifest_rows": len(manifest),
            "rendered_panel_files": len(render_cases),
            "fixed_experiment1_cases_linked": len(fixed_cases),
            "linked_existing_figure_references": 2 * len(fixed_cases),
            "output_file_count": len(render_cases) + 2,
            "unique_signal_npz_files_read": len(unique_artifacts),
            "source_npz_manifest_hashes_verified": True,
            "source_npz_unchanged": unchanged,
            "missing_data_placeholders_generated": False,
            "model_execution": False,
            "model_loaded": False,
            "scientific_report_decision_performed": False,
            "canonical_num_images_per_model": canonical_count,
            "dataset_num_images": dataset_count,
            "canonical_is_full_set": canonical_is_full,
            "smoke_rendering_allowed_without_scientific_claims": not canonical_is_full,
        }
        metadata_path = staging / "render_metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        actual_output_files = sum(path.is_file() for path in staging.rglob("*"))
        if actual_output_files != metadata["output_file_count"]:
            raise RuntimeError(
                f"render output file-count mismatch: {actual_output_files} != "
                f"{metadata['output_file_count']}"
            )
        staging.replace(output_dir)
    except Exception:
        shutil.rmtree(staging)
        raise
    return metadata


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    command = shlex.join([sys.executable, *sys.argv])
    result = render_examples(
        args.example_selection,
        args.canonical_metadata,
        args.source_metadata,
        args.output_dir,
        dpi=args.dpi,
        command=command,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
