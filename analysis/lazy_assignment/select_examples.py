#!/usr/bin/env python3
"""Select and render Experiment 1 examples using prespecified deterministic rules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize

from analysis.lazy_assignment.experiment1_analysis_common import (
    MODEL_ORDER,
    VOC_CLASS_NAMES,
)


MODEL_LABELS = {"mctformer": "MCTformer", "mctformer_plus": "MCTformer+"}
DISPLAY_LAYERS = (1, 4, 8, 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--voc-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--input-size", type=int, default=448)
    return parser.parse_args()


def stable_top(
    frame: pd.DataFrame, metric: str, n: int, ascending: bool
) -> pd.DataFrame:
    tie_columns = [
        column
        for column in ("image_id", "class_id", "class_id_a", "class_id_b")
        if column in frame.columns
    ]
    return frame.sort_values(
        [metric, *tie_columns], ascending=[ascending, *([True] * len(tie_columns))]
    ).head(n)


def select_cases(canonical: Path, top_n: int) -> pd.DataFrame:
    maps = pd.read_parquet(canonical / "per_image_class_layer.parquet")
    diversity = pd.read_parquet(canonical / "multiclass_pair_layer.parquet")
    paired = pd.read_parquet(canonical / "per_pair_layer.parquet")
    rows: list[dict[str, object]] = []

    q95 = maps[maps["layer"].isin([1, 12])].pivot(
        index=["model", "image_id", "class_id", "class_name"],
        columns="layer",
        values="score_q95",
    )
    q95["q95_l12_minus_l1"] = q95[12] - q95[1]
    q95 = q95.reset_index()
    for model in MODEL_ORDER:
        selected = stable_top(
            q95[q95["model"] == model], "q95_l12_minus_l1", top_n, False
        )
        for rank, item in enumerate(selected.itertuples(index=False), start=1):
            rows.append(
                {
                    "category": "A_max_q95_change_l12_vs_l1",
                    "selection_scope_model": model,
                    "selection_rank": rank,
                    "image_id": item.image_id,
                    "class_id": int(item.class_id),
                    "class_name": item.class_name,
                    "comparison_class_id": np.nan,
                    "comparison_class_name": "",
                    "selection_layer": 12,
                    "selection_metric": "q95_l12_minus_l1",
                    "selection_value": item.q95_l12_minus_l1,
                    "selection_direction": "largest",
                    "selection_rule": "largest q95(layer12)-q95(layer1), separately within each model; deterministic key tie-break",
                }
            )

    layer12_diversity = diversity[diversity["layer"] == 12]
    for category, direction, ascending in (
        ("B_highest_class_map_overlap_l12", "largest", False),
        ("C_lowest_class_map_overlap_l12", "smallest", True),
    ):
        for model in MODEL_ORDER:
            selected = stable_top(
                layer12_diversity[layer12_diversity["model"] == model],
                "top10_class_map_jaccard",
                top_n,
                ascending,
            )
            for rank, item in enumerate(selected.itertuples(index=False), start=1):
                rows.append(
                    {
                        "category": category,
                        "selection_scope_model": model,
                        "selection_rank": rank,
                        "image_id": item.image_id,
                        "class_id": int(item.class_id_a),
                        "class_name": item.class_name_a,
                        "comparison_class_id": int(item.class_id_b),
                        "comparison_class_name": item.class_name_b,
                        "selection_layer": 12,
                        "selection_metric": "top10_class_map_jaccard",
                        "selection_value": item.top10_class_map_jaccard,
                        "selection_direction": direction,
                        "selection_rule": f"{direction} layer-12 within-image class-pair top-10% Jaccard, separately within each model; deterministic key tie-break",
                    }
                )

    layer12_paired = paired[paired["layer"] == 12]
    selected = stable_top(
        layer12_paired, "cross_model_map_spearman", top_n, ascending=True
    )
    for rank, item in enumerate(selected.itertuples(index=False), start=1):
        rows.append(
            {
                "category": "D_largest_cross_model_disagreement_l12",
                "selection_scope_model": "paired",
                "selection_rank": rank,
                "image_id": item.image_id,
                "class_id": int(item.class_id),
                "class_name": item.class_name,
                "comparison_class_id": np.nan,
                "comparison_class_name": "",
                "selection_layer": 12,
                "selection_metric": "cross_model_map_spearman",
                "selection_value": item.cross_model_map_spearman,
                "selection_direction": "smallest",
                "selection_rule": "smallest layer-12 MCTformer-vs-MCTformer+ map Spearman on common image-class pairs; deterministic key tie-break",
            }
        )
    selection = pd.DataFrame(rows)
    selection.insert(0, "case_id", [f"case_{index:03d}" for index in range(1, len(selection) + 1)])
    return selection


def load_sources(canonical: Path) -> dict[tuple[str, str], dict[str, Any]]:
    source_index = pd.read_parquet(canonical / "source_index.parquet")
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in source_index.itertuples(index=False):
        result[(str(row.model), str(row.image_id))] = {
            "path": Path(row.score_path),
            "class_ids": json.loads(row.positive_class_ids_json),
            "grid_h": int(row.grid_h),
            "grid_w": int(row.grid_w),
        }
    return result


def load_case_maps(
    source_lookup: dict[tuple[str, str], dict[str, Any]],
    image_id: str,
    class_ids: list[int],
) -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    for model in MODEL_ORDER:
        source = source_lookup[(model, image_id)]
        with np.load(source["path"], allow_pickle=False) as artifact:
            positive_ids = np.asarray(artifact["positive_class_ids"], dtype=np.int64)
            scores = np.asarray(artifact["scores_raw"], dtype=np.float32)
        index = {int(class_id): offset for offset, class_id in enumerate(positive_ids)}
        for class_id in class_ids:
            if class_id not in index:
                raise RuntimeError(f"class {class_id} absent in {model}/{image_id}")
            result[(model, class_id)] = scores[:, index[class_id], :].reshape(
                scores.shape[0], source["grid_h"], source["grid_w"]
            )
    return result


def transformed_image(voc_root: Path, image_id: str, input_size: int) -> Image.Image:
    path = voc_root / "JPEGImages" / f"{image_id}.jpg"
    transform = Compose(
        (
            Resize(
                int(256 / 224 * input_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            CenterCrop(input_size),
        )
    )
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def render_case(
    row: pd.Series,
    maps: dict[tuple[str, int], np.ndarray],
    original: Image.Image,
    output: Path,
    mode: str,
) -> str:
    primary_class = int(row["class_id"])
    class_ids = [primary_class]
    if pd.notna(row["comparison_class_id"]):
        class_ids.append(int(row["comparison_class_id"]))
    display_rows = [(model, class_id) for model in MODEL_ORDER for class_id in class_ids]
    figure = plt.figure(figsize=(15, max(5.2, 2.45 * len(display_rows))))
    grid = figure.add_gridspec(
        len(display_rows), 5, width_ratios=(1.35, 1.0, 1.0, 1.0, 1.0)
    )
    original_axis = figure.add_subplot(grid[:, 0])
    original_axis.imshow(original)
    original_axis.set_title(f"VOC input transform\n{row['image_id']}")
    original_axis.axis("off")
    map_axes: list[plt.Axes] = []
    image_handle = None
    for row_index, (model, class_id) in enumerate(display_rows):
        class_name = VOC_CLASS_NAMES[class_id]
        values_by_layer = maps[(model, class_id)]
        for column_index, layer in enumerate(DISPLAY_LAYERS, start=1):
            axis = figure.add_subplot(grid[row_index, column_index])
            values = values_by_layer[layer - 1]
            if mode == "raw":
                display = values
                image_handle = axis.imshow(
                    display, cmap="coolwarm", vmin=-1.0, vmax=1.0, interpolation="bicubic"
                )
            elif mode == "minmax":
                minimum = float(values.min())
                maximum = float(values.max())
                display = (
                    (values - minimum) / (maximum - minimum)
                    if maximum > minimum
                    else np.zeros_like(values)
                )
                image_handle = axis.imshow(
                    display, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="bicubic"
                )
            else:
                raise ValueError(mode)
            if row_index == 0:
                axis.set_title(f"Layer {layer}")
            if column_index == 1:
                axis.set_ylabel(f"{MODEL_LABELS[model]}\n{class_name}")
            axis.set_xticks([])
            axis.set_yticks([])
            map_axes.append(axis)
    assert image_handle is not None
    colorbar = figure.colorbar(image_handle, ax=map_axes, fraction=0.018, pad=0.01)
    colorbar.set_label("Cosine score" if mode == "raw" else "Per-map scaled value")
    scale_note = (
        "Raw cosine; fixed color scale [-1, 1]"
        if mode == "raw"
        else "Min-max visualization only; absolute magnitudes are not comparable"
    )
    figure.suptitle(
        f"{row['category']} · rank {int(row['selection_rank'])} · "
        f"{row['selection_metric']}={float(row['selection_value']):.4f}\n{scale_note}",
        fontsize=12,
    )
    figure.subplots_adjust(top=0.88, left=0.04, right=0.92, wspace=0.08, hspace=0.12)
    filename = safe_filename(
        f"{row['case_id']}_{row['category']}_{row['selection_scope_model']}_{row['image_id']}_class{primary_class:02d}_{mode}.png"
    )
    path = output / filename
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return filename


def main() -> None:
    args = parse_args()
    canonical = args.canonical_dir.resolve()
    voc_root = args.voc_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    selection = select_cases(canonical, args.top_n)
    source_lookup = load_sources(canonical)
    raw_paths: list[str] = []
    minmax_paths: list[str] = []
    for index, row in selection.iterrows():
        class_ids = [int(row["class_id"])]
        if pd.notna(row["comparison_class_id"]):
            class_ids.append(int(row["comparison_class_id"]))
        maps = load_case_maps(source_lookup, str(row["image_id"]), class_ids)
        original = transformed_image(voc_root, str(row["image_id"]), args.input_size)
        raw_paths.append(render_case(row, maps, original, output, "raw"))
        minmax_paths.append(render_case(row, maps, original, output, "minmax"))
        if (index + 1) % 10 == 0:
            print(f"Rendered {index + 1}/{len(selection)} selected cases", flush=True)
    selection["raw_cosine_figure"] = raw_paths
    selection["minmax_figure"] = minmax_paths
    selection.to_csv(output / "example_selection.csv", index=False, float_format="%.10g")
    expected_rows = args.top_n * 7
    if len(selection) != expected_rows:
        raise RuntimeError(f"expected {expected_rows} selections, found {len(selection)}")
    if len(list(output.glob("*.png"))) != expected_rows * 2:
        raise RuntimeError("example visualization count mismatch")


if __name__ == "__main__":
    main()
