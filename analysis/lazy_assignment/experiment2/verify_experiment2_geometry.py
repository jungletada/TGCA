#!/usr/bin/env python3
"""Verify and visualize the matched Experiment 1 RGB / VOC mask geometry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.lazy_assignment.experiment2.patch_regions import (  # noqa: E402
    REGION_CODE_TO_NAME,
    assign_patch_regions,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (  # noqa: E402
    VOCSemanticDataset,
)
from datasets_cam import build_transform  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=20)
    args = parser.parse_args()
    if args.num_images < 1:
        parser.error("--num-images must be positive")
    return args


def _unnormalize(image: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406])[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225])[:, None, None]
    return np.clip((image * std + mean).transpose(1, 2, 0), 0.0, 1.0)


def run(args: argparse.Namespace) -> dict[str, object]:
    metadata = json.loads(args.source_metadata.resolve().read_text(encoding="utf-8"))
    if not metadata.get("integrity_passed"):
        raise RuntimeError("geometry verification requires a passing input audit")
    dataset_info = metadata["dataset"]
    if int(dataset_info["input_size"]) != 448 or int(dataset_info["patch_size"]) != 16:
        raise ValueError("Experiment 2 geometry is fixed to input=448, patch=16")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    panels = output / "panels"
    panels.mkdir()

    dataset = VOCSemanticDataset(
        dataset_info["voc_root"], dataset_info["list_path"], input_size=448
    )
    count = min(args.num_images, len(dataset))
    # Fixed coverage across the ordered validation list, including both ends.
    indices = np.linspace(0, len(dataset) - 1, count, dtype=int).tolist()
    exp1_transform = build_transform(
        is_train=False, make_cam=False, args=SimpleNamespace(input_size=448)
    )
    maximum_difference = 0.0
    records: list[dict[str, object]] = []
    for index in indices:
        sample = dataset[index]
        image_id = str(sample["name"])
        image_path = Path(dataset_info["voc_root"]) / "JPEGImages" / f"{image_id}.jpg"
        with Image.open(image_path) as source:
            expected = exp1_transform(source.convert("RGB"))
        observed = sample["image"]
        difference = float((expected - observed).abs().max().item())
        maximum_difference = max(maximum_difference, difference)
        if difference >= 1e-6:
            raise RuntimeError(f"RGB transform mismatch for {image_id}: {difference}")
        positive = np.flatnonzero(sample["label"].numpy() > 0)
        target = int(positive[0])
        assignment = assign_patch_regions(
            sample["mask"], target, patch_size=16, rho=0.5, valid_fraction=0.5
        )
        codes = np.asarray(assignment["region_codes"])

        figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        axes[0].imshow(_unnormalize(observed.numpy()))
        axes[0].set_title(f"RGB {image_id}")
        axes[1].imshow(sample["mask"].numpy(), vmin=0, vmax=20, interpolation="nearest")
        axes[1].set_title("Matched semantic mask")
        axes[2].imshow(codes, vmin=0, vmax=4, interpolation="nearest", cmap="tab10")
        axes[2].set_title(f"Patch regions, class={target}, rho=.5")
        for axis in axes:
            axis.axis("off")
        panel_path = panels / f"{image_id}.png"
        figure.savefig(panel_path, dpi=140)
        plt.close(figure)
        records.append(
            {
                "image_id": image_id,
                "dataset_index": int(index),
                "positive_class_ids": positive.tolist(),
                "visualized_target_class_id": target,
                "rgb_max_abs_diff": difference,
                "geometry": sample["mask_geometry"],
                "region_counts": assignment["composition"],
                "panel": str(panel_path),
            }
        )

    panel_images = []
    for record in records:
        with Image.open(record["panel"]) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail((840, 280), resample=Image.Resampling.LANCZOS)
            panel_images.append(thumbnail.copy())
    columns = 2
    rows = int(np.ceil(len(panel_images) / columns))
    overview = Image.new("RGB", (840 * columns, 280 * rows), color="white")
    for panel_index, panel_image in enumerate(panel_images):
        overview.paste(
            panel_image,
            ((panel_index % columns) * 840, (panel_index // columns) * 280),
        )
    overview_path = output / "geometry_overview_20.png"
    overview.save(overview_path)

    result = {
        "status": "complete",
        "source_metadata": str(args.source_metadata.resolve()),
        "num_images": len(records),
        "selection": "evenly spaced deterministic indices over ordered VOC val list",
        "indices": indices,
        "experiment1_rgb_max_abs_diff": maximum_difference,
        "tolerance": 1e-6,
        "orientation_review_status": "panels_generated_for_manual_review",
        "overview_path": str(overview_path),
        "region_codebook": {
            str(key): value for key, value in REGION_CODE_TO_NAME.items()
        },
        "records": records,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (output / "geometry_verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "command.txt").write_text(
        " ".join(["python", *sys.argv]) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "num_images": result["num_images"],
                "experiment1_rgb_max_abs_diff": result["experiment1_rgb_max_abs_diff"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
