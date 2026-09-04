from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from analysis.lazy_assignment.experiment2.common import sha256_file
from analysis.lazy_assignment.experiment2.patch_regions import patch_label_counts
from analysis.lazy_assignment.experiment2.render_experiment2_examples import (
    PANEL_COLUMNS,
    render_examples,
)
from analysis.lazy_assignment.experiment2.voc_semantic_dataset import (
    build_joint_transform,
)


def _write_fixture(root: Path, *, omit_npz_key: str | None = None) -> dict[str, Path]:
    voc = root / "VOC2012"
    jpeg_dir = voc / "JPEGImages"
    mask_dir = voc / "SegmentationClass"
    jpeg_dir.mkdir(parents=True)
    mask_dir.mkdir()
    image_id = "synthetic_0001"

    y, x = np.mgrid[:448, :448]
    rgb = np.stack(
        (
            (x % 256).astype(np.uint8),
            (y % 256).astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=-1,
    )
    mask = np.zeros((448, 448), dtype=np.uint8)
    mask[80:368, 60:216] = 3  # zero-based bird class 2
    mask[96:352, 232:400] = 19  # zero-based train class 18
    mask[:24, :24] = 255
    jpeg_path = jpeg_dir / f"{image_id}.jpg"
    mask_path = mask_dir / f"{image_id}.png"
    Image.fromarray(rgb).save(jpeg_path, quality=95, subsampling=0)
    Image.fromarray(mask).save(mask_path)

    with Image.open(jpeg_path) as image_source, Image.open(mask_path) as mask_source:
        _, transformed_mask, _ = build_joint_transform(448)(
            image_source.convert("RGB"), mask_source.copy()
        )
    counts = patch_label_counts(transformed_mask, 16)

    signal_root = root / "signals-plus"
    signal_dir = signal_root / "signals"
    signal_dir.mkdir(parents=True)
    patches = 28 * 28
    base = np.linspace(-0.9, 0.9, patches, dtype=np.float32)
    feature = np.empty((12, 2, patches), dtype=np.float32)
    attention = np.empty_like(feature)
    for layer in range(12):
        feature[layer, 0] = base + np.float32(layer * 0.001)
        feature[layer, 1] = -base + np.float32(layer * 0.001)
        raw_attention = np.stack(
            (
                np.arange(1, patches + 1, dtype=np.float32),
                np.arange(patches, 0, -1, dtype=np.float32),
            )
        )
        attention[layer] = raw_attention / raw_attention.sum(axis=1, keepdims=True)
    maps = {
        "image_id": np.asarray(image_id),
        "positive_class_ids": np.asarray([2, 18], dtype=np.int64),
        "grid_h": np.asarray(28, dtype=np.int32),
        "grid_w": np.asarray(28, dtype=np.int32),
        "patch_label_counts": counts.astype(np.uint16),
        "feature_post_scores": feature,
        "attn_c2p_conditional": attention,
        "patch_cam": np.stack((base + 1, 1 - base)).astype(np.float32),
        "c2p_cam": np.stack((base + 1.2, 1.2 - base)).astype(np.float32),
        "final_cam": np.stack((base + 1.4, 1.4 - base)).astype(np.float32),
    }
    if omit_npz_key is not None:
        maps.pop(omit_npz_key)
    npz_path = signal_dir / f"{image_id}.npz"
    np.savez_compressed(npz_path, **maps)
    (signal_root / "completion.json").write_text(
        json.dumps({"status": "complete", "run_kind": "smoke", "num_images": 1}),
        encoding="utf-8",
    )
    (signal_root / "metadata.json").write_text(
        json.dumps({"model": "mctformer_plus"}), encoding="utf-8"
    )
    (signal_root / "manifest.jsonl").write_text(
        json.dumps(
            {
                "image_id": image_id,
                "signal_path": f"signals/{image_id}.npz",
                "artifact_sha256": sha256_file(npz_path),
                "positive_class_ids": [2, 18],
                "grid_h": 28,
                "grid_w": 28,
                "num_layers": 12,
                "num_patches": patches,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    canonical_metadata = root / "canonical_metadata.json"
    canonical_metadata.write_text(
        json.dumps(
            {
                "status": "complete",
                "num_manifest_images_per_model": 1,
                "source_roots": {"mctformer_plus": str(signal_root.resolve())},
            }
        ),
        encoding="utf-8",
    )
    source_metadata = root / "source_metadata.json"
    source_metadata.write_text(
        json.dumps(
            {
                "integrity_passed": True,
                "dataset": {
                    "voc_root": str(voc.resolve()),
                    "input_size": 448,
                    "patch_size": 16,
                    "num_images": 1449,
                },
            }
        ),
        encoding="utf-8",
    )

    experiment1 = root / "experiment1" / "examples"
    experiment1.mkdir(parents=True)
    exp1_selection = experiment1 / "example_selection.csv"
    exp1_selection.write_text("case_id\nold_case\n", encoding="utf-8")
    raw_figure = experiment1 / "old_raw.png"
    minmax_figure = experiment1 / "old_minmax.png"
    Image.new("RGB", (16, 16), "red").save(raw_figure)
    Image.new("RGB", (16, 16), "blue").save(minmax_figure)

    selection = root / "example_selection.csv"
    pd.DataFrame(
        [
            {
                "case_id": "exp2_pair",
                "category": "shared_support_mostly_background",
                "model": "mctformer_plus",
                "image_id": image_id,
                "class_id": "",
                "companion_class_id": "",
                "class_a": 2,
                "class_b": 18,
                "positive_class_ids_json": "[2, 18]",
                "num_positive_classes": 2,
                "selection_metric": "shared_background_fraction",
                "selection_value": 0.75,
                "experiment1_case_id": "",
                "experiment1_raw_figure": "",
                "experiment1_minmax_figure": "",
                "source_table": "per_shared_patch_ownership.parquet",
            },
            {
                "case_id": "exp2_train",
                "category": "train_representative",
                "model": "mctformer_plus",
                "image_id": image_id,
                "class_id": 18,
                "companion_class_id": 2,
                "class_a": "",
                "class_b": "",
                "positive_class_ids_json": "[2, 18]",
                "num_positive_classes": 2,
                "selection_metric": "bg_tail_enrich_10",
                "selection_value": 1.1,
                "experiment1_case_id": "",
                "experiment1_raw_figure": "",
                "experiment1_minmax_figure": "",
                "source_table": "per_image_class_layer_signal.parquet",
            },
            {
                "case_id": "fixed_old",
                "category": "experiment1_fixed::A_fixed",
                "model": "mctformer",
                "image_id": "old_image",
                "class_id": 2,
                "companion_class_id": "",
                "class_a": "",
                "class_b": "",
                "positive_class_ids_json": "",
                "num_positive_classes": "",
                "selection_metric": "q95_l12_minus_l1",
                "selection_value": 0.3,
                "experiment1_case_id": "old_case",
                "experiment1_raw_figure": raw_figure.name,
                "experiment1_minmax_figure": minmax_figure.name,
                "source_table": str(exp1_selection.resolve()),
            },
        ]
    ).to_csv(selection, index=False)
    return {
        "selection": selection,
        "canonical_metadata": canonical_metadata,
        "source_metadata": source_metadata,
        "npz": npz_path,
        "raw_figure": raw_figure,
        "minmax_figure": minmax_figure,
    }


def test_model_free_renderer_renders_gt_cases_and_links_fixed_cases(
    tmp_path: Path,
) -> None:
    inputs = _write_fixture(tmp_path)
    before = sha256_file(inputs["npz"])
    output = tmp_path / "rendered"

    metadata = render_examples(
        inputs["selection"],
        inputs["canonical_metadata"],
        inputs["source_metadata"],
        output,
        dpi=72,
        command="synthetic deterministic render",
    )

    assert metadata["model_execution"] is False
    assert metadata["model_loaded"] is False
    assert metadata["canonical_is_full_set"] is False
    assert metadata["smoke_rendering_allowed_without_scientific_claims"] is True
    assert metadata["rendered_panel_files"] == 2
    assert metadata["fixed_experiment1_cases_linked"] == 1
    assert metadata["output_file_count"] == 4
    assert tuple(metadata["panel_columns"]) == PANEL_COLUMNS
    assert sha256_file(inputs["npz"]) == before

    manifest = pd.read_csv(output / "render_manifest.csv", keep_default_na=False)
    assert len(manifest) == 3
    pair = manifest[manifest["case_id"] == "exp2_pair"].iloc[0]
    assert pair["render_status"] == "rendered_from_existing_npz"
    assert json.loads(pair["class_ids_json"]) == [2, 18]
    assert pair["num_class_rows"] == 2
    assert Path(pair["panel_path"]).is_file()
    focal = manifest[manifest["case_id"] == "exp2_train"].iloc[0]
    assert json.loads(focal["class_ids_json"]) == [18, 2]
    assert focal["num_class_rows"] == 2
    with Image.open(pair["panel_path"]) as panel:
        assert panel.width > panel.height

    fixed = manifest[manifest["case_id"] == "fixed_old"].iloc[0]
    assert fixed["render_status"] == "linked_existing_not_redrawn"
    assert fixed["panel_path"] == ""
    assert Path(fixed["experiment1_raw_figure"]) == inputs["raw_figure"].resolve()
    assert Path(fixed["experiment1_minmax_figure"]) == inputs["minmax_figure"].resolve()
    assert sorted(path.name for path in (output / "panels").glob("*.png")) == [
        Path(value).name
        for value in manifest.loc[
            manifest["render_status"] == "rendered_from_existing_npz", "panel_path"
        ].sort_values()
    ]

    repeated = tmp_path / "rendered-repeat"
    render_examples(
        inputs["selection"],
        inputs["canonical_metadata"],
        inputs["source_metadata"],
        repeated,
        dpi=72,
        command="synthetic deterministic render repeat",
    )
    first_hashes = sorted(
        sha256_file(path) for path in (output / "panels").glob("*.png")
    )
    repeated_hashes = sorted(
        sha256_file(path) for path in (repeated / "panels").glob("*.png")
    )
    assert repeated_hashes == first_hashes


def test_renderer_refuses_missing_signal_instead_of_fabricating_panel(
    tmp_path: Path,
) -> None:
    inputs = _write_fixture(tmp_path, omit_npz_key="final_cam")
    output = tmp_path / "rendered"

    with pytest.raises(KeyError, match="final_cam"):
        render_examples(
            inputs["selection"],
            inputs["canonical_metadata"],
            inputs["source_metadata"],
            output,
            dpi=72,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".rendered.tmp-*"))


def test_renderer_refuses_single_class_panel_for_multilabel_case(
    tmp_path: Path,
) -> None:
    inputs = _write_fixture(tmp_path)
    selection = pd.read_csv(inputs["selection"], dtype=str, keep_default_na=False)
    selection.loc[selection["case_id"] == "exp2_train", "companion_class_id"] = ""
    selection.to_csv(inputs["selection"], index=False)

    with pytest.raises(ValueError, match="must display at least two positives"):
        render_examples(
            inputs["selection"],
            inputs["canonical_metadata"],
            inputs["source_metadata"],
            tmp_path / "rendered",
            dpi=72,
        )
