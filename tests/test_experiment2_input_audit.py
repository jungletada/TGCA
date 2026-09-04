from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from analysis.lazy_assignment.experiment2.audit_experiment2_inputs import (
    EXPECTED_TRANSFORM,
    AuditConfig,
    run_audit,
)
from analysis.lazy_assignment.experiment2.common import (
    LOW_LEVEL_SOURCE_PATHS,
    resolve_completed_experiment1_root,
    sha256_file,
)


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_result(
    root: Path,
    model_name: str,
    image_ids: list[str],
    labels: dict[str, np.ndarray],
    checkpoint: Path,
    list_path: Path,
    labels_path: Path,
) -> None:
    root.mkdir(parents=True)
    (root / "scores").mkdir()
    checkpoint_hash = sha256_file(checkpoint)
    metadata = {
        "status": "complete",
        "model": {"name": model_name, "depth": 12, "patch_size": [16, 16]},
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": checkpoint_hash,
            "epoch": 44,
            "strict_load": {"missing_keys": [], "unexpected_keys": []},
        },
        "dataset": {
            "list_sha256": sha256_file(list_path),
            "labels_sha256": sha256_file(labels_path),
            "num_samples": len(image_ids),
        },
        "input": {
            "size": 448,
            "scale": 1.0,
            "horizontal_flip": False,
            "transform": EXPECTED_TRANSFORM,
        },
        "representation": "post_block_pre_final_norm",
        "positive_class_filter": True,
        "git": {"commit": "", "runtime_source_sha256": {}},
    }
    _json(root / "metadata.json", metadata)
    _json(root / "completion.json", {"status": "complete", "run_kind": "full"})
    manifest_lines = []
    for image_id in image_ids:
        positive = np.flatnonzero(labels[image_id] > 0).astype(np.int64)
        score_path = root / "scores" / f"{image_id}.npz"
        np.savez_compressed(
            score_path,
            image_id=np.asarray(image_id),
            positive_class_ids=positive,
            saved_class_ids=positive,
            scores_raw=np.zeros((12, len(positive), 784), dtype=np.float32),
            grid_h=np.asarray(28, dtype=np.int32),
            grid_w=np.asarray(28, dtype=np.int32),
        )
        manifest_lines.append(
            json.dumps(
                {
                    "image_id": image_id,
                    "positive_class_ids": positive.tolist(),
                    "saved_class_ids": positive.tolist(),
                    "score_path": f"scores/{image_id}.npz",
                    "grid_h": 28,
                    "grid_w": 28,
                    "num_layers": 12,
                    "num_patches": 784,
                }
            )
        )
    (root / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )


def _source_hashes(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): sha256_file(path)
        for root in roots
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _fixture(tmp_path: Path) -> tuple[AuditConfig, dict[str, Path]]:
    repo = tmp_path / "repo"
    for relative in LOW_LEVEL_SOURCE_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"synthetic {relative}\n", encoding="utf-8")

    voc = tmp_path / "voc"
    for relative in ("ImageLists", "ImageLabel", "JPEGImages", "SegmentationClass"):
        (voc / relative).mkdir(parents=True, exist_ok=True)
    image_ids = ["2007_000001", "2008_005245"]
    list_path = voc / "ImageLists/val_id.txt"
    list_path.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    labels = {
        image_ids[0]: np.eye(20, dtype=np.float32)[0],
        image_ids[1]: (
            np.eye(20, dtype=np.float32)[6] + np.eye(20, dtype=np.float32)[13]
        ),
    }
    labels_path = voc / "ImageLabel/cls_labels.npy"
    np.save(labels_path, labels, allow_pickle=True)

    for image_id in image_ids:
        Image.fromarray(np.full((64, 64, 3), 127, dtype=np.uint8), mode="RGB").save(
            voc / "JPEGImages" / f"{image_id}.jpg"
        )
    first_mask = np.zeros((64, 64), dtype=np.uint8)
    first_mask[12:52, 12:52] = 1
    first_mask[:2] = 255
    Image.fromarray(first_mask, mode="L").save(
        voc / "SegmentationClass/2007_000001.png"
    )
    mismatch_mask = np.zeros((64, 64), dtype=np.uint8)
    mismatch_mask[8:32, 8:32] = 7
    mismatch_mask[32:56, 8:32] = 14
    mismatch_mask[16:48, 36:56] = 15  # person exists in mask but not ImageLabel
    mismatch_mask[:2] = 255
    Image.fromarray(mismatch_mask, mode="L").save(
        voc / "SegmentationClass/2008_005245.png"
    )

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    base_checkpoint = checkpoints / "mctformerv2_final.pth"
    plus_checkpoint = checkpoints / "mctformerplus_final.pth"
    base_checkpoint.write_bytes(b"synthetic mctformer checkpoint")
    plus_checkpoint.write_bytes(b"synthetic mctformer plus checkpoint")

    base_root = tmp_path / "experiment1/mctformer/full"
    plus_root = tmp_path / "experiment1/mctformer_plus/full"
    _create_result(
        base_root,
        "mctformerv2",
        image_ids,
        labels,
        base_checkpoint,
        list_path,
        labels_path,
    )
    _create_result(
        plus_root,
        "mctformerplus",
        image_ids,
        labels,
        plus_checkpoint,
        list_path,
        labels_path,
    )

    paired = tmp_path / "experiment1/paired/full"
    (paired / "reports").mkdir(parents=True)
    for name in ("EXPERIMENT1_ANALYSIS_REPORT.md", "EXPERIMENT2_READINESS.md"):
        (paired / "reports" / name).write_text("synthetic\n", encoding="utf-8")
    _json(
        paired / "run_metadata.json",
        {
            "status": "complete",
            "source_immutability_passed": True,
            "source_roots": {
                "mctformer": str(base_root.resolve()),
                "mctformer_plus": str(plus_root.resolve()),
            },
            "source_checkpoints": {
                "mctformer": sha256_file(base_checkpoint),
                "mctformer_plus": sha256_file(plus_checkpoint),
            },
        },
    )
    config = AuditConfig(
        repo_root=repo,
        voc_root=voc,
        val_list=list_path,
        output_dir=tmp_path / "audit-output",
        mctformer_search_root=base_root.parent,
        mctformer_plus_search_root=plus_root.parent,
        paired_analysis_search_root=paired.parent,
    )
    return config, {
        "base": base_root,
        "plus": plus_root,
        "paired": paired,
        "base_checkpoint": base_checkpoint,
        "plus_checkpoint": plus_checkpoint,
        "voc": voc,
    }


def test_full_synthetic_audit_passes_and_preserves_sources(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    immutable_roots = [paths["base"], paths["plus"], paths["paired"], paths["voc"]]
    before = _source_hashes(immutable_roots)

    report = run_audit(config)

    assert report["integrity_passed"] is True
    assert report["error_count"] == 0
    assert report["warning_count"] == 1
    assert report["sources"]["mctformer"]["model_cli_name"] == "mctformerv2"
    assert report["sources"]["mctformer_plus"]["model_cli_name"] == "mctformerplus"
    assert report["dataset"] == {
        "voc_root": str(config.voc_root),
        "list_path": str(config.val_list),
        "labels_path": str(config.voc_root / "ImageLabel/cls_labels.npy"),
        "input_size": 448,
        "patch_size": 16,
        "num_images": 2,
    }
    assert report["gt_summary"]["raw_mask_image_label_mismatch_count"] == 1
    assert report["gt_summary"]["raw_mask_image_label_mismatches"] == [
        {
            "image_id": "2008_005245",
            "mask_only_class_ids": [14],
            "label_only_class_ids": [],
        }
    ]
    assert _source_hashes(immutable_roots) == before
    assert set(path.name for path in config.output_dir.iterdir()) == {
        "INPUT_AUDIT.md",
        "source_metadata.json",
        "gt_manifest.csv",
        "checkpoint_verification.json",
        "experiment1_linkage.json",
        "file_manifest_before.csv",
    }
    linkage = json.loads((config.output_dir / "experiment1_linkage.json").read_text())
    assert linkage["integrity_passed"] is True
    assert linkage["sources"]["mctformer"]["result_root"] == str(
        paths["base"].resolve()
    )
    assert linkage["dataset"]["num_images"] == 2


def test_discovery_rejects_multiple_completed_full_runs(tmp_path: Path) -> None:
    search = tmp_path / "results"
    for name in ("one", "two"):
        root = search / name
        _json(root / "completion.json", {"status": "complete", "run_kind": "full"})
        _json(root / "metadata.json", {"model": {"name": "mctformerv2"}})
    with pytest.raises(RuntimeError, match="exactly one completed full"):
        resolve_completed_experiment1_root(search, "mctformer")


def test_npz_positive_id_corruption_is_audit_failure(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    score_path = paths["base"] / "scores/2007_000001.npz"
    with np.load(score_path, allow_pickle=False) as artifact:
        values = {key: artifact[key] for key in artifact.files}
    values["positive_class_ids"] = np.asarray([1], dtype=np.int64)
    np.savez_compressed(score_path, **values)

    report = run_audit(replace(config, output_dir=tmp_path / "failed-audit"))

    assert report["integrity_passed"] is False
    assert any(
        issue["code"] == "npz_positive_ids_mismatch" for issue in report["issues"]
    )


def test_output_cannot_be_nested_in_an_immutable_source(tmp_path: Path) -> None:
    config, paths = _fixture(tmp_path)
    unsafe = paths["base"] / "audit-output"
    with pytest.raises(ValueError, match="immutable input"):
        run_audit(replace(config, output_dir=unsafe))
