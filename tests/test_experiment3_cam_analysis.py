from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.lazy_assignment.experiment3.analyze_cam_layer_readout import (
    CONFUSION_ENCODING,
    _assert_consumed_files_unchanged,
    _canonical_region_metrics,
    _paired_class_bootstrap,
    _pair_record,
    _validate_bootstrap_policy,
    decode_confusion,
    encode_confusion,
    execute,
    normalized_curve_auc,
    validate_inputs,
)
from analysis.lazy_assignment.experiment3.bootstrap_experiment3 import (
    image_multinomial_draws,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (
    CAM_VARIANT_SPECS,
    cam_threshold_grid,
)
from analysis.lazy_assignment.experiment3.common import sha256_file


def test_bootstrap_policy_locks_production_to_5000() -> None:
    _validate_bootstrap_policy({"full"}, 5000, allow_smoke=False)
    _validate_bootstrap_policy({"smoke"}, 17, allow_smoke=True)
    with pytest.raises(RuntimeError, match="exactly 5000"):
        _validate_bootstrap_policy({"full"}, 17, allow_smoke=True)
    with pytest.raises(RuntimeError, match="explicit --allow-smoke"):
        _validate_bootstrap_policy({"smoke"}, 5000, allow_smoke=False)


def _json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def test_conditional_bg_mass_uses_complete_spatial_map_including_void():
    values = np.ones(784, dtype=np.float64)
    # target, other foreground, background, mixed, then 780 void patches.
    regions = np.full(784, 4, dtype=np.int8)
    regions[:4] = np.asarray([0, 1, 2, 3], dtype=np.int8)

    metrics = _canonical_region_metrics(values, regions)

    assert metrics["conditional_bg_mass"] == pytest.approx(1.0 / 784.0)


def test_pair_record_uses_experiment2_mixed_void_key_contract():
    metrics = {
        "topk_jaccard": 0.1,
        "shared_set_size": 3,
        "shared_target_a_fraction": 0.2,
        "shared_target_b_fraction": 0.3,
        "shared_other_fg_fraction": 0.1,
        "shared_background_fraction": 0.25,
        "shared_mixed_void_fraction": 0.15,
        "shared_target_a_enrichment": 1.0,
        "shared_target_b_enrichment": 1.1,
        "shared_other_fg_enrichment": 0.9,
        "shared_background_enrichment": 1.2,
    }

    record = _pair_record(metrics)

    assert record["shared_mixed_fraction"] == pytest.approx(0.15)


def test_paired_class_bootstrap_zero_delta_and_equal_class_macro() -> None:
    records = []
    for image_id, class_id, value in (
        ("a", 0, 1.0),
        ("b", 0, 3.0),
        ("b", 1, 5.0),
    ):
        for variant in ("B0", "B1"):
            records.append(
                {
                    "image_id": image_id,
                    "class_id": class_id,
                    "variant_code": variant,
                    "value": value,
                }
            )
    frame = pd.DataFrame.from_records(records)
    draws = image_multinomial_draws(["a", "b"], repeats=40, seed=9)
    result = _paired_class_bootstrap(
        frame,
        comparison="B1",
        key_cols=("image_id", "class_id"),
        value_cols=("value",),
        class_col="class_id",
        draws=draws,
        identity={"model": "synthetic"},
    )

    assert set(result["aggregation"]) == {"micro", "macro_class", "class_wise"}
    delta = result[result["paired_delta"]]
    assert np.allclose(delta[["estimate", "ci_low", "ci_high"]], 0.0)
    macro_b0 = result[
        (result["aggregation"] == "macro_class") & (result["series"] == "B0")
    ].iloc[0]
    assert macro_b0["estimate"] == pytest.approx(3.5)
    assert int(macro_b0["num_classes"]) == 2


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _semantic_patch_counts(two_classes: bool) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((784, 22), dtype=np.uint16)
    thirds = (262, 524)
    counts[: thirds[0], 1] = 256
    if two_classes:
        counts[thirds[0] : thirds[1], 2] = 256
    else:
        counts[thirds[0] : thirds[1], 0] = 256
    counts[thirds[1] :, 0] = 256
    class0 = np.full(784, 2, dtype=np.int8)
    class0[: thirds[0]] = 0
    if two_classes:
        class0[thirds[0] : thirds[1]] = 1
        class1 = np.full(784, 2, dtype=np.int8)
        class1[: thirds[0]] = 1
        class1[thirds[0] : thirds[1]] = 0
        regions = np.stack((class0, class1))
    else:
        regions = class0[None, :]
    return counts, regions


def _confusion_stack(two_classes: bool) -> np.ndarray:
    thresholds = cam_threshold_grid()
    output = np.zeros((6, len(thresholds), 21, 21), dtype=np.int64)
    for variant in range(6):
        for threshold_index, _ in enumerate(thresholds):
            matrix = output[variant, threshold_index]
            matrix[0, 0] = 100
            matrix[1, 1] = 55 + variant
            matrix[1, 0] = 25 - variant
            if two_classes:
                matrix[2, 2] = 45 + variant
                matrix[2, 0] = 35 - variant
    return output


def _maps(num_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = np.linspace(0.01, 1.0, 784, dtype=np.float32)
    attention = np.empty((6, num_classes, 784), dtype=np.float32)
    preprop = np.empty_like(attention)
    final = np.empty_like(attention)
    for variant in range(6):
        for class_offset in range(num_classes):
            values = np.roll(base, 37 * variant + 181 * class_offset)
            attention[variant, class_offset] = values / values.sum(dtype=np.float32)
            preprop[variant, class_offset] = values * (variant + 1) / 6.0
            final[variant, class_offset] = np.sqrt(preprop[variant, class_offset])
    return attention * 0.8, attention, preprop, final


def _create_source_root(
    root: Path,
    model: str,
    image_ids: list[str],
    positives: list[list[int]],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for image_id, positive in zip(image_ids, positives):
        counts, regions = _semantic_patch_counts(len(positive) == 2)
        _, _, _, final = _maps(len(positive))
        confusions = _confusion_stack(len(positive) == 2)
        patch_cam = np.stack(
            [
                np.roll(np.linspace(0.01, 1.0, 784), 91 * offset)
                for offset in range(len(positive))
            ]
        ).astype(np.float32)
        path = root / "signals" / f"{image_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_id=np.asarray(image_id),
            positive_class_ids=np.asarray(positive, dtype=np.int64),
            grid_h=np.asarray(28, dtype=np.int32),
            grid_w=np.asarray(28, dtype=np.int32),
            patch_label_counts=counts,
            region_masks_rho05=regions,
            region_masks_rho07=regions,
            patch_cam=patch_cam,
            final_cam=final[0],
            raw_final_cam_confusion_t045=confusions[0, 25],
        )
        rows.append(
            {
                "image_id": image_id,
                "positive_class_ids": positive,
                "signal_path": str(path.relative_to(root)),
                "artifact_sha256": sha256_file(path),
            }
        )
    _jsonl(root / "manifest.jsonl", rows)
    _json(root / "metadata.json", {"status": "complete", "model": model})
    _json(root / "completion.json", {"status": "complete", "model": model})
    return {
        "root": str(root.resolve()),
        "manifest_rows": len(rows),
        "manifest_sha256": sha256_file(root / "manifest.jsonl"),
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "completion_sha256": sha256_file(root / "completion.json"),
    }


def _cam_contract() -> dict[str, object]:
    return {
        "variant_order": [spec.code for spec in CAM_VARIANT_SPECS],
        "variants": {
            spec.code: {
                "name": spec.name,
                "layers_one_based": list(spec.layers_one_based),
            }
            for spec in CAM_VARIANT_SPECS
        },
        "thresholds": cam_threshold_grid().tolist(),
        "primary_threshold": 0.45,
    }


def _create_cam_root(
    root: Path,
    model: str,
    image_ids: list[str],
    positives: list[list[int]],
    source_root: Path,
    source_metadata: Path,
    checkpoint: Path,
) -> None:
    source_rows = {
        row["image_id"]: row
        for row in (
            json.loads(line)
            for line in (source_root / "manifest.jsonl").read_text().splitlines()
        )
    }
    rows: list[dict[str, object]] = []
    for image_id, positive in zip(image_ids, positives):
        attention_raw, attention, preprop, final = _maps(len(positive))
        confusions = _confusion_stack(len(positive) == 2)
        source_digest = str(source_rows[image_id]["artifact_sha256"])
        path = root / "cams" / f"{image_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            image_id=np.asarray(image_id),
            positive_class_ids=np.asarray(positive, dtype=np.int64),
            variant_codes=np.asarray([spec.code for spec in CAM_VARIANT_SPECS]),
            thresholds=cam_threshold_grid(),
            attention_raw=attention_raw,
            attention_conditional=attention,
            preprop_cam=preprop,
            final_cam=final,
            confusions=confusions,
            source_signal_sha256=np.asarray(source_digest),
        )
        rows.append(
            {
                "image_id": image_id,
                "positive_class_ids": positive,
                "artifact_path": str(path.relative_to(root)),
                "artifact_sha256": sha256_file(path),
            }
        )
    _jsonl(root / "manifest.jsonl", rows)
    metadata = {
        "status": "complete",
        "analysis": "experiment3_validation_b_cam_layer_readout",
        "model": model,
        "run_kind": "smoke",
        "processed_images": len(rows),
        "source_metadata": str(source_metadata.resolve()),
        "source_metadata_sha256": sha256_file(source_metadata),
        "experiment2_signal_root": str(source_root.resolve()),
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
        },
        "execution": {"requested_images": len(rows)},
        "cam_contract": _cam_contract(),
        "source_variant_final_cam_max_abs_diff": {
            code: 0.0 for code in ("B0", "B1", "B2", "B3", "B5")
        },
        "source_patch_cam_max_abs_diff": 0.0,
        "native_cam_max_abs_diff": 0.0,
        "mask_patch_count_max_abs_diff": 0,
    }
    _json(root / "metadata.json", metadata)
    _json(
        root / "completion.json",
        {
            "status": "complete",
            "analysis": "experiment3_validation_b_cam_layer_readout",
            "model": model,
            "run_kind": "smoke",
            "num_images": len(rows),
        },
    )


def _synthetic_inputs(tmp_path: Path) -> argparse.Namespace:
    image_ids = ["image_a", "image_b", "image_c"]
    positives = [[0], [0, 1], [2]]
    val_list = tmp_path / "val.txt"
    val_list.write_text("\n".join(image_ids) + "\n", encoding="utf-8")
    labels = {
        image_id: np.eye(20, dtype=np.uint8)[positive].sum(axis=0)
        for image_id, positive in zip(image_ids, positives)
    }
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels, allow_pickle=True)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"immutable synthetic checkpoint")
    source_roots: dict[str, Path] = {}
    linkage_signals: dict[str, object] = {}
    for model in ("mctformer", "mctformer_plus"):
        source_root = tmp_path / "experiment2" / model
        source_roots[model] = source_root
        linkage_signals[model] = _create_source_root(
            source_root, model, image_ids[:2], positives[:2]
        )
    linkage_path = tmp_path / "audit" / "experiment2_linkage.json"
    _json(
        linkage_path,
        {
            "status": "complete",
            "experiment2_root": str((tmp_path / "experiment2").resolve()),
            "signals": linkage_signals,
        },
    )
    source_metadata = tmp_path / "audit" / "source_metadata.json"
    checkpoint_record = {
        "path": str(checkpoint.resolve()),
        "actual_sha256": sha256_file(checkpoint),
        "expected_sha256": sha256_file(checkpoint),
        "passed": True,
    }
    _json(
        source_metadata,
        {
            "status": "complete",
            "integrity_passed": True,
            "experiment2_linkage": str(linkage_path.resolve()),
            "experiment2_root": str((tmp_path / "experiment2").resolve()),
            "signal_roots": {
                model: str(root.resolve()) for model, root in source_roots.items()
            },
            "checkpoints": {
                "mctformer": checkpoint_record,
                "mctformer_plus": checkpoint_record,
            },
            "dataset": {
                "list_path": str(val_list.resolve()),
                "labels_path": str(labels_path.resolve()),
                "num_images": 3,
                "positive_image_class_pairs": 4,
                "multilabel_images": 1,
                "input_size": 448,
                "patch_size": 16,
            },
        },
    )
    run_roots: dict[str, Path] = {}
    for model in ("mctformer", "mctformer_plus"):
        root = tmp_path / "runs" / model
        run_roots[model] = root
        _create_cam_root(
            root,
            model,
            image_ids[:2],
            positives[:2],
            source_roots[model],
            source_metadata,
            checkpoint,
        )
    return argparse.Namespace(
        mctformer_run_root=run_roots["mctformer"],
        mctformer_plus_run_root=run_roots["mctformer_plus"],
        source_metadata=source_metadata,
        output_dir=tmp_path / "analysis",
        bootstrap_repeats=12,
        bootstrap_seed=17,
        allow_smoke=True,
    )


def test_confusion_codec_and_normalized_auc_contract() -> None:
    confusion = np.arange(21 * 21, dtype=np.int64).reshape(21, 21)
    blob = encode_confusion(confusion)
    assert len(blob) == 21 * 21 * 8
    assert np.array_equal(decode_confusion(blob), confusion)
    assert normalized_curve_auc([0.2, 0.4, 0.6], [0.3, 0.3, 0.3]) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="strictly increasing"):
        normalized_curve_auc([0.2, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError, match="invalid encoded"):
        decode_confusion(b"short")


def test_validation_rejects_manifest_artifact_hash_mismatch(tmp_path: Path) -> None:
    args = _synthetic_inputs(tmp_path)
    first = json.loads(
        (args.mctformer_run_root / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    artifact = args.mctformer_run_root / first["artifact_path"]
    with artifact.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(RuntimeError, match="CAM artifact hash mismatch"):
        validate_inputs(args)


def test_consumed_artifact_and_checkpoint_rehash_is_fail_closed(tmp_path: Path) -> None:
    args = _synthetic_inputs(tmp_path)
    validated = validate_inputs(args)
    observed = _assert_consumed_files_unchanged(validated.consumed_file_hashes)
    assert observed == validated.consumed_file_hashes
    # 2 hosts x (2 B artifacts + 2 Experiment 2 artifacts), plus one shared
    # synthetic checkpoint.
    assert len(observed) == 9

    checkpoint = next(
        Path(path) for path in observed if Path(path).name == "checkpoint.pth"
    )
    with checkpoint.open("ab") as stream:
        stream.write(b"changed after validation")
    with pytest.raises(RuntimeError, match="consumed immutable input changed"):
        _assert_consumed_files_unchanged(validated.consumed_file_hashes)


def test_execute_synthetic_smoke_produces_complete_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _synthetic_inputs(tmp_path)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "tgca-repro")
    output = execute(args)
    required = {
        "VALIDATION_B_CAM_LAYER_READOUT.md",
        "analysis_metadata.json",
        "canonical_image_cam_t045.parquet",
        "canonical_aggregate_threshold_confusions.parquet",
        "canonical_image_class_region_metrics.parquet",
        "canonical_positive_class_pair_metrics.parquet",
        "canonical_stage_transitions.parquet",
        "canonical_metadata.json",
        "consumed_input_manifest.csv",
        "threshold_curves.csv",
        "per_class_iou_thresholds.csv",
        "fixed_t045_metrics.csv",
        "native_b0_anchor.csv",
        "normalized_curve_auc.csv",
        "paired_cam_bootstrap.csv",
        "paired_region_bootstrap.csv",
        "paired_class_pair_bootstrap.csv",
        "paired_stage_transition_bootstrap.csv",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    metadata = json.loads((output / "analysis_metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["run_kind"] == "smoke"
    assert metadata["num_images"] == 2
    assert metadata["bootstrap"]["repeats"] == 12
    assert metadata["input_hashes_before_and_after_equal"] is True
    assert metadata["source_immutability_verified"] is True
    assert metadata["consumed_immutable_files"] == 9
    fixed = pd.read_parquet(output / "canonical_image_cam_t045.parquet")
    assert len(fixed) == 2 * 2 * 6
    assert fixed["confusion_encoding"].eq(CONFUSION_ENCODING).all()
    decoded = decode_confusion(fixed.iloc[0]["confusion"])
    assert decoded.shape == (21, 21)
    curves = pd.read_csv(output / "threshold_curves.csv")
    assert set(curves["threshold"]) == set(cam_threshold_grid())
    assert {"all", "single", "exactly_2"}.issubset(curves["label_stratum"])
    bootstrap = pd.read_csv(output / "paired_cam_bootstrap.csv")
    deltas = bootstrap[bootstrap["paired_delta"]]
    assert set(deltas["comparison_role"]) == {"primary", "secondary"}
    assert deltas["bootstrap_unit"].eq("image").all()
    region_bootstrap = pd.read_csv(output / "paired_region_bootstrap.csv")
    assert set(region_bootstrap["stage"]) == {"attention", "preprop", "final"}
    assert {"micro", "macro_class", "class_wise"}.issubset(
        region_bootstrap["aggregation"]
    )
    pair_bootstrap = pd.read_csv(output / "paired_class_pair_bootstrap.csv")
    assert set(pair_bootstrap["stage"]) == {"attention", "preprop", "final"}
    assert {"micro", "macro_class", "class_wise"}.issubset(
        pair_bootstrap["aggregation"]
    )
    transition_bootstrap = pd.read_csv(output / "paired_stage_transition_bootstrap.csv")
    assert set(transition_bootstrap["transition"]) == {
        "patch_cam_to_preprop",
        "preprop_to_final",
    }
    assert {"micro", "macro_class", "class_wise"}.issubset(
        transition_bootstrap["aggregation"]
    )
    report = (output / "VALIDATION_B_CAM_LAYER_READOUT.md").read_text()
    assert "[Fact]" in report
    assert "[Statistical inference]" in report
    assert "[Mechanistic interpretation]" in report
    assert "[Unsupported]" in report
    assert "Object-size stratification is N/A" in report
    assert "proposed methods" in report
    assert (output / "plots/mctformer_miou_threshold_curve.png").is_file()
