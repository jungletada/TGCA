"""Synthetic end-to-end checks for Experiment 2 canonical tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from analysis.lazy_assignment.experiment2.build_experiment2_canonical import (
    PATCH_NORM_JOINT_COLUMNS,
    REQUIRED_NPZ_KEYS,
    _feature_patch_norm_controls,
    assert_exact_manifest_match,
    build_canonical_tables,
    load_and_validate_artifact,
    load_signal_root,
)
from analysis.lazy_assignment.experiment2.canonical_io import TABLE_FILENAMES
from analysis.lazy_assignment.experiment2.patch_regions import (
    assign_patch_regions_from_counts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _signal_artifact(image_id: str, model_offset: float = 0.0) -> dict[str, np.ndarray]:
    layers, classes, patches, heads = 12, 2, 4, 2
    positives = np.asarray([0, 1], dtype=np.int64)
    counts = np.zeros((patches, 22), dtype=np.uint16)
    counts[0, 1] = 256  # class zero target
    counts[1, 2] = 256  # class one target
    counts[2, 0] = 256  # background
    counts[3, 1] = 128  # class-zero/background tie => mixed
    counts[3, 0] = 128
    regions05 = np.stack(
        [
            assign_patch_regions_from_counts(counts, class_id, rho=0.5)["region_codes"]
            for class_id in positives
        ]
    ).astype(np.uint8)
    regions07 = np.stack(
        [
            assign_patch_regions_from_counts(counts, class_id, rho=0.7)["region_codes"]
            for class_id in positives
        ]
    ).astype(np.uint8)

    feature_post = np.empty((layers, classes, patches), dtype=np.float32)
    feature_norm = np.empty_like(feature_post)
    qk_mean = np.empty_like(feature_post)
    for layer in range(layers):
        # For class zero, target score < background score: signed AUC must stay 0.
        feature_post[layer, 0] = np.asarray([0.0, 1.0, 3.0, 2.0]) + layer * 0.01
        feature_post[layer, 1] = np.asarray([1.0, 3.0, 0.0, 2.0]) + layer * 0.01
        feature_norm[layer] = feature_post[layer] * 0.5 - 0.2
        qk_mean[layer] = feature_norm[layer] * 0.7 + 0.1
    feature_post += np.float32(model_offset)
    qk_head_std = np.abs(qk_mean * 0.1).astype(np.float32)

    logits = qk_mean - qk_mean.max(axis=-1, keepdims=True)
    attention_conditional = np.exp(logits).astype(np.float32)
    attention_conditional /= attention_conditional.sum(axis=-1, keepdims=True)
    patch_mass = np.full((layers, classes), 0.8, dtype=np.float32)
    attention_raw = attention_conditional * patch_mass[..., None]
    official_raw = attention_raw[-3:].mean(axis=0).astype(np.float32)
    official_conditional = official_raw / official_raw.sum(axis=-1, keepdims=True)
    mid3_raw = attention_raw[3:6].mean(axis=0).astype(np.float32)
    mid3_conditional = mid3_raw / mid3_raw.sum(axis=-1, keepdims=True)

    patch_logits = np.asarray(
        [[-1.0, 2.0, 0.5, -0.25], [0.25, 3.0, -0.5, 1.0]], dtype=np.float32
    ) + np.float32(model_offset)
    patch_cam = np.maximum(patch_logits, 0.0).astype(np.float32)
    c2p_cam = np.sqrt(official_raw * patch_cam).astype(np.float32)
    final_cam = (c2p_cam + np.roll(c2p_cam, 1, axis=-1) * 0.1).astype(np.float32)
    diagnostics = {
        f"diagnostic_c2p_cam_l{layer}": np.sqrt(
            attention_raw[layer - 1] * patch_cam
        ).astype(np.float32)
        for layer in (10, 11, 12)
    }
    diagnostics["diagnostic_c2p_cam_mid3"] = np.sqrt(mid3_raw * patch_cam).astype(
        np.float32
    )

    pairwise = np.empty((layers, classes, classes), dtype=np.float32)
    pairwise[:] = np.asarray([[1.0, 0.2], [0.2, 1.0]], dtype=np.float32)
    head_regions = np.empty((layers, heads, classes, 3), dtype=np.float32)
    for layer in range(layers):
        for head in range(heads):
            head_regions[layer, head] = np.asarray(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32
            ) + np.float32(layer * 0.01 + head * 0.001)

    class_logits_all = np.linspace(-1.0, 1.0, 20, dtype=np.float32)
    class_logits_all[:2] = np.asarray([1.0, -0.2], dtype=np.float32)
    patch_class_logits_all = np.linspace(1.0, -1.0, 20, dtype=np.float32)
    patch_class_logits_all[:2] = np.asarray([0.8, 0.4], dtype=np.float32)
    raw_confusion = np.zeros((21, 21), dtype=np.int64)
    raw_confusion[0, 0] = 384
    raw_confusion[1, 1] = 384
    raw_confusion[2, 2] = 256

    artifact = {
        "image_id": np.asarray(image_id),
        "positive_class_ids": positives,
        "grid_h": np.asarray(2, dtype=np.int32),
        "grid_w": np.asarray(2, dtype=np.int32),
        "patch_label_counts": counts,
        "region_masks_rho05": regions05,
        "region_masks_rho07": regions07,
        "feature_post_scores": feature_post.astype(np.float32),
        "feature_norm_scores": feature_norm.astype(np.float32),
        "feature_final_norm_scores": feature_norm[-1].astype(np.float32),
        "qk_mean_scores": qk_mean.astype(np.float32),
        "qk_head_std": qk_head_std,
        "attn_c2p_raw": attention_raw.astype(np.float32),
        "attn_c2p_conditional": attention_conditional.astype(np.float32),
        "attn_patch_mass": patch_mass,
        "patch_logits": patch_logits.astype(np.float32),
        "patch_cam": patch_cam,
        "attn_official_raw": official_raw,
        "attn_official_conditional": official_conditional.astype(np.float32),
        "attn_mid3_raw": mid3_raw,
        "attn_mid3_conditional": mid3_conditional.astype(np.float32),
        "c2p_cam": c2p_cam,
        "final_cam": final_cam,
        "class_logits": np.asarray([1.0, -0.2], dtype=np.float32),
        "patch_class_logits": np.asarray([0.8, 0.4], dtype=np.float32),
        "class_logits_all": class_logits_all,
        "patch_class_logits_all": patch_class_logits_all,
        "raw_final_cam_confusion_t045": raw_confusion,
        "class_token_pairwise_cosine": pairwise,
        "patch_norms": np.tile(
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32), (layers, 1)
        ),
        "qk_head_region_mean_rho05": head_regions,
        "qk_head_region_mean_rho07": head_regions + np.float32(0.01),
        **diagnostics,
    }
    assert set(artifact) == REQUIRED_NPZ_KEYS
    return artifact


def _signal_root(tmp_path: Path, model: str, image_id: str = "2007_000001") -> Path:
    root = tmp_path / model
    (root / "signals").mkdir(parents=True)
    _write_json(root / "metadata.json", {"model": model, "input_size": 448})
    _write_json(root / "completion.json", {"status": "complete", "run_kind": "smoke"})
    artifact_path = root / "signals" / f"{image_id}.npz"
    np.savez_compressed(
        artifact_path,
        **_signal_artifact(
            image_id, model_offset=0.05 if model == "mctformer_plus" else 0.0
        ),
    )
    manifest = {
        "image_id": image_id,
        "positive_class_ids": [0, 1],
        "grid_h": 2,
        "grid_w": 2,
        "num_layers": 12,
        "num_patches": 4,
        "signal_path": f"signals/{image_id}.npz",
        "artifact_sha256": _sha256(artifact_path),
    }
    (root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return root


def _source_hashes(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): _sha256(path)
        for root in roots
        for path in sorted(value for value in root.rglob("*") if value.is_file())
    }


def test_canonical_builder_emits_all_tables_and_preserves_sources(
    tmp_path: Path,
) -> None:
    roots = {
        "mctformer": _signal_root(tmp_path, "mctformer"),
        "mctformer_plus": _signal_root(tmp_path, "mctformer_plus"),
    }
    before = _source_hashes(list(roots.values()))
    output = tmp_path / "canonical"
    metadata = build_canonical_tables(
        roots,
        output,
        require_full=False,
        expected_grid=(2, 2),
        flush_rows=75,
        command="synthetic canonical fixture",
    )

    assert metadata["status"] == "complete"
    assert metadata["source_manifests_exact_match"] is True
    assert metadata["source_immutability_verified"] is True
    assert _source_hashes(list(roots.values())) == before
    assert set(TABLE_FILENAMES.values()).issubset(
        path.name for path in output.iterdir()
    )
    assert (output / "canonical_metadata.json").is_file()
    for table_name, filename in TABLE_FILENAMES.items():
        path = output / filename
        parquet = pq.ParquetFile(path)
        assert parquet.metadata.num_rows == metadata["tables"][table_name]["rows"]
        assert metadata["tables"][table_name]["roundtrip_verified"] is True
        for row_group in range(parquet.num_row_groups):
            for column in range(parquet.metadata.num_columns):
                assert (
                    parquet.metadata.row_group(row_group).column(column).compression
                    == "ZSTD"
                )

    layer = pd.read_parquet(output / TABLE_FILENAMES["per_image_class_layer_signal"])
    expected_signals = {
        "feature_post",
        "feature_norm",
        "feature_final_norm",
        "qk_mean",
        "qk_head_std",
        "patch_norm",
        "attn_c2p_raw",
        "attn_c2p_conditional",
        "attn_official_raw",
        "attn_official_conditional",
        "attn_mid3_raw",
        "attn_mid3_conditional",
        "feature_post_relative",
        "feature_post_active_softmax",
    }
    assert expected_signals.issubset(set(layer["signal"]))
    assert metadata["probe_controls"]["claim_status"].startswith("probe controls")
    qk = layer[layer["signal"] == "qk_mean"]
    for head in range(6):
        for region in ("target", "other_fg", "bg"):
            assert f"qk_head{head}_{region}_mean" in qk.columns
    assert qk["qk_head0_target_mean"].notna().all()
    assert qk["qk_head2_target_mean"].isna().all()  # fixture has two heads
    signed = layer[
        (layer["model"] == "mctformer")
        & (layer["class_id"] == 0)
        & (layer["layer"] == 1)
        & (layer["signal"] == "feature_post")
        & (layer["rho"] == 0.5)
    ].iloc[0]
    assert signed["auc_target_bg"] == 0.0  # never flipped to 1.0
    assert signed["orientation_target_bg"] == -1.0
    assert set(PATCH_NORM_JOINT_COLUMNS).issubset(layer.columns)
    assert signed["feature_top10_bg_patch_count"] == 1
    assert signed["feature_top10_bg_patch_l2norm_mean"] == 3.0
    assert signed["feature_top10_bg_patch_l2norm_enrichment_vs_bg"] == 1.0
    assert signed["feature_top10_bg_below_valid_l2norm_median_fraction"] == 0.0
    assert signed["feature_top10_bg_above_valid_l2norm_q75_fraction"] == 0.0
    assert np.isfinite(signed["post_cosine_patch_l2norm_pearson_valid"])
    assert np.isnan(signed["post_cosine_patch_l2norm_pearson_bg"])

    shared = pd.read_parquet(output / TABLE_FILENAMES["per_shared_patch_ownership"])
    late = shared[
        (shared["signal"] == "feature_post")
        & (shared["layer"] == 10)
        & (shared["rho"] == 0.5)
        & (shared["topk_ratio"] == 0.10)
    ]
    assert not late.empty
    assert set(late["previous_layer_or_stage"]) == {"L9"}
    assert set(late["has_previous_layer"]) == {1}
    assert {"feature_post_relative", "feature_post_active_softmax"}.issubset(
        set(shared["signal"])
    )

    source_index = pd.read_parquet(output / TABLE_FILENAMES["source_index"])
    assert len(source_index) == 2
    assert source_index["hash_verified"].all()
    assert source_index["source_unchanged"].all()


def test_manifest_mismatch_is_rejected_before_output_creation(tmp_path: Path) -> None:
    left_path = _signal_root(tmp_path, "mctformer")
    right_path = _signal_root(tmp_path, "mctformer_plus", image_id="2007_000002")
    left = load_signal_root("mctformer", left_path, require_full=False)
    right = load_signal_root("mctformer_plus", right_path, require_full=False)
    with pytest.raises(RuntimeError, match="do not match exactly"):
        assert_exact_manifest_match(left, right)
    output = tmp_path / "canonical"
    with pytest.raises(RuntimeError, match="do not match exactly"):
        build_canonical_tables(
            {"mctformer": left_path, "mctformer_plus": right_path},
            output,
            require_full=False,
            expected_grid=(2, 2),
        )
    assert not output.exists()


def test_patch_norm_joint_control_handles_no_background_and_degenerate_pearson() -> (
    None
):
    result = _feature_patch_norm_controls(
        np.asarray([1.0, 1.0, 1.0, 1.0]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([0, 0, 1, 3], dtype=np.uint8),
    )
    assert result["feature_top10_bg_patch_count"] == 0
    assert np.isnan(result["post_cosine_patch_l2norm_pearson_valid"])
    assert np.isnan(result["post_cosine_patch_l2norm_pearson_bg"])
    assert np.isnan(result["feature_top10_bg_patch_l2norm_mean"])
    assert np.isnan(result["feature_top10_bg_patch_l2norm_enrichment_vs_bg"])
    assert np.isnan(result["feature_top10_bg_below_valid_l2norm_median_fraction"])
    assert np.isnan(result["feature_top10_bg_above_valid_l2norm_q75_fraction"])


def test_npz_schema_and_manifest_sha_are_enforced(tmp_path: Path) -> None:
    root = _signal_root(tmp_path, "mctformer")
    source = load_signal_root("mctformer", root, require_full=False)
    entry = source.entries[0]
    artifact = load_and_validate_artifact(entry, expected_grid=(2, 2))
    assert artifact["feature_post_scores"].shape == (12, 2, 4)

    artifact["attn_c2p_conditional"][0, 0] = 0.0
    np.savez_compressed(entry.artifact_path, **artifact)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        load_signal_root("mctformer", root, require_full=False)
