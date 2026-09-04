from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from analysis.lazy_assignment.experiment2.build_experiment2_canonical import (
    ManifestEntry,
)
from analysis.lazy_assignment.experiment3.analyze_presence_axis import (
    CANONICAL_SCHEMAS,
    CONTROL_LAYERS,
    SHARED_OWNERSHIP_VARIANTS,
    AtomicParquetWriter,
    PresenceManifestEntry,
    _canonical_rows_for_image,
    _weighted_auc_samples,
    aligned_normalized_maps,
    clustered_projection_auc_summary,
    load_and_validate_presence_artifact,
    paired_ci_decision,
    pairwise_map_geometry,
    validation_a_decision,
)
from analysis.lazy_assignment.experiment2.metrics_region import map_overlap_metrics
from analysis.lazy_assignment.experiment3.bootstrap_experiment3 import (
    image_multinomial_draws,
)
from analysis.lazy_assignment.experiment3.common import sha256_file


def test_timing_alignment_uses_next_norm1_and_final_ln_only_at_l12():
    norm = np.stack([np.full((2, 3), layer, dtype=np.float32) for layer in range(12)])
    final = np.full((2, 3), 99, dtype=np.float32)
    aligned, timing = aligned_normalized_maps(
        {
            "feature_norm_scores": norm,
            "feature_final_norm_scores": final,
        }
    )
    np.testing.assert_array_equal(aligned[:11], norm[1:])
    np.testing.assert_array_equal(aligned[11], final)
    assert timing[0] == "post_L1_vs_norm1_L2"
    assert timing[10] == "post_L11_vs_norm1_L12"
    assert "analysis_only" in timing[11]


def test_pairwise_map_geometry_handles_ties_and_exact_topk():
    maps = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 1.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
        ]
    )
    result = pairwise_map_geometry(maps, ratios=(0.5,))
    assert result["spearman"].shape == (3, 3)
    assert result["spearman"][0, 0] == pytest.approx(1.0)
    assert result["top50_jaccard"][0, 1] == pytest.approx(1.0)
    assert result["top50_jaccard"][0, 2] == pytest.approx(0.0)
    np.testing.assert_allclose(result["spearman"], result["spearman"].T)


def test_weighted_auc_bootstrap_matches_sklearn_sample_weights_with_ties():
    image_ids = ("a", "b", "c")
    draws = image_multinomial_draws(image_ids, repeats=24, seed=17)
    scores = np.asarray([0.1, 0.5, 0.5, 0.9, 0.3, 0.7])
    labels = np.asarray([0, 1, 0, 1, 0, 1])
    columns = np.asarray([0, 0, 1, 1, 2, 2])
    point, samples = _weighted_auc_samples(
        scores, labels, columns, draws.multiplicities, chunk_size=5
    )
    assert point == pytest.approx(roc_auc_score(labels, scores))
    for repeat, multiplicity in enumerate(draws.multiplicities):
        weights = multiplicity[columns]
        if weights[labels == 1].sum() and weights[labels == 0].sum():
            expected = roc_auc_score(labels, scores, sample_weight=weights)
            assert samples[repeat] == pytest.approx(expected)
        else:
            assert math.isnan(samples[repeat])


def test_projection_summary_is_image_clustered_and_conditional():
    rows = []
    for image_index, image_id in enumerate(("a", "b", "c", "d")):
        for class_id in (0, 1):
            target = class_id == image_index % 2
            rows.append(
                {
                    "image_id": image_id,
                    "class_id": class_id,
                    "target": target,
                    "heldout_projection": float(target) + 0.1 * image_index,
                }
            )
    frame = pd.DataFrame(rows)
    draws = image_multinomial_draws(frame.image_id, repeats=80, seed=29)
    result = clustered_projection_auc_summary(frame, draws=draws)
    assert set(result["aggregation"]) == {"micro", "macro_class", "classwise"}
    assert set(result["bootstrap_unit"]) == {"image"}
    assert set(result["bootstrap_repeats"]) == {80}
    assert set(result["fit_uncertainty"]) == {
        "conditional_on_fixed_crossfit_directions"
    }
    micro = result[result.aggregation == "micro"].iloc[0]
    assert micro.estimate == pytest.approx(1.0)


def _valid_presence_fixture(tmp_path: Path):
    image_id = "2007_000001"
    positive = np.asarray([1, 4], dtype=np.int64)
    source_path = tmp_path / "source.npz"
    np.savez(source_path, marker=np.asarray(1))
    source_hash = sha256_file(source_path)
    source = {
        "positive_class_ids": positive,
        "feature_post_scores": np.zeros((12, 2, 784), dtype=np.float32),
        "feature_norm_scores": np.zeros((12, 2, 784), dtype=np.float32),
        "feature_final_norm_scores": np.zeros((2, 784), dtype=np.float32),
        "qk_mean_scores": np.zeros((12, 2, 784), dtype=np.float32),
        "attn_c2p_conditional": np.full((12, 2, 784), 1.0 / 784.0, dtype=np.float32),
        "region_masks_rho05": np.full((2, 784), 2, dtype=np.int8),
        "region_masks_rho07": np.full((2, 784), 2, dtype=np.int8),
        "patch_label_counts": np.zeros((784, 22), dtype=np.uint16),
        "class_logits_all": np.zeros(20, dtype=np.float32),
        "class_token_pairwise_cosine": np.tile(np.eye(2, dtype=np.float32), (12, 1, 1)),
    }
    source["patch_label_counts"][:, 0] = 256
    identity = np.tile(np.eye(20, dtype=np.float32), (12, 1, 1))
    zeros_positive = np.zeros((12, 2, 784), dtype=np.float32)
    zeros_control = np.zeros((6, 20, 784), dtype=np.float32)
    payload = {
        "image_id": np.asarray(image_id),
        "eval_fold": np.asarray(0, dtype=np.int8),
        "fit_fold": np.asarray(1, dtype=np.int8),
        "positive_class_ids": positive,
        "class_removed_scores": zeros_positive,
        "patch_removed_scores": zeros_positive,
        "both_removed_scores": zeros_positive,
        "shared_both_removed_scores": zeros_positive,
        "class_coefficients": np.zeros((12, 20), dtype=np.float32),
        "class_axis_energy": np.zeros((12, 20), dtype=np.float32),
        "class_norms": np.ones((12, 20), dtype=np.float32),
        "class_residual_norms": np.ones((12, 20), dtype=np.float32),
        "patch_coefficient_mean": np.zeros(12, dtype=np.float32),
        "patch_coefficient_std": np.zeros(12, dtype=np.float32),
        "patch_axis_energy_mean": np.zeros(12, dtype=np.float32),
        "patch_axis_energy_std": np.zeros(12, dtype=np.float32),
        "raw_pair_cosine_all": identity,
        "residual_pair_cosine_all": identity,
        "pair_axis_dot_all": np.zeros((12, 20, 20), dtype=np.float32),
        "pair_residual_dot_all": identity,
        "heldout_projection_all": np.zeros((12, 20), dtype=np.float32),
        "shared_axis_energy_all": np.zeros((12, 20), dtype=np.float32),
        "class_logits_all": np.zeros(20, dtype=np.float32),
        "source_signal_sha256": np.asarray(source_hash),
        "control_layer_ids": np.asarray(CONTROL_LAYERS, dtype=np.int64),
        "raw_control_all": zeros_control,
        "class_removed_control_all": zeros_control,
        "patch_removed_control_all": zeros_control,
        "both_removed_control_all": zeros_control,
        "shared_both_removed_control_all": zeros_control,
        "feature_norm_control_all": zeros_control,
        "qk_control_all": zeros_control,
        "attention_conditional_control_all": np.full(
            (6, 20, 784), 1.0 / 784.0, dtype=np.float32
        ),
    }
    presence_path = tmp_path / "presence.npz"
    np.savez(presence_path, **payload)
    presence_entry = PresenceManifestEntry(
        model="mctformer",
        image_id=image_id,
        eval_fold=0,
        positive_class_ids=tuple(positive.tolist()),
        artifact_path=presence_path,
        artifact_sha256=sha256_file(presence_path),
    )
    source_entry = ManifestEntry(
        model="mctformer",
        line_number=1,
        image_id=image_id,
        positive_class_ids=tuple(positive.tolist()),
        grid_h=28,
        grid_w=28,
        num_layers=12,
        num_patches=784,
        artifact_path=source_path,
        relative_path=source_path.name,
        artifact_sha256=source_hash,
    )
    return presence_entry, source_entry, source, payload


def test_presence_payload_fail_closed_schema_and_source_equivalence(tmp_path):
    presence_entry, source_entry, source, _ = _valid_presence_fixture(tmp_path)
    result = load_and_validate_presence_artifact(
        presence_entry,
        experiment2_entry=source_entry,
        experiment2_artifact=source,
    )
    assert tuple(result["control_layer_ids"]) == CONTROL_LAYERS


def test_image_canonicalization_covers_every_declared_table_schema(tmp_path):
    presence_entry, source_entry, source, _ = _valid_presence_fixture(tmp_path)
    presence = load_and_validate_presence_artifact(
        presence_entry,
        experiment2_entry=source_entry,
        experiment2_artifact=source,
    )
    rows = _canonical_rows_for_image("mctformer", presence_entry, presence, source)
    assert len(rows["token_axis"]) == 12 * 20
    assert len(rows["token_pairs"]) == 12 * 190
    assert len(rows["patch_axis"]) == 12
    assert len(rows["oof_projection"]) == 12 * 20
    assert len(rows["probe_region"]) == 2 * (12 * 9 + 1) * 2
    assert len(rows["positive_map_overlap"]) == 12 * 9 * 3
    assert len(rows["shared_ownership"]) == 12 * 6 * 2 * 3
    assert len(rows["probe_linkage"]) == 2 * 12 * 7 * 3
    assert len(rows["all_class_control_map_strata"]) == 6 * 8 * 3
    assert {row["variant"] for row in rows["shared_ownership"]} == set(
        SHARED_OWNERSHIP_VARIANTS
    )
    assert {row["variant"] for row in rows["all_class_control_map_strata"]} == {
        "raw",
        "class_removed",
        "patch_removed",
        "both_removed",
        "shared_both_removed",
        "norm1_pre_same_index",
        "qk_mean",
        "attn_c2p_conditional",
    }
    for name, table_rows in rows.items():
        assert table_rows
        assert set(table_rows[0]) == set(CANONICAL_SCHEMAS[name].names)


def test_positive_pair_and_probe_linkage_topk_exclude_high_score_void(tmp_path):
    presence_entry, _, source, presence = _valid_presence_fixture(tmp_path)
    void_count = 80
    first_valid_tail = slice(void_count, void_count + 71)
    second_valid_tail = slice(void_count + 71, void_count + 142)
    source["patch_label_counts"][:void_count, 0] = 0
    source["patch_label_counts"][:void_count, 21] = 256
    source["region_masks_rho05"][:, :void_count] = 4
    source["region_masks_rho07"][:, :void_count] = 4

    raw = source["feature_post_scores"]
    raw[:, :, :void_count] = 1.0
    raw[:, 0, first_valid_tail] = 0.9
    raw[:, 1, second_valid_tail] = 0.9
    normalized = source["feature_norm_scores"]
    normalized[:, :, :void_count] = 1.0
    normalized[:, 0, second_valid_tail] = 0.9
    normalized[:, 1, first_valid_tail] = 0.9
    source["feature_final_norm_scores"][:, :void_count] = 1.0
    source["feature_final_norm_scores"][0, second_valid_tail] = 0.9
    source["feature_final_norm_scores"][1, first_valid_tail] = 0.9

    unmasked_pair = map_overlap_metrics(raw[0, 0], raw[0, 1], ratio=0.10)
    assert unmasked_pair["topk_jaccard"] == pytest.approx(1.0)
    unmasked_link = map_overlap_metrics(raw[0, 0], normalized[1, 0], ratio=0.10)
    assert unmasked_link["topk_jaccard"] == pytest.approx(1.0)
    valid = np.ones(784, dtype=bool)
    valid[:void_count] = False
    expected_pair = map_overlap_metrics(raw[0, 0, valid], raw[0, 1, valid], ratio=0.10)
    expected_link = map_overlap_metrics(
        raw[0, 0, valid], normalized[1, 0, valid], ratio=0.10
    )
    assert expected_pair["spearman"] != pytest.approx(unmasked_pair["spearman"])
    assert expected_link["spearman"] != pytest.approx(unmasked_link["spearman"])

    rows = _canonical_rows_for_image("mctformer", presence_entry, presence, source)
    overlap = [
        row
        for row in rows["positive_map_overlap"]
        if row["layer"] == 1 and row["variant"] == "raw" and row["topk_ratio"] == 0.10
    ]
    assert len(overlap) == 1
    assert overlap[0]["topk_jaccard"] == pytest.approx(0.0)
    assert overlap[0]["spearman"] == pytest.approx(expected_pair["spearman"])
    linkage = [
        row
        for row in rows["probe_linkage"]
        if row["layer"] == 1
        and row["class_id"] == 1
        and row["link"] == "raw_to_norm_timing_aligned"
        and row["topk_ratio"] == 0.10
    ]
    assert len(linkage) == 1
    assert linkage[0]["topk_jaccard"] == pytest.approx(0.0)
    assert linkage[0]["spearman"] == pytest.approx(expected_link["spearman"])


def test_presence_payload_rejects_wrong_control_layer_contract(tmp_path):
    presence_entry, source_entry, source, payload = _valid_presence_fixture(tmp_path)
    payload["control_layer_ids"] = np.asarray([3, 4, 8, 9, 10, 11], dtype=np.int64)
    np.savez(presence_entry.artifact_path, **payload)
    broken_entry = PresenceManifestEntry(
        **{
            **presence_entry.__dict__,
            "artifact_sha256": sha256_file(presence_entry.artifact_path),
        }
    )
    with pytest.raises(ValueError, match="one-based"):
        load_and_validate_presence_artifact(
            broken_entry,
            experiment2_entry=source_entry,
            experiment2_artifact=source,
        )


def test_atomic_parquet_writer_has_typed_roundtrip_and_hash(tmp_path):
    writer = AtomicParquetWriter(
        tmp_path, "source_index", CANONICAL_SCHEMAS["source_index"]
    )
    writer.append(
        [
            {
                "model": "mctformer",
                "image_id": "a",
                "presence_artifact_path": "/presence/a.npz",
                "presence_artifact_sha256": "0" * 64,
                "experiment2_artifact_path": "/exp2/a.npz",
                "experiment2_artifact_sha256": "1" * 64,
                "source_hash_link_verified": True,
                "presence_schema_verified": True,
            }
        ]
    )
    metadata = writer.close()
    assert metadata["rows"] == 1
    assert len(metadata["sha256"]) == 64
    restored = pd.read_parquet(tmp_path / "source_index.parquet")
    assert restored.to_dict(orient="records")[0]["image_id"] == "a"


def test_frozen_paired_ci_and_combined_decision_rules():
    assert paired_ci_decision(-0.3, -0.01, expected="decrease") == "supported"
    assert paired_ci_decision(-0.3, 0.01, expected="decrease") == "not_supported"
    assert paired_ci_decision(0.01, 0.3, expected="increase") == "supported"
    decision = validation_a_decision(
        token_delta_ci=(-0.2, -0.1),
        map_delta_ci=(-0.08, -0.01),
        linkage_delta_ci=(0.01, 0.09),
        projection_auc_ci=(0.60, 0.75),
        final_signed_alignment=0.91,
    )
    assert decision["decision"] == "strong_support"
    partial = validation_a_decision(
        token_delta_ci=(-0.2, -0.1),
        map_delta_ci=(-0.02, 0.03),
        linkage_delta_ci=(-0.01, 0.02),
        projection_auc_ci=(0.45, 0.70),
        final_signed_alignment=0.89,
    )
    assert partial["decision"] == "partial_support_token_geometry_only"
    mixed = validation_a_decision(
        token_delta_ci=(-0.02, 0.03),
        map_delta_ci=(-0.08, -0.01),
        linkage_delta_ci=(0.01, 0.09),
        projection_auc_ci=(0.60, 0.75),
        final_signed_alignment=0.91,
    )
    assert mixed["decision"] == "mixed_or_indeterminate"
    unsupported = validation_a_decision(
        token_delta_ci=(-0.02, 0.03),
        map_delta_ci=(-0.02, 0.03),
        linkage_delta_ci=(-0.01, 0.02),
        projection_auc_ci=(0.45, 0.70),
        final_signed_alignment=0.89,
    )
    assert unsupported["decision"] == "not_supported"
