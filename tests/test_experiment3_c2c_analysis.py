from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis.lazy_assignment.experiment3.analyze_c2c_intervention import (
    BOOTSTRAP_REPEATS,
    CollectedRun,
    ValidatedRun,
    _collect_run,
    _map_metric_record,
    _native_stage_maps,
    _positive_recall_bootstrap,
    _shared_draws,
    _shared_support_bootstrap,
    _stage_transition_record,
    _transition_bootstrap,
    _validate_bootstrap_policy,
    _validate_source_equivalence,
)
from analysis.lazy_assignment.experiment3.cam_layer_intervention import (
    cam_threshold_grid,
)
from analysis.lazy_assignment.experiment3.c2c_intervention import (
    C2C_VARIANT_LAYERS_1BASED,
)


def _regions() -> np.ndarray:
    first = np.full(784, 4, dtype=np.int8)
    first[:4] = np.asarray([0, 1, 2, 3], dtype=np.int8)
    second = first.copy()
    second[0], second[1] = 1, 0
    return np.stack((first, second))


def _counts() -> np.ndarray:
    counts = np.zeros((784, 22), dtype=np.uint16)
    counts[0, 1] = 256
    counts[1, 2] = 256
    counts[2, 0] = 256
    counts[3, 1:3] = 128
    counts[4:, 21] = 256
    return counts


def _payload(variant: str, offset: float = 0.0) -> dict[str, np.ndarray]:
    positive = np.asarray([0, 1], dtype=np.int64)
    base = np.linspace(0.01, 1.0, 784, dtype=np.float32)
    feature = np.empty((3, 2, 784), dtype=np.float32)
    for layer in range(3):
        feature[layer, 0] = np.roll(base, layer * 7)
        feature[layer, 1] = np.roll(base, 101 + layer * 11)
    raw = np.maximum(feature + np.float32(offset), np.float32(1e-4))
    conditional = raw / raw.sum(axis=-1, keepdims=True, dtype=np.float32)
    patch_logits = np.stack((base - 0.4, np.roll(base, 101) - 0.4)).astype(np.float32)
    patch_cam = np.maximum(patch_logits, np.float32(0.0))
    preprop = np.sqrt(raw.mean(axis=0, dtype=np.float32) * patch_cam)
    final = np.sqrt(preprop + np.float32(0.01))
    logits = np.full(20, -1.0, dtype=np.float32)
    logits[0] = np.float32(1.0 + offset)
    logits[1] = np.float32(-0.1 + offset)
    patch_class_logits = logits.copy()
    labels = np.zeros(20, dtype=np.uint8)
    labels[positive] = 1
    heads = 2
    pre_offdiag = np.full((12, heads, 20), 0.1, dtype=np.float32)
    pre_diagonal = np.full((12, heads, 20), 0.2, dtype=np.float32)
    pre_class = pre_offdiag + pre_diagonal
    post_offdiag = pre_offdiag.copy()
    post_diagonal = pre_diagonal.copy()
    for layer in (number - 1 for number in C2C_VARIANT_LAYERS_1BASED[variant]):
        post_offdiag[layer] = 0
        post_diagonal[layer] = pre_class[layer]
    head_raw = np.full((3, heads, 2, 3), 0.001, dtype=np.float32)
    head_conditional = np.full((3, heads, 2, 3), 1.0 / 784, dtype=np.float32)
    confusions = np.zeros((41, 21, 21), dtype=np.int64)
    confusions[:, 0, 0] = 100
    confusions[:, 1, 1] = 20
    confusions[:, 2, 2] = 20
    return {
        "image_id": np.asarray("image_a"),
        "variant_code": np.asarray(variant),
        "positive_class_ids": positive,
        "image_labels": labels,
        "pair_class_ids": np.asarray([[0, 1]], dtype=np.int64),
        "late_layers_one_based": np.asarray([10, 11, 12], dtype=np.int16),
        "thresholds": cam_threshold_grid(),
        "patch_label_counts": _counts(),
        "region_masks_rho05": _regions(),
        "region_masks_rho07": _regions(),
        "class_logits_all": logits,
        "patch_class_logits_all": patch_class_logits,
        "patch_head_logits_positive": patch_logits,
        "feature_post_l10_l12": feature + np.float32(offset),
        "feature_both_axis_removed_l10_l12": feature * np.float32(0.5),
        "positive_pair_raw_cosine_l10_l12": np.full(
            (3, 1), 0.5 + offset, dtype=np.float32
        ),
        "positive_pair_residual_cosine_l10_l12": np.full(
            (3, 1), 0.2 + offset, dtype=np.float32
        ),
        "attention_c2p_raw_l10_l12": raw,
        "attention_c2p_conditional_l10_l12": conditional,
        "attention_head_region_raw_rho05": head_raw,
        "attention_head_region_conditional_rho05": head_conditional,
        "attention_head_region_raw_rho07": head_raw,
        "attention_head_region_conditional_rho07": head_conditional,
        "c2c_pre_offdiag_mass": pre_offdiag,
        "c2c_pre_diagonal_mass": pre_diagonal,
        "c2c_pre_class_mass": pre_class,
        "c2c_post_offdiag_mass": post_offdiag,
        "c2c_post_diagonal_mass": post_diagonal,
        "c2c_post_class_mass": pre_class.copy(),
        "final_cam": final,
        "threshold_confusions": confusions,
        "source_signal_sha256": np.asarray("a" * 64),
    }


def _collected_run(tmp_path: Path) -> CollectedRun:
    records = {}
    for index, variant in enumerate(C2C_VARIANT_LAYERS_1BASED):
        path = tmp_path / f"{variant}.npz"
        np.savez_compressed(path, **_payload(variant, index * 0.001))
        records[(variant, "image_a")] = {"artifact_path": path.name}
    run = ValidatedRun(
        model="mctformer_plus",
        root=tmp_path,
        metadata={},
        completion={},
        manifest=(),
        records=records,
        image_ids=("image_a",),
        positives=((0, 1),),
        run_kind="smoke",
        num_heads=2,
        source_root=tmp_path,
        source_records={},
    )
    return _collect_run(run)


def test_conditional_bg_mass_uses_complete_map_including_void() -> None:
    values = np.ones(784, dtype=np.float32)
    metrics = _map_metric_record(values, _regions()[0], mass=True)
    assert metrics["conditional_bg_mass"] == pytest.approx(1.0 / 784.0)


def test_transition_spearman_excludes_high_score_void_patches() -> None:
    regions = np.full(784, 4, dtype=np.int8)
    regions[:5] = np.asarray([0, 0, 1, 2, 3], dtype=np.int8)
    source = np.concatenate((np.arange(1.0, 6.0), np.arange(1000.0, 1779.0)))
    destination = np.concatenate((np.arange(1.0, 6.0), np.arange(1778.0, 999.0, -1.0)))
    result = _stage_transition_record(source, destination, regions)
    assert result["spearman"] == pytest.approx(1.0)


def test_native_stage_reconstruction_uses_exact_host_formulas() -> None:
    payload = _payload("C0")
    patch, plus_preprop, final = _native_stage_maps("mctformer_plus", payload)
    expected_patch = np.maximum(payload["patch_head_logits_positive"], 0)
    np.testing.assert_array_equal(patch, expected_patch)
    np.testing.assert_allclose(
        plus_preprop,
        np.sqrt(payload["attention_c2p_raw_l10_l12"].mean(axis=0) * patch),
        rtol=0,
        atol=0,
    )
    _, vanilla_preprop, _ = _native_stage_maps("mctformer", payload)
    np.testing.assert_allclose(
        vanilla_preprop,
        payload["attention_c2p_raw_l10_l12"].sum(axis=0) * patch,
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(final, payload["final_cam"])


def test_source_equivalence_includes_reconstructed_native_stages() -> None:
    payload = _payload("C0")
    patch, preprop, _ = _native_stage_maps("mctformer_plus", payload)
    source = {
        "positive_class_ids": payload["positive_class_ids"],
        "feature_post_scores": payload["feature_post_l10_l12"],
        "attn_c2p_raw": payload["attention_c2p_raw_l10_l12"],
        "attn_c2p_conditional": payload["attention_c2p_conditional_l10_l12"],
        "class_logits_all": payload["class_logits_all"],
        "patch_class_logits_all": payload["patch_class_logits_all"],
        "patch_logits": payload["patch_head_logits_positive"],
        "patch_cam": patch,
        "c2p_cam": preprop,
        "final_cam": payload["final_cam"],
    }
    _validate_source_equivalence("mctformer_plus", payload, source, Path("source"))
    source["c2p_cam"] = source["c2p_cam"].copy()
    source["c2p_cam"][0, 0] += np.float32(2e-6)
    with pytest.raises(RuntimeError, match="c2p_cam differs"):
        _validate_source_equivalence("mctformer_plus", payload, source, Path("source"))


def test_collection_covers_stage_transitions_and_shared_ownership(
    tmp_path: Path,
) -> None:
    collected = _collected_run(tmp_path)
    assert len(collected.transitions) == 6 * 2 * 2 * 2
    assert set(collected.transitions["transition"]) == {
        "patch_to_preprop",
        "preprop_to_final",
    }
    assert set(collected.shared_support["map_family"]) == {
        "feature_raw",
        "feature_axis_removed",
        "attention_raw",
        "attention_conditional",
        "patch_cam",
        "preprop_cam",
        "final_cam",
    }
    assert len(collected.shared_support) == 6 * 2 * (4 * 3 + 3)
    conditional = collected.region[
        (collected.region["variant_code"] == "C0")
        & (collected.region["map_family"] == "attention_conditional")
        & (collected.region["layer"] == 10)
        & (collected.region["rho_name"] == "rho05")
        & (collected.region["class_id"] == 0)
    ].iloc[0]
    expected = float(_payload("C0")["attention_c2p_conditional_l10_l12"][0, 0, 2])
    assert conditional["conditional_bg_mass"] == pytest.approx(expected)


def test_paired_products_use_image_draws_and_include_recall_classwise(
    tmp_path: Path,
) -> None:
    collected = _collected_run(tmp_path)
    labels = np.zeros((1, 20), dtype=np.uint8)
    labels[0, :2] = 1
    draws = _shared_draws(("image_a",), labels, repeats=7, seed=19)
    transitions = _transition_bootstrap(collected.transitions, draws)
    shared = _shared_support_bootstrap(collected.shared_support, draws)
    recall = _positive_recall_bootstrap(collected.classification, draws)
    for frame in (transitions, shared, recall):
        assert not frame.empty
        assert frame["bootstrap_unit"].eq("image").all()
        assert frame["bootstrap_repeats"].eq(7).all()
        assert "C4_minus_C0" in set(frame["series"])
    assert {"micro", "macro_class", "classwise"}.issubset(set(recall["aggregation"]))
    classwise = recall[recall["aggregation"] == "classwise"]
    assert set(classwise["semantic_class_id"].dropna().astype(int)) == {0, 1}
    assert set(recall["metric"]) == {
        "class_token_positive_recall",
        "patch_head_positive_recall",
    }


def test_bootstrap_policy_locks_production_to_5000() -> None:
    _validate_bootstrap_policy({"full"}, BOOTSTRAP_REPEATS, allow_smoke=False)
    _validate_bootstrap_policy({"smoke"}, 17, allow_smoke=True)
    with pytest.raises(RuntimeError, match="exactly 5000"):
        _validate_bootstrap_policy({"full"}, 17, allow_smoke=True)
    with pytest.raises(RuntimeError, match="--allow-smoke"):
        _validate_bootstrap_policy({"smoke"}, 17, allow_smoke=False)
    with pytest.raises(RuntimeError, match="explicit --allow-smoke"):
        _validate_bootstrap_policy({"smoke"}, BOOTSTRAP_REPEATS, allow_smoke=False)
