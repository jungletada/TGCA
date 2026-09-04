import inspect

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from analysis.lazy_assignment.experiment2 import evaluation_metrics as metrics


def _all_strata_labels() -> np.ndarray:
    labels = np.zeros((6, 20), dtype=np.uint8)
    labels[0, [0]] = 1
    labels[1, [1]] = 1
    labels[2, [0, 1]] = 1
    labels[3, [1, 2]] = 1
    labels[4, [0, 1, 2]] = 1
    labels[5, [2, 3, 4]] = 1
    return labels


def _logits(images: int) -> np.ndarray:
    return np.random.default_rng(17).normal(size=(images, 20)).astype(np.float32)


def _confusions(images: int) -> np.ndarray:
    result = np.zeros((images, 21, 21), dtype=np.int64)
    for image_index in range(images):
        class_id = image_index % 3 + 1
        result[image_index, 0, 0] = 10 + image_index
        result[image_index, class_id, class_id] = 5
        result[image_index, class_id, 0] = 1
    return result


def _one_record(records, *, stratum, source, metric, class_id):
    selected = [
        row
        for row in records
        if row["label_stratum"] == stratum
        and row["model_or_delta"] == source
        and row["metric"] == metric
        and row["class_id"] == class_id
    ]
    assert len(selected) == 1
    return selected[0]


def test_cam_upsample_and_normalize_exactly_matches_repository_formula():
    cam = torch.randn(2, 28, 28, generator=torch.Generator().manual_seed(9))
    upsampled = F.interpolate(
        cam.unsqueeze(0),
        size=(448, 448),
        mode="bilinear",
        align_corners=False,
    )[0]
    minimum = upsampled.view(2, -1).min(dim=-1, keepdim=True)[0].view(2, 1, 1)
    maximum = upsampled.view(2, -1).max(dim=-1, keepdim=True)[0].view(2, 1, 1)
    expected = (upsampled - minimum) / (maximum - minimum + 1e-8)

    torch_result = metrics.upsample_and_normalize_active_cams(cam)
    numpy_result = metrics.upsample_and_normalize_active_cams(cam.numpy())
    flattened_result = metrics.upsample_and_normalize_active_cams(
        cam.numpy().reshape(2, -1)
    )

    assert isinstance(torch_result, torch.Tensor)
    assert isinstance(numpy_result, np.ndarray)
    torch.testing.assert_close(torch_result, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(numpy_result, expected.numpy())
    np.testing.assert_array_equal(flattened_result, expected.numpy())


def test_raw_final_cam_prediction_matches_fixed_background_reference():
    cam = torch.randn(2, 28, 28, generator=torch.Generator().manual_seed(31))
    active = np.asarray([1, 7], dtype=np.int64)
    normalized = metrics.upsample_and_normalize_active_cams(cam).numpy()
    reference_scores = np.concatenate(
        (
            np.full((1, 448, 448), 0.45, dtype=np.float32),
            normalized,
        )
    )
    reference_keys = np.asarray([0, 2, 8], dtype=np.int64)
    expected = reference_keys[np.argmax(reference_scores, axis=0)]

    prediction = metrics.raw_final_cam_prediction(cam, torch.from_numpy(active))

    assert prediction.shape == (448, 448)
    assert prediction.dtype == np.int64
    np.testing.assert_array_equal(prediction, expected)


def test_constant_cam_is_background_after_repository_minmax_normalization():
    cam = np.full((1, 28, 28), 3.0, dtype=np.float32)
    prediction = metrics.raw_final_cam_prediction(cam, np.asarray([4]))
    assert np.count_nonzero(prediction) == 0


def test_fixed_threshold_tie_resolves_to_prepended_background(monkeypatch):
    normalized = np.zeros((1, 448, 448), dtype=np.float32)
    normalized[0, 0, 0] = np.float32(0.45)
    normalized[0, 0, 1] = np.nextafter(np.float32(0.45), np.float32(1.0))
    monkeypatch.setattr(
        metrics,
        "upsample_and_normalize_active_cams",
        lambda _cam: normalized,
    )

    prediction = metrics.raw_final_cam_prediction(
        np.zeros((1, 28, 28), dtype=np.float32), [6]
    )

    assert prediction[0, 0] == 0
    assert prediction[0, 1] == 7


def test_voc_confusion_is_fixed_21_by_21_and_ignores_void():
    prediction = torch.tensor([[0, 1], [2, 20]], dtype=torch.int64)
    target = np.asarray([[0, 1], [255, 2]], dtype=np.uint8)

    confusion = metrics.voc_confusion_matrix(prediction, target)

    assert confusion.shape == (21, 21)
    assert confusion.dtype == np.int64
    assert confusion.sum() == 3
    assert confusion[0, 0] == 1
    assert confusion[1, 1] == 1
    assert confusion[2, 20] == 1


def test_raw_final_cam_confusion_uses_exact_448_crop_and_preserves_valid_count():
    row = np.linspace(0.0, 1.0, 28, dtype=np.float32)
    cam = np.broadcast_to(row[None, None, :], (1, 28, 28)).copy()
    prediction = metrics.raw_final_cam_prediction(cam, [3])
    target = prediction.astype(np.uint8)
    target[:10, :10] = 255

    confusion = metrics.raw_final_cam_confusion(cam, [3], torch.from_numpy(target))

    assert confusion.shape == (21, 21)
    assert confusion.sum() == 448 * 448 - 100
    assert confusion.trace() == confusion.sum()


def test_iou_from_confusion_matches_hand_calculation():
    confusion = np.zeros((21, 21), dtype=np.int64)
    confusion[0, 0] = 3
    confusion[0, 1] = 1
    confusion[1, 0] = 1
    confusion[1, 1] = 2

    iou, mean_iou = metrics.iou_from_confusion(confusion)

    assert iou[0] == pytest.approx(3 / 5)
    assert iou[1] == pytest.approx(2 / 4)
    assert np.isnan(iou[2:]).all()
    assert mean_iou == pytest.approx(0.55)

    with pytest.raises(ValueError, match=r"\[21,21\]"):
        metrics.iou_from_confusion(np.stack((confusion, confusion)))


def test_weighted_average_precision_groups_tied_scores_exactly():
    labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    scores = np.asarray([0.9, 0.8, 0.8, 0.1], dtype=np.float64)
    weights = np.asarray([2, 1, 3, 0], dtype=np.float64)
    # Threshold 0.9 contributes (2/5)*1; threshold 0.8 contributes
    # (3/5)*(5/6).  The zero-weight final item does not contribute.
    expected = (2 / 5) * 1.0 + (3 / 5) * (5 / 6)

    observed = metrics.average_precision_from_scores(labels, scores, weights)

    assert observed == pytest.approx(expected)
    assert metrics.average_precision_from_scores([1, 1], [0.0, 0.0]) == 1.0
    assert np.isnan(metrics.average_precision_from_scores([0, 0], [1.0, 0.0]))


def test_classification_point_metrics_accept_torch_and_return_macro_finite_classes():
    labels = _all_strata_labels().astype(bool)
    logits = _logits(len(labels))

    class_ap, mean_ap = metrics.classification_average_precision(
        torch.from_numpy(labels), torch.from_numpy(logits)
    )

    assert class_ap.shape == (20,)
    assert np.isfinite(class_ap[:5]).all()
    assert np.isnan(class_ap[5:]).all()
    assert mean_ap == pytest.approx(np.mean(class_ap[:5]))


def test_paired_classification_bootstrap_has_all_strata_and_exact_zero_delta():
    labels = _all_strata_labels()
    logits = _logits(len(labels))
    ids = [f"image_{index}" for index in range(len(labels))]

    records = metrics.paired_classification_bootstrap(
        ids,
        labels,
        logits,
        logits.copy(),
        repeats=128,
        seed=73,
        chunk_size=17,
    )

    assert {row["label_stratum"] for row in records} == set(metrics.LABEL_STRATA)
    assert len(records) == 4 * 21 * 3
    assert {row["logit_source"] for row in records} == {"class_token"}
    for row in records:
        if row["model_or_delta"] != "mctformer_plus_minus_mctformer":
            continue
        if row["bootstrap_valid_repeats"]:
            assert row["estimate"] == pytest.approx(0.0)
            assert row["ci_low"] == pytest.approx(0.0)
            assert row["ci_high"] == pytest.approx(0.0)
            assert row["bootstrap_unit"] == "image"
            assert row["delta_definition"] == "MCTformer+ - MCTformer"

    rare = _one_record(
        records,
        stratum="all",
        source="mctformer",
        metric="average_precision",
        class_id=4,
    )
    assert 0 < rare["bootstrap_valid_repeats"] < 128


def test_classification_bootstrap_is_deterministic_under_input_reordering():
    labels = _all_strata_labels()
    baseline = _logits(len(labels))
    comparison = baseline + np.linspace(-0.2, 0.2, 20, dtype=np.float32)
    ids = np.asarray([f"image_{index}" for index in range(len(labels))])
    permutation = np.asarray([3, 0, 5, 1, 4, 2])

    first = metrics.paired_classification_bootstrap(
        ids,
        labels,
        baseline,
        comparison,
        repeats=91,
        seed=11,
    )
    second = metrics.paired_classification_bootstrap(
        ids[permutation],
        labels[permutation],
        baseline[permutation],
        comparison[permutation],
        repeats=91,
        seed=11,
    )
    first_map = _one_record(
        first,
        stratum="all",
        source="mctformer_plus_minus_mctformer",
        metric="mean_average_precision",
        class_id=None,
    )
    second_map = _one_record(
        second,
        stratum="all",
        source="mctformer_plus_minus_mctformer",
        metric="mean_average_precision",
        class_id=None,
    )

    assert first_map == second_map


def test_nonidentical_model_delta_uses_replicate_wise_paired_draws(monkeypatch):
    labels = np.zeros((4, 20), dtype=np.uint8)
    labels[[0, 2], 0] = 1
    labels[[1, 3], 1] = 1
    baseline = np.zeros((4, 20), dtype=np.float64)
    comparison = np.zeros_like(baseline)
    baseline[:, 0] = [0.9, 0.8, 0.1, 0.0]
    comparison[:, 0] = [0.2, 0.9, 0.8, 0.1]
    draws = np.asarray(
        [
            [1, 1, 1, 1],
            [2, 0, 2, 0],
            [0, 2, 0, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.uint16,
    )
    monkeypatch.setattr(
        metrics,
        "_bootstrap_multiplicities",
        lambda images, repeats, seed: draws.copy(),
    )

    records = metrics.paired_classification_bootstrap(
        ["a", "b", "c", "d"],
        labels,
        baseline,
        comparison,
        repeats=len(draws),
        seed=101,
    )
    row = _one_record(
        records,
        stratum="all",
        source="mctformer_plus_minus_mctformer",
        metric="average_precision",
        class_id=0,
    )
    replicate_deltas = np.asarray(
        [
            metrics.average_precision_from_scores(
                labels[:, 0], comparison[:, 0], weights
            )
            - metrics.average_precision_from_scores(
                labels[:, 0], baseline[:, 0], weights
            )
            for weights in draws
        ]
    )
    finite = replicate_deltas[np.isfinite(replicate_deltas)]
    expected_low, expected_high = np.quantile(finite, (0.025, 0.975))

    assert row["bootstrap_valid_repeats"] == len(finite)
    assert row["ci_low"] == pytest.approx(expected_low)
    assert row["ci_high"] == pytest.approx(expected_high)


def test_paired_cam_bootstrap_has_classwise_macro_and_zero_paired_delta():
    labels = _all_strata_labels()
    confusions = _confusions(len(labels))
    ids = [f"image_{index}" for index in range(len(labels))]

    records = metrics.paired_cam_iou_bootstrap(
        ids,
        torch.from_numpy(labels),
        confusions,
        torch.from_numpy(confusions.copy()),
        repeats=101,
        seed=83,
        chunk_size=19,
    )

    assert {row["label_stratum"] for row in records} == set(metrics.LABEL_STRATA)
    assert len(records) == 4 * 22 * 3
    mean_delta = _one_record(
        records,
        stratum="all",
        source="mctformer_plus_minus_mctformer",
        metric="mean_intersection_over_union",
        class_id=None,
    )
    assert mean_delta["aggregation"] == "macro_class"
    assert mean_delta["cam_stage"] == "final_cam"
    assert mean_delta["background_threshold"] == pytest.approx(0.45)
    assert mean_delta["estimate"] == pytest.approx(0.0)
    assert mean_delta["ci_low"] == pytest.approx(0.0)
    assert mean_delta["ci_high"] == pytest.approx(0.0)
    class_delta = _one_record(
        records,
        stratum="all",
        source="mctformer_plus_minus_mctformer",
        metric="intersection_over_union",
        class_id=1,
    )
    assert class_delta["estimate"] == pytest.approx(0.0)


def test_paired_cam_bootstrap_rejects_misaligned_gt_marginals():
    labels = _all_strata_labels()
    baseline = _confusions(len(labels))
    comparison = baseline.copy()
    comparison[0, 0, 0] += 1

    with pytest.raises(ValueError, match="GT row marginals"):
        metrics.paired_cam_iou_bootstrap(
            [f"image_{index}" for index in range(len(labels))],
            labels,
            baseline,
            comparison,
            repeats=10,
        )


def test_public_bootstrap_defaults_are_preregistered_values():
    classification_signature = inspect.signature(
        metrics.paired_classification_bootstrap
    )
    cam_signature = inspect.signature(metrics.paired_cam_iou_bootstrap)
    assert classification_signature.parameters["repeats"].default == 5000
    assert classification_signature.parameters["seed"].default == 20260901
    assert cam_signature.parameters["repeats"].default == 5000
    assert cam_signature.parameters["seed"].default == 20260901


def test_evaluation_helpers_reject_invalid_shapes_and_class_ids():
    with pytest.raises(ValueError, match="28,28"):
        metrics.raw_final_cam_prediction(np.zeros((1, 14, 14), np.float32), [0])
    with pytest.raises(ValueError, match="unique"):
        metrics.raw_final_cam_prediction(np.zeros((2, 28, 28), np.float32), [1, 1])
    with pytest.raises(ValueError, match="448x448"):
        metrics.raw_final_cam_confusion(
            np.zeros((1, 28, 28), np.float32),
            [0],
            np.zeros((28, 28), np.uint8),
        )
    with pytest.raises(ValueError, match="at least one image"):
        metrics.paired_classification_bootstrap(
            [],
            np.zeros((0, 20), np.uint8),
            np.zeros((0, 20), np.float32),
            np.zeros((0, 20), np.float32),
            repeats=2,
        )
