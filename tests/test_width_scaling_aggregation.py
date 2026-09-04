from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from tools.aggregate_mctformerplus_width_scaling import (
    cam_bootstrap,
    classification_bootstrap,
    semantic_bootstrap,
)
from tools.evaluate_cam_threshold_grid import (
    image_threshold_confusions,
    load_cam_winner,
)
from tools.evaluate_mctformerplus_classification import (
    _weighted_average_precision,
)
from tools.evaluate_mctformerplus_semantic_ownership import METRICS


def test_weighted_ap_equals_explicit_duplicate_bootstrap_with_ties():
    labels = np.asarray([1, 0, 1, 0, 1], dtype=np.uint8)
    scores = np.asarray([0.9, 0.9, 0.4, 0.2, 0.2], dtype=np.float32)
    weights = np.asarray([
        [2, 0, 1, 1, 1],
        [0, 2, 2, 0, 1],
    ])
    observed = _weighted_average_precision(labels, scores, weights)
    expected = []
    for row in weights:
        indices = np.repeat(np.arange(len(labels)), row)
        expected.append(average_precision_score(labels[indices], scores[indices]))
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)


def test_fast_threshold_confusions_equal_brute_force():
    generator = np.random.default_rng(4)
    thresholds = np.arange(60, dtype=np.float64) / 100
    scores = generator.choice(thresholds.tolist() + [0.8], size=(17, 13))
    classes = generator.integers(1, 21, size=scores.shape)
    target = generator.integers(0, 21, size=scores.shape).astype(np.uint8)
    target[0, 0] = 255
    observed = image_threshold_confusions(
        scores, classes, target, thresholds
    )
    valid = target != 255
    for index, threshold in enumerate(thresholds):
        prediction = np.where(scores > threshold, classes, 0)
        integer_target = target.astype(np.int64)
        expected = np.bincount(
            21 * integer_target[valid] + prediction[valid], minlength=441
        ).reshape(21, 21)
        np.testing.assert_array_equal(observed[index], expected)


def test_empty_native_cam_is_all_background_at_every_threshold(tmp_path):
    path = tmp_path / 'empty.npy'
    np.save(path, {})
    target = np.asarray([[0, 1], [2, 255]], dtype=np.uint8)
    thresholds = np.arange(60, dtype=np.float64) / 100
    scores, classes = load_cam_winner(path, empty_spatial_shape=target.shape)
    observed = image_threshold_confusions(
        scores, classes, target, thresholds
    )
    expected = np.zeros((21, 21), dtype=np.int32)
    expected[0, 0] = expected[1, 0] = expected[2, 0] = 1
    assert np.all(observed == expected[None])


def _write_classification_bundle(root, variant_index):
    directory = root / 'classification'
    directory.mkdir(parents=True)
    image_ids = np.asarray([f'image-{index:03d}' for index in range(60)])
    labels = np.zeros((60, 20), dtype=np.uint8)
    for image in range(60):
        labels[image, image % 20] = 1
        labels[image, (image + 7) % 20] = 1
    generator = np.random.default_rng(10 + variant_index)
    scores = generator.random((60, 20), dtype=np.float32)
    scores += labels * (0.2 * variant_index)
    np.savez_compressed(
        directory / 'classification_predictions.npz',
        image_ids=image_ids,
        labels=labels,
        class_token_scores=scores,
        patch_gwrp_scores=scores * 0.9,
    )
    with (directory / 'classification_per_class.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                'class_id', 'class_name', 'positive_images',
                'class_token_ap', 'patch_gwrp_ap',
            ),
        )
        writer.writeheader()
        for class_id in range(20):
            writer.writerow({
                'class_id': class_id,
                'class_name': str(class_id),
                'positive_images': int(labels[:, class_id].sum()),
                'class_token_ap': 0.5,
                'patch_gwrp_ap': 0.5,
            })


def _make_confusions(variant_index):
    thresholds = np.arange(60, dtype=np.float32) / 100
    confusions = np.zeros((10, 60, 21, 21), dtype=np.int32)
    for image in range(10):
        for threshold in range(60):
            confusions[image, threshold, 0, 0] = 100
            confusions[image, threshold, 1, 1] = 10 + variant_index
            confusions[image, threshold, 1, 0] = 5 - variant_index
    return thresholds, confusions


def _write_cam_bundle(root, variant_index):
    for split in ('train', 'val'):
        directory = root / f'cam_evaluation_{split}'
        directory.mkdir(parents=True)
        thresholds, confusions = _make_confusions(variant_index)
        np.savez_compressed(
            directory / 'per_image_confusions.npz',
            image_ids=np.asarray([f'image-{i}' for i in range(10)]),
            thresholds=thresholds,
            confusions=confusions,
        )


def _write_semantic_bundle(root, variant_index):
    directory = root / 'semantic_ownership'
    directory.mkdir(parents=True)
    sums = np.zeros((10, 12, len(METRICS)), dtype=np.float64)
    counts = np.ones_like(sums)
    for metric_index in range(len(METRICS)):
        sums[:, :, metric_index] = variant_index + metric_index / 100
    np.savez_compressed(
        directory / 'per_image_metric_sufficient_statistics.npz',
        image_ids=np.asarray([f'image-{i}' for i in range(10)]),
        layers=np.arange(1, 13),
        metric_names=np.asarray(METRICS),
        metric_sums=sums,
        metric_counts=counts,
    )


def _bundles(tmp_path):
    result = {}
    for index, variant in enumerate(('tiny', 'small', 'base')):
        root = tmp_path / variant
        root.mkdir()
        _write_classification_bundle(root, index)
        _write_cam_bundle(root, index)
        _write_semantic_bundle(root, index)
        result[variant] = {'root': root}
    return result


def test_paired_bootstraps_use_complete_images(tmp_path):
    bundles = _bundles(tmp_path)
    _, classification_rows, class_rows = classification_bootstrap(
        bundles, resamples=40, seed=2027
    )
    assert len(classification_rows) == 3 * 2 * 3
    assert len(class_rows) == 60
    assert {row['bootstrap_unit'] for row in classification_rows} == {'image'}

    cam_rows, point_rows = cam_bootstrap(
        bundles, small_tau=0.45, resamples=40, seed=2027
    )
    assert len(cam_rows) == 2 * 2 * 3
    assert len(point_rows) == 2 * 2 * 3
    assert all(row['bootstrap_unit'] == 'image' for row in cam_rows)
    base_small = next(
        row for row in cam_rows
        if row['split'] == 'val' and row['selection'] == 'fixed_0.45'
        and row['contrast'] == 'base-small'
    )
    assert base_small['delta'] > 0
    assert base_small['unit'] == 'percentage_points'

    semantic_rows, semantic_pairs = semantic_bootstrap(
        bundles, resamples=40, seed=2027
    )
    assert len(semantic_rows) == 17 * 3 * 12
    assert len(semantic_pairs) == 17 * 3 * 12
    assert all(row['bootstrap_unit'] == 'image' for row in semantic_pairs)
