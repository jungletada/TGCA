"""Shared image-clustered bootstrap utilities for Experiment 3.

All resampling occurs at the image level.  A single immutable multiplicity
matrix can be passed to every model/variant in a paired family, which prevents
patches, positive classes, or class pairs from the same image being sampled as
independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis.lazy_assignment.experiment3.cam_layer_intervention import (
    cam_metrics_from_confusion,
)


DEFAULT_BOOTSTRAP_REPEATS = 5000
DEFAULT_BOOTSTRAP_SEED = 20260901
CI_QUANTILES = (0.025, 0.975)
BOOTSTRAP_UNIT = "image"


@dataclass(frozen=True)
class ImageBootstrapDraws:
    """Deterministic whole-image multinomial multiplicities."""

    image_ids: tuple[str, ...]
    multiplicities: np.ndarray
    repeats: int
    seed: int


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    valid_repeats: int
    finite_images: int
    total_images: int
    finite_rows: int
    total_rows: int
    finite_classes: int | None


_CAM_SCALAR_METRICS = (
    "mean_iou",
    "binary_foreground_precision",
    "binary_foreground_recall",
    "semantic_correct_foreground_precision",
    "semantic_correct_foreground_recall",
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _canonical_image_ids(image_ids: Sequence[str]) -> tuple[str, ...]:
    values = list(image_ids)
    if not values:
        raise ValueError("image_ids must be non-empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise TypeError("every image ID must be a non-empty string")
    return tuple(sorted(set(values)))


def image_multinomial_draws(
    image_ids: Sequence[str],
    *,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ImageBootstrapDraws:
    """Generate deterministic multinomial counts over sorted unique images."""

    repeat_count = _positive_integer(repeats, "repeats")
    seed_value = _nonnegative_integer(seed, "seed")
    clusters = _canonical_image_ids(image_ids)
    num_images = len(clusters)
    rng = np.random.default_rng(seed_value)
    probabilities = np.full(num_images, 1.0 / num_images, dtype=np.float64)
    multiplicities = rng.multinomial(num_images, probabilities, size=repeat_count)
    dtype = np.uint16 if num_images <= np.iinfo(np.uint16).max else np.uint32
    multiplicities = multiplicities.astype(dtype, copy=False)
    multiplicities.setflags(write=False)
    return ImageBootstrapDraws(
        image_ids=clusters,
        multiplicities=multiplicities,
        repeats=repeat_count,
        seed=seed_value,
    )


def _validate_draws(draws: ImageBootstrapDraws) -> ImageBootstrapDraws:
    if not isinstance(draws, ImageBootstrapDraws):
        raise TypeError("draws must be an ImageBootstrapDraws instance")
    if not draws.image_ids or len(set(draws.image_ids)) != len(draws.image_ids):
        raise ValueError("draw image IDs must be non-empty and unique")
    if tuple(sorted(draws.image_ids)) != draws.image_ids:
        raise ValueError("draw image IDs must be sorted")
    values = np.asarray(draws.multiplicities)
    expected = (draws.repeats, len(draws.image_ids))
    if values.shape != expected or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"draw multiplicities must be an integer array {expected}")
    if np.any(values < 0) or not np.all(values.sum(axis=1) == len(draws.image_ids)):
        raise ValueError("every bootstrap draw must resample exactly N whole images")
    if draws.repeats < 1 or draws.seed < 0:
        raise ValueError("draw metadata is invalid")
    return draws


def multiplicities_for_rows(
    draws: ImageBootstrapDraws, row_image_ids: Sequence[str]
) -> np.ndarray:
    """Expand image multiplicities to rows for tests or small diagnostics.

    Full analyses should use the cluster-sufficient-statistic functions below
    rather than materializing a potentially large repeat-by-row matrix.
    """

    checked = _validate_draws(draws)
    lookup = {image_id: index for index, image_id in enumerate(checked.image_ids)}
    columns: list[int] = []
    for image_id in row_image_ids:
        if image_id not in lookup:
            raise ValueError(f"row references image outside draws: {image_id!r}")
        columns.append(lookup[image_id])
    return checked.multiplicities[:, columns]


def _finite_interval(samples: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    low, high = np.quantile(finite, CI_QUANTILES)
    return float(low), float(high), int(len(finite))


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")


def _draws_for_frame(
    frame: pd.DataFrame,
    cluster_col: str,
    draws: ImageBootstrapDraws | None,
    repeats: int,
    seed: int,
) -> ImageBootstrapDraws:
    _require_columns(frame, (cluster_col,))
    frame_ids = _canonical_image_ids(frame[cluster_col].tolist())
    if draws is None:
        return image_multinomial_draws(frame_ids, repeats=repeats, seed=seed)
    checked = _validate_draws(draws)
    missing = set(frame_ids).difference(checked.image_ids)
    if missing:
        raise ValueError(
            f"frame images are absent from supplied draws: {sorted(missing)}"
        )
    return checked


def _cluster_sums_counts(
    frame: pd.DataFrame,
    value_cols: Sequence[str],
    cluster_col: str,
    draws: ImageBootstrapDraws,
) -> tuple[np.ndarray, np.ndarray]:
    lookup = {image_id: index for index, image_id in enumerate(draws.image_ids)}
    try:
        indices = np.asarray(
            [lookup[value] for value in frame[cluster_col]], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"frame image is absent from supplied draws: {error.args[0]}")
    values = (
        frame[list(value_cols)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    finite = np.isfinite(values)
    sums = np.zeros((len(draws.image_ids), len(value_cols)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for column in range(len(value_cols)):
        np.add.at(
            sums[:, column],
            indices[finite[:, column]],
            values[finite[:, column], column],
        )
        np.add.at(counts[:, column], indices[finite[:, column]], 1.0)
    return sums, counts


def _micro_estimates(
    sums: np.ndarray,
    counts: np.ndarray,
    draws: ImageBootstrapDraws,
    *,
    total_rows: int,
    chunk_size: int = 256,
) -> list[BootstrapEstimate]:
    point_denominator = counts.sum(axis=0)
    points = np.divide(
        sums.sum(axis=0),
        point_denominator,
        out=np.full(sums.shape[1], np.nan, dtype=np.float64),
        where=point_denominator > 0,
    )
    samples = np.full((draws.repeats, sums.shape[1]), np.nan, dtype=np.float64)
    for start in range(0, draws.repeats, chunk_size):
        weights = draws.multiplicities[start : start + chunk_size].astype(
            np.float64, copy=False
        )
        numerator = weights @ sums
        denominator = weights @ counts
        np.divide(
            numerator,
            denominator,
            out=samples[start : start + len(weights)],
            where=denominator > 0,
        )
    estimates: list[BootstrapEstimate] = []
    for column in range(sums.shape[1]):
        low, high, valid = _finite_interval(samples[:, column])
        estimates.append(
            BootstrapEstimate(
                estimate=float(points[column]),
                ci_low=low,
                ci_high=high,
                valid_repeats=valid,
                finite_images=int(np.count_nonzero(counts[:, column] > 0)),
                total_images=len(draws.image_ids),
                finite_rows=int(point_denominator[column]),
                total_rows=int(total_rows),
                finite_classes=None,
            )
        )
    return estimates


def _macro_class_estimates(
    frame: pd.DataFrame,
    value_cols: Sequence[str],
    cluster_col: str,
    class_col: str,
    draws: ImageBootstrapDraws,
    *,
    chunk_size: int = 256,
) -> list[BootstrapEstimate]:
    _require_columns(frame, (class_col,))
    if bool(frame[class_col].isna().any()):
        raise ValueError(f"{class_col} must not contain missing values")
    classes = tuple(sorted(frame[class_col].dropna().unique().tolist()))
    if not classes:
        return []
    class_sufficient: list[tuple[np.ndarray, np.ndarray]] = []
    point_class = np.full((len(classes), len(value_cols)), np.nan, dtype=np.float64)
    image_has_metric = np.zeros((len(draws.image_ids), len(value_cols)), dtype=bool)
    finite_rows = np.zeros(len(value_cols), dtype=np.int64)
    for class_index, class_id in enumerate(classes):
        subset = frame[frame[class_col] == class_id]
        sums, counts = _cluster_sums_counts(subset, value_cols, cluster_col, draws)
        denominators = counts.sum(axis=0)
        np.divide(
            sums.sum(axis=0),
            denominators,
            out=point_class[class_index],
            where=denominators > 0,
        )
        image_has_metric |= counts > 0
        finite_rows += denominators.astype(np.int64)
        class_sufficient.append((sums, counts))
    finite_point = np.isfinite(point_class)
    point_class_count = finite_point.sum(axis=0)
    points = np.divide(
        np.nansum(point_class, axis=0),
        point_class_count,
        out=np.full(len(value_cols), np.nan, dtype=np.float64),
        where=point_class_count > 0,
    )
    samples = np.full((draws.repeats, len(value_cols)), np.nan, dtype=np.float64)
    for start in range(0, draws.repeats, chunk_size):
        weights = draws.multiplicities[start : start + chunk_size].astype(
            np.float64, copy=False
        )
        sum_of_class_means = np.zeros((len(weights), len(value_cols)))
        valid_class_count = np.zeros_like(sum_of_class_means)
        for sums, counts in class_sufficient:
            numerator = weights @ sums
            denominator = weights @ counts
            class_mean = np.full_like(numerator, np.nan)
            np.divide(numerator, denominator, out=class_mean, where=denominator > 0)
            finite = np.isfinite(class_mean)
            sum_of_class_means += np.where(finite, class_mean, 0.0)
            valid_class_count += finite
        np.divide(
            sum_of_class_means,
            valid_class_count,
            out=samples[start : start + len(weights)],
            where=valid_class_count > 0,
        )
    estimates: list[BootstrapEstimate] = []
    for column in range(len(value_cols)):
        low, high, valid = _finite_interval(samples[:, column])
        estimates.append(
            BootstrapEstimate(
                estimate=float(points[column]),
                ci_low=low,
                ci_high=high,
                valid_repeats=valid,
                finite_images=int(image_has_metric[:, column].sum()),
                total_images=len(draws.image_ids),
                finite_rows=int(finite_rows[column]),
                total_rows=len(frame),
                finite_classes=int(point_class_count[column]),
            )
        )
    return estimates


def _estimate_record(
    metric: str,
    aggregation: str,
    estimate: BootstrapEstimate,
    draws: ImageBootstrapDraws,
    identity: Mapping[str, object] | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "aggregation": aggregation,
        "metric": metric,
        "estimate": estimate.estimate,
        "ci_low": estimate.ci_low,
        "ci_high": estimate.ci_high,
        "num_images": estimate.finite_images,
        "num_images_total": estimate.total_images,
        "num_rows": estimate.finite_rows,
        "num_rows_total": estimate.total_rows,
        "num_classes": estimate.finite_classes,
        "bootstrap_repeats": draws.repeats,
        "bootstrap_valid_repeats": estimate.valid_repeats,
        "bootstrap_valid_fraction": estimate.valid_repeats / draws.repeats,
        "bootstrap_seed": draws.seed,
        "bootstrap_unit": BOOTSTRAP_UNIT,
        "ci_method": "95% percentile",
    }
    if identity:
        overlap = set(record).intersection(identity)
        if overlap:
            raise ValueError(
                f"identity cannot replace statistic fields: {sorted(overlap)}"
            )
        record.update(identity)
    return record


def summarize_clustered_means(
    frame: pd.DataFrame,
    *,
    value_cols: Sequence[str],
    draws: ImageBootstrapDraws | None = None,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    cluster_col: str = "image_id",
    class_col: str = "class_id",
    include_macro_class: bool = True,
    identity: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Summarize row-micro and equal-class means with whole-image draws."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        return pd.DataFrame()
    if not value_cols or len(set(value_cols)) != len(value_cols):
        raise ValueError("value_cols must be non-empty and unique")
    _require_columns(frame, (cluster_col, *value_cols))
    selected_draws = _draws_for_frame(frame, cluster_col, draws, repeats, seed)
    sums, counts = _cluster_sums_counts(frame, value_cols, cluster_col, selected_draws)
    rows: list[dict[str, object]] = []
    for metric, estimate in zip(
        value_cols,
        _micro_estimates(sums, counts, selected_draws, total_rows=len(frame)),
    ):
        rows.append(
            _estimate_record(metric, "micro", estimate, selected_draws, identity)
        )
    if include_macro_class:
        macro = _macro_class_estimates(
            frame, value_cols, cluster_col, class_col, selected_draws
        )
        for metric, estimate in zip(value_cols, macro):
            rows.append(
                _estimate_record(
                    metric, "macro_class", estimate, selected_draws, identity
                )
            )
    return pd.DataFrame.from_records(rows)


def paired_clustered_mean_summary(
    frame: pd.DataFrame,
    *,
    system_col: str,
    baseline: str,
    comparison: str,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    draws: ImageBootstrapDraws | None = None,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    cluster_col: str = "image_id",
    class_col: str = "class_id",
    include_macro_class: bool = True,
    identity: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Return baseline, comparison, and paired deltas on exact common rows."""

    if baseline == comparison:
        raise ValueError("baseline and comparison system names must differ")
    if cluster_col not in key_cols:
        raise ValueError("key_cols must include the image cluster column")
    _require_columns(frame, (system_col, *key_cols, *value_cols))
    selected = frame[frame[system_col].isin((baseline, comparison))]
    if selected.duplicated([system_col, *key_cols]).any():
        raise ValueError("system/key rows must be unique for a paired summary")
    left = selected[selected[system_col] == baseline][[*key_cols, *value_cols]]
    right = selected[selected[system_col] == comparison][[*key_cols, *value_cols]]
    paired = left.merge(
        right,
        on=list(key_cols),
        how="outer",
        suffixes=("_baseline", "_comparison"),
        indicator=True,
        validate="one_to_one",
    )
    if paired.empty or not bool((paired["_merge"] == "both").all()):
        raise ValueError("paired systems must have identical non-empty key sets")
    paired = paired.drop(columns="_merge")
    selected_draws = _draws_for_frame(paired, cluster_col, draws, repeats, seed)
    output: list[pd.DataFrame] = []
    delta_name = f"{comparison}_minus_{baseline}"
    for metric in value_cols:
        baseline_values = pd.to_numeric(
            paired[f"{metric}_baseline"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        comparison_values = pd.to_numeric(
            paired[f"{metric}_comparison"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        finite = np.isfinite(baseline_values) & np.isfinite(comparison_values)
        working = paired.loc[finite, list(key_cols)].copy()
        if working.empty:
            continue
        working[baseline] = baseline_values[finite]
        working[comparison] = comparison_values[finite]
        working[delta_name] = comparison_values[finite] - baseline_values[finite]
        summary = summarize_clustered_means(
            working,
            value_cols=(baseline, comparison, delta_name),
            draws=selected_draws,
            cluster_col=cluster_col,
            class_col=class_col,
            include_macro_class=include_macro_class,
            identity=identity,
        )
        summary = summary.rename(columns={"metric": "series"})
        summary.insert(0, "metric", metric)
        summary["paired_delta"] = summary["series"] == delta_name
        summary["delta_definition"] = np.where(
            summary["paired_delta"], f"{comparison} - {baseline}", ""
        )
        output.append(summary)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def _ordered_image_arrays(
    image_ids: Sequence[str],
    arrays: Sequence[np.ndarray],
    draws: ImageBootstrapDraws,
) -> list[np.ndarray]:
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("confusion image IDs must be unique")
    lookup = {image_id: index for index, image_id in enumerate(image_ids)}
    if set(lookup) != set(draws.image_ids):
        raise ValueError("confusion image IDs must exactly match supplied draws")
    order = np.asarray(
        [lookup[image_id] for image_id in draws.image_ids], dtype=np.int64
    )
    return [array[order] for array in arrays]


def _validated_confusion_stack(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != (21, 21) or not len(array):
        raise ValueError(f"{name} must have shape [N,21,21]")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must have an integer dtype")
    if np.any(array < 0) or np.any(array.sum(axis=(1, 2)) <= 0):
        raise ValueError(f"{name} must contain non-negative, non-empty confusions")
    return array.astype(np.int64, copy=False)


def _cam_scalar_values(confusion: np.ndarray) -> np.ndarray:
    metrics = cam_metrics_from_confusion(confusion)
    return np.asarray([metrics[name] for name in _CAM_SCALAR_METRICS], dtype=np.float64)


def paired_confusion_metric_summary(
    image_ids: Sequence[str],
    baseline_confusions: np.ndarray,
    comparison_confusions: np.ndarray,
    *,
    baseline_name: str = "baseline",
    comparison_name: str = "comparison",
    draws: ImageBootstrapDraws | None = None,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    identity: Mapping[str, object] | None = None,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Bootstrap nonlinear CAM metrics from paired aggregate confusions."""

    if baseline_name == comparison_name:
        raise ValueError("baseline_name and comparison_name must differ")
    chunk = _positive_integer(chunk_size, "chunk_size")
    clusters = _canonical_image_ids(image_ids)
    selected_draws = draws or image_multinomial_draws(
        clusters, repeats=repeats, seed=seed
    )
    selected_draws = _validate_draws(selected_draws)
    baseline_stack = _validated_confusion_stack(
        baseline_confusions, "baseline_confusions"
    )
    comparison_stack = _validated_confusion_stack(
        comparison_confusions, "comparison_confusions"
    )
    if len(baseline_stack) != len(image_ids) or len(comparison_stack) != len(image_ids):
        raise ValueError("one paired confusion is required for every image ID")
    baseline_stack, comparison_stack = _ordered_image_arrays(
        image_ids,
        (baseline_stack, comparison_stack),
        selected_draws,
    )
    if not np.array_equal(baseline_stack.sum(axis=2), comparison_stack.sum(axis=2)):
        raise ValueError("paired confusions must have identical per-image GT marginals")

    baseline_point = _cam_scalar_values(baseline_stack.sum(axis=0))
    comparison_point = _cam_scalar_values(comparison_stack.sum(axis=0))
    baseline_samples = np.full(
        (selected_draws.repeats, len(_CAM_SCALAR_METRICS)), np.nan
    )
    comparison_samples = np.full_like(baseline_samples, np.nan)
    baseline_flat = baseline_stack.reshape(len(baseline_stack), -1)
    comparison_flat = comparison_stack.reshape(len(comparison_stack), -1)
    for start in range(0, selected_draws.repeats, chunk):
        weights = selected_draws.multiplicities[start : start + chunk].astype(
            np.int64, copy=False
        )
        baseline_aggregates = (weights @ baseline_flat).reshape(-1, 21, 21)
        comparison_aggregates = (weights @ comparison_flat).reshape(-1, 21, 21)
        for offset, (baseline_matrix, comparison_matrix) in enumerate(
            zip(baseline_aggregates, comparison_aggregates)
        ):
            baseline_samples[start + offset] = _cam_scalar_values(baseline_matrix)
            comparison_samples[start + offset] = _cam_scalar_values(comparison_matrix)

    series = {
        baseline_name: (baseline_point, baseline_samples),
        comparison_name: (comparison_point, comparison_samples),
        f"{comparison_name}_minus_{baseline_name}": (
            comparison_point - baseline_point,
            comparison_samples - baseline_samples,
        ),
    }
    rows: list[dict[str, object]] = []
    for series_name, (points, samples) in series.items():
        for index, metric in enumerate(_CAM_SCALAR_METRICS):
            low, high, valid = _finite_interval(samples[:, index])
            record: dict[str, object] = {
                "series": series_name,
                "metric": metric,
                "estimate": float(points[index]),
                "ci_low": low,
                "ci_high": high,
                "num_images": len(selected_draws.image_ids),
                "bootstrap_repeats": selected_draws.repeats,
                "bootstrap_valid_repeats": valid,
                "bootstrap_valid_fraction": valid / selected_draws.repeats,
                "bootstrap_seed": selected_draws.seed,
                "bootstrap_unit": BOOTSTRAP_UNIT,
                "ci_method": "95% percentile",
                "paired_delta": series_name
                == f"{comparison_name}_minus_{baseline_name}",
                "delta_definition": (
                    f"{comparison_name} - {baseline_name}"
                    if series_name == f"{comparison_name}_minus_{baseline_name}"
                    else ""
                ),
                "estimand": "metric of image-weighted aggregate confusion",
            }
            if identity:
                overlap = set(record).intersection(identity)
                if overlap:
                    raise ValueError(
                        f"identity cannot replace statistic fields: {sorted(overlap)}"
                    )
                record.update(identity)
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def _within_map_auc_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    truth = np.asarray(labels)
    values = np.asarray(scores, dtype=np.float64)
    if (
        truth.ndim != 1
        or values.ndim != 1
        or len(truth) != len(values)
        or not len(truth)
    ):
        raise ValueError("binary labels and scores must be equal non-empty vectors")
    if not np.isfinite(values).all():
        raise ValueError("discrimination scores must be finite")
    if not np.all(np.isin(truth, (0, 1, False, True))):
        raise ValueError("discrimination labels must be binary")
    truth = truth.astype(np.uint8, copy=False)
    if truth.min() == truth.max():
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(truth, values)),
        float(average_precision_score(truth, values)),
    )


def paired_clustered_auc_ap_summary(
    image_ids: Sequence[str],
    class_ids: Sequence[int],
    binary_labels: Sequence[int] | np.ndarray,
    baseline_scores: Sequence[float] | np.ndarray,
    comparison_scores: Sequence[float] | np.ndarray,
    *,
    baseline_name: str = "baseline",
    comparison_name: str = "comparison",
    draws: ImageBootstrapDraws | None = None,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    identity: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Summarize paired within-image-class AUROC/AP with image bootstrap.

    Input rows are patches retained for one binary region contrast. AUROC/AP is
    first computed within each image-class map. Whole images are then sampled;
    all maps and all patch rows belonging to one image therefore move together.
    The reported micro estimand weights finite image-class maps equally, while
    ``macro_class`` gives each semantic class equal weight.
    """

    if baseline_name == comparison_name:
        raise ValueError("baseline_name and comparison_name must differ")
    lengths = {
        len(image_ids),
        len(class_ids),
        len(binary_labels),
        len(baseline_scores),
        len(comparison_scores),
    }
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("all discrimination inputs must have equal non-zero length")
    ids = list(image_ids)
    if any(not isinstance(value, str) or not value for value in ids):
        raise TypeError("every image ID must be a non-empty string")
    classes = np.asarray(class_ids)
    if not np.issubdtype(classes.dtype, np.integer):
        raise TypeError("class_ids must have an integer dtype")
    if np.any((classes < 0) | (classes >= 20)):
        raise ValueError("class_ids must lie in the zero-based VOC range 0..19")
    labels = np.asarray(binary_labels)
    baseline_values = np.asarray(baseline_scores, dtype=np.float64)
    comparison_values = np.asarray(comparison_scores, dtype=np.float64)
    if (
        not np.isfinite(baseline_values).all()
        or not np.isfinite(comparison_values).all()
    ):
        raise ValueError("paired discrimination scores must be finite")
    if not np.all(np.isin(labels, (0, 1, False, True))):
        raise ValueError("binary_labels must contain only 0/1")

    patches = pd.DataFrame(
        {
            "image_id": ids,
            "class_id": classes.astype(np.int64, copy=False),
            "label": labels.astype(np.uint8, copy=False),
            "baseline_score": baseline_values,
            "comparison_score": comparison_values,
        }
    )
    units: list[dict[str, object]] = []
    for (image_id, class_id), group in patches.groupby(
        ["image_id", "class_id"], sort=True
    ):
        baseline_auc, baseline_ap = _within_map_auc_ap(
            group["label"].to_numpy(), group["baseline_score"].to_numpy()
        )
        comparison_auc, comparison_ap = _within_map_auc_ap(
            group["label"].to_numpy(), group["comparison_score"].to_numpy()
        )
        units.append(
            {
                "image_id": image_id,
                "class_id": int(class_id),
                "baseline_auroc": baseline_auc,
                "comparison_auroc": comparison_auc,
                "delta_auroc": comparison_auc - baseline_auc,
                "baseline_average_precision": baseline_ap,
                "comparison_average_precision": comparison_ap,
                "delta_average_precision": comparison_ap - baseline_ap,
            }
        )
    unit_frame = pd.DataFrame.from_records(units)
    selected_draws = _draws_for_frame(unit_frame, "image_id", draws, repeats, seed)
    columns = (
        "baseline_auroc",
        "comparison_auroc",
        "delta_auroc",
        "baseline_average_precision",
        "comparison_average_precision",
        "delta_average_precision",
    )
    summary = summarize_clustered_means(
        unit_frame,
        value_cols=columns,
        draws=selected_draws,
        include_macro_class=True,
        identity=identity,
    )
    decoded = {
        "baseline_auroc": ("auroc", baseline_name, False),
        "comparison_auroc": ("auroc", comparison_name, False),
        "delta_auroc": (
            "auroc",
            f"{comparison_name}_minus_{baseline_name}",
            True,
        ),
        "baseline_average_precision": ("average_precision", baseline_name, False),
        "comparison_average_precision": (
            "average_precision",
            comparison_name,
            False,
        ),
        "delta_average_precision": (
            "average_precision",
            f"{comparison_name}_minus_{baseline_name}",
            True,
        ),
    }
    summary[["metric", "series", "paired_delta"]] = pd.DataFrame(
        [decoded[value] for value in summary["metric"]], index=summary.index
    )
    summary["delta_definition"] = np.where(
        summary["paired_delta"], f"{comparison_name} - {baseline_name}", ""
    )
    summary["num_patch_rows"] = len(patches)
    summary["discrimination_unit"] = "within-image-class patch map"
    return summary
