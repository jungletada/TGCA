"""Image-clustered and paired bootstrap summaries for Experiment 2."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from analysis.lazy_assignment.bootstrap import derived_seed


@dataclass(frozen=True)
class Experiment2BootstrapEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    n_finite_clusters: int
    n_total_clusters: int
    n_finite_rows: int
    n_total_rows: int
    n_finite_classes: int | None
    valid_repeats: int


# The same image draw must be used across signals/models within an analysis
# family.  Reusing it is both the statistically natural paired design and the
# difference between a tractable 5,000-repeat full run and generating thousands
# of redundant 5,000 x 1,449 multinomial matrices.  Four entries cover the
# all/single/2-label/3+-label cycle while bounding memory (~60 MiB at VOC size).
_DRAW_CACHE: OrderedDict[tuple[tuple[str, ...], int, int], np.ndarray] = OrderedDict()
_MAX_DRAW_CACHE_ENTRIES = 4


def _cluster_arrays(
    frame: pd.DataFrame,
    cluster_col: str,
    value_cols: Sequence[str],
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    clusters = tuple(sorted(frame[cluster_col].astype(str).unique()))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    sums = np.zeros((len(clusters), len(value_cols)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for row in frame[[cluster_col, *value_cols]].itertuples(index=False, name=None):
        index = cluster_index[str(row[0])]
        values = np.asarray(row[1:], dtype=np.float64)
        finite = np.isfinite(values)
        sums[index, finite] += values[finite]
        counts[index, finite] += 1.0
    return clusters, sums, counts


def _cluster_draws(clusters: tuple[str, ...], repeats: int, seed: int) -> np.ndarray:
    key = (clusters, int(repeats), int(seed))
    cached = _DRAW_CACHE.pop(key, None)
    if cached is not None:
        _DRAW_CACHE[key] = cached
        return cached
    rng = np.random.default_rng(seed)
    probabilities = np.full(len(clusters), 1.0 / len(clusters), dtype=np.float64)
    draws = rng.multinomial(len(clusters), probabilities, size=repeats)
    dtype = np.uint16 if len(clusters) <= np.iinfo(np.uint16).max else np.uint32
    draws = draws.astype(dtype, copy=False)
    draws.setflags(write=False)
    _DRAW_CACHE[key] = draws
    while len(_DRAW_CACHE) > _MAX_DRAW_CACHE_ENTRIES:
        _DRAW_CACHE.popitem(last=False)
    return draws


def _finite_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan"), 0
    return (
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
        int(len(finite)),
    )


def _finite_intervals(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized column-wise finite percentile intervals.

    This is numerically equivalent to calling ``_finite_interval`` for every
    metric column, but avoids millions of Python-level ``np.quantile`` calls in
    the full class-wise analysis. Each column still retains its own finite
    bootstrap denominator and percentile calculation.
    """

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("finite interval matrix must have shape [repeat, metric]")
    valid_counts = np.isfinite(array).sum(axis=0).astype(np.int64, copy=False)
    low = np.full(array.shape[1], np.nan, dtype=np.float64)
    high = np.full(array.shape[1], np.nan, dtype=np.float64)
    usable = valid_counts > 0
    if bool(usable.any()):
        # All selected columns contain at least one finite value. nanquantile
        # therefore emits no all-NaN warning and exactly matches the scalar
        # finite-value quantile definition above.
        selected = array[:, usable]
        selected[~np.isfinite(selected)] = np.nan
        quantiles = np.nanquantile(selected, (0.025, 0.975), axis=0)
        low[usable] = quantiles[0]
        high[usable] = quantiles[1]
    return low, high, valid_counts


def _micro_estimates(
    sums: np.ndarray,
    counts: np.ndarray,
    draws: np.ndarray,
    value_cols: Sequence[str],
    *,
    total_rows: int,
    chunk_size: int = 250,
) -> dict[str, Experiment2BootstrapEstimate]:
    point_denominator = counts.sum(axis=0)
    point = np.divide(
        sums.sum(axis=0),
        point_denominator,
        out=np.full(len(value_cols), np.nan, dtype=np.float64),
        where=point_denominator > 0,
    )
    boot = np.full((len(draws), len(value_cols)), np.nan, dtype=np.float64)
    for offset in range(0, len(draws), chunk_size):
        draw = draws[offset : offset + chunk_size].astype(np.float64, copy=False)
        numerator = draw @ sums
        denominator = draw @ counts
        np.divide(
            numerator,
            denominator,
            out=boot[offset : offset + len(draw)],
            where=denominator > 0,
        )
    lows, highs, valid_repeat_counts = _finite_intervals(boot)
    estimates: dict[str, Experiment2BootstrapEstimate] = {}
    for index, column in enumerate(value_cols):
        estimates[column] = Experiment2BootstrapEstimate(
            estimate=float(point[index]),
            ci_low=float(lows[index]),
            ci_high=float(highs[index]),
            n_finite_clusters=int(np.count_nonzero(counts[:, index] > 0)),
            n_total_clusters=int(len(counts)),
            n_finite_rows=int(point_denominator[index]),
            n_total_rows=int(total_rows),
            n_finite_classes=None,
            valid_repeats=int(valid_repeat_counts[index]),
        )
    return estimates


def _macro_class_estimates(
    frame: pd.DataFrame,
    cluster_col: str,
    class_col: str,
    clusters: tuple[str, ...],
    draws: np.ndarray,
    value_cols: Sequence[str],
    *,
    chunk_size: int = 250,
) -> dict[str, Experiment2BootstrapEstimate]:
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    classes = tuple(sorted(frame[class_col].astype(int).unique()))
    class_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    point_class = np.full((len(classes), len(value_cols)), np.nan, dtype=np.float64)
    metric_cluster_presence = np.zeros((len(clusters), len(value_cols)), dtype=bool)
    metric_row_counts = np.zeros(len(value_cols), dtype=np.int64)
    for class_index, class_id in enumerate(classes):
        subset = frame[frame[class_col].astype(int) == class_id]
        image_indices = np.asarray(
            [cluster_index[str(value)] for value in subset[cluster_col]], dtype=np.int64
        )
        values = subset[list(value_cols)].to_numpy(dtype=np.float64)
        valid = np.isfinite(values).astype(np.float64)
        clean = np.where(np.isfinite(values), values, 0.0)
        denominator = valid.sum(axis=0)
        point_class[class_index] = np.divide(
            clean.sum(axis=0),
            denominator,
            out=np.full(len(value_cols), np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        for local_index, image_index in enumerate(image_indices):
            metric_cluster_presence[image_index] |= valid[local_index].astype(bool)
        metric_row_counts += valid.sum(axis=0).astype(np.int64)
        class_arrays.append((image_indices, clean, valid))
    finite_point_classes = np.isfinite(point_class)
    point_denominator = finite_point_classes.sum(axis=0)
    point = np.divide(
        np.nansum(point_class, axis=0),
        point_denominator,
        out=np.full(len(value_cols), np.nan, dtype=np.float64),
        where=point_denominator > 0,
    )
    boot = np.full((len(draws), len(value_cols)), np.nan, dtype=np.float64)
    for offset in range(0, len(draws), chunk_size):
        draw = draws[offset : offset + chunk_size].astype(np.float64, copy=False)
        class_means = np.full(
            (len(draw), len(classes), len(value_cols)), np.nan, dtype=np.float64
        )
        for class_index, (image_indices, values, valid) in enumerate(class_arrays):
            selected = draw[:, image_indices]
            numerator = selected @ values
            denominator = selected @ valid
            np.divide(
                numerator,
                denominator,
                out=class_means[:, class_index, :],
                where=denominator > 0,
            )
        finite_classes = np.isfinite(class_means)
        denominator = finite_classes.sum(axis=1)
        np.divide(
            np.nansum(class_means, axis=1),
            denominator,
            out=boot[offset : offset + len(draw)],
            where=denominator > 0,
        )
    lows, highs, valid_repeat_counts = _finite_intervals(boot)
    estimates: dict[str, Experiment2BootstrapEstimate] = {}
    for index, column in enumerate(value_cols):
        estimates[column] = Experiment2BootstrapEstimate(
            estimate=float(point[index]),
            ci_low=float(lows[index]),
            ci_high=float(highs[index]),
            n_finite_clusters=int(metric_cluster_presence[:, index].sum()),
            n_total_clusters=int(len(clusters)),
            n_finite_rows=int(metric_row_counts[index]),
            n_total_rows=int(len(frame)),
            n_finite_classes=int(point_denominator[index]),
            valid_repeats=int(valid_repeat_counts[index]),
        )
    return estimates


def _estimate_rows(
    estimates: dict[str, Experiment2BootstrapEstimate],
    identity: dict[str, object],
    aggregation: str,
    repeats: int,
    base_seed: int,
    draw_seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metric, estimate in estimates.items():
        rows.append(
            {
                **identity,
                "aggregation": aggregation,
                "metric": metric,
                "estimate": estimate.estimate,
                "ci_low": estimate.ci_low,
                "ci_high": estimate.ci_high,
                "num_images": estimate.n_finite_clusters,
                "num_images_total": estimate.n_total_clusters,
                "num_rows": estimate.n_finite_rows,
                "num_rows_total": estimate.n_total_rows,
                "num_classes": estimate.n_finite_classes,
                "bootstrap_repeats": int(repeats),
                "bootstrap_valid_repeats": estimate.valid_repeats,
                "bootstrap_valid_fraction": estimate.valid_repeats / repeats,
                "bootstrap_seed": int(draw_seed),
                "bootstrap_base_seed": int(base_seed),
            }
        )
    return rows


def summarize_clustered(
    frame: pd.DataFrame,
    *,
    value_cols: Sequence[str],
    identity: dict[str, object],
    repeats: int,
    seed: int,
    cluster_col: str = "image_id",
    class_col: str = "class_id",
    include_macro_class: bool = True,
) -> list[dict[str, object]]:
    """Return micro and optional equal-class mean estimates with image clusters."""

    if frame.empty:
        return []
    usable = [
        column
        for column in value_cols
        if column in frame and np.isfinite(frame[column].to_numpy(dtype=float)).any()
    ]
    if not usable:
        return []
    clusters, sums, counts = _cluster_arrays(frame, cluster_col, usable)
    draw_seed = derived_seed(seed, "image_cluster_draws")
    draws = _cluster_draws(clusters, repeats, draw_seed)
    micro = _micro_estimates(sums, counts, draws, usable, total_rows=len(frame))
    rows = _estimate_rows(micro, identity, "micro", repeats, seed, draw_seed)
    if include_macro_class and class_col in frame and frame[class_col].nunique() > 0:
        macro = _macro_class_estimates(
            frame, cluster_col, class_col, clusters, draws, usable
        )
        rows.extend(
            _estimate_rows(macro, identity, "macro_class", repeats, seed, draw_seed)
        )
    return rows


def summarize_image_mean_correlations(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_cols: Sequence[str],
    identity: dict[str, object],
    repeats: int,
    seed: int,
    cluster_col: str = "image_id",
) -> list[dict[str, object]]:
    """Bootstrap Pearson associations between per-image mean quantities.

    A multi-label image may contribute several class pairs, so the pair rows
    are first collapsed to one mean x/y observation per image.  The subsequent
    ordinary cluster bootstrap is therefore exactly an image-level bootstrap,
    not a class-pair bootstrap masquerading as independent observations.
    """

    requested = [column for column in y_cols if column in frame]
    if frame.empty or x_col not in frame or not requested:
        return []
    image_means = frame.groupby(cluster_col, sort=True)[[x_col, *requested]].mean()
    clusters = tuple(image_means.index.astype(str))
    if len(clusters) < 2:
        return []
    draw_seed = derived_seed(seed, "image_cluster_draws")
    draws = _cluster_draws(clusters, repeats, draw_seed)
    rows: list[dict[str, object]] = []
    for y_col in requested:
        x = image_means[x_col].to_numpy(dtype=np.float64)
        y = image_means[y_col].to_numpy(dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 2 or np.ptp(x[finite]) <= 1e-12 or np.ptp(y[finite]) <= 1e-12:
            continue
        clean_x = np.where(finite, x, 0.0)
        clean_y = np.where(finite, y, 0.0)
        valid = finite.astype(np.float64)
        point = float(np.corrcoef(x[finite], y[finite])[0, 1])
        boot = np.full(repeats, np.nan, dtype=np.float64)
        for offset in range(0, repeats, 500):
            draw = draws[offset : offset + 500].astype(np.float64, copy=False)
            weight = draw * valid[None, :]
            count = weight.sum(axis=1)
            sum_x = weight @ clean_x
            sum_y = weight @ clean_y
            sum_x2 = weight @ np.square(clean_x)
            sum_y2 = weight @ np.square(clean_y)
            sum_xy = weight @ (clean_x * clean_y)
            covariance = sum_xy - np.divide(
                sum_x * sum_y,
                count,
                out=np.zeros_like(count),
                where=count > 0,
            )
            variance_x = sum_x2 - np.divide(
                np.square(sum_x),
                count,
                out=np.zeros_like(count),
                where=count > 0,
            )
            variance_y = sum_y2 - np.divide(
                np.square(sum_y),
                count,
                out=np.zeros_like(count),
                where=count > 0,
            )
            denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
            np.divide(
                covariance,
                denominator,
                out=boot[offset : offset + len(draw)],
                where=(count >= 2) & (denominator > 1e-12),
            )
        low, high, valid_repeats = _finite_interval(boot)
        rows.append(
            {
                **identity,
                "aggregation": "image_mean_pearson",
                "metric": f"pearson_{x_col}_vs_{y_col}",
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "num_images": int(finite.sum()),
                "num_images_total": int(len(clusters)),
                "num_rows": int(finite.sum()),
                "num_rows_total": int(len(clusters)),
                "num_classes": None,
                "bootstrap_repeats": int(repeats),
                "bootstrap_valid_repeats": valid_repeats,
                "bootstrap_valid_fraction": valid_repeats / repeats,
                "bootstrap_seed": int(draw_seed),
                "bootstrap_base_seed": int(seed),
                "association_unit": "per-image mean over positive class pairs",
            }
        )
    return rows


def summarize_clustered_pair_correlations(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_cols: Sequence[str],
    identity: dict[str, object],
    repeats: int,
    seed: int,
    cluster_col: str = "image_id",
) -> list[dict[str, object]]:
    """Pearson associations over pair rows with image-clustered resampling.

    The point estimand weights each finite class-pair row equally. Bootstrap
    multiplicities are sampled only at image level and then applied to every
    pair row belonging to that image. This complements, rather than replaces,
    the per-image-mean association sensitivity analysis above.
    """

    requested = [column for column in y_cols if column in frame]
    if frame.empty or x_col not in frame or not requested:
        return []
    clusters = tuple(sorted(frame[cluster_col].astype(str).unique()))
    if len(clusters) < 2:
        return []
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    row_clusters = np.asarray(
        [cluster_index[str(value)] for value in frame[cluster_col]], dtype=np.int64
    )
    x_all = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=np.float64)
    draw_seed = derived_seed(seed, "image_cluster_draws")
    draws = _cluster_draws(clusters, repeats, draw_seed)
    rows: list[dict[str, object]] = []
    for y_col in requested:
        y_all = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(x_all) & np.isfinite(y_all)
        if finite.sum() < 2:
            continue
        x = x_all[finite]
        y = y_all[finite]
        if np.ptp(x) <= 1e-12 or np.ptp(y) <= 1e-12:
            continue
        indices = row_clusters[finite]
        sufficient = np.zeros((len(clusters), 6), dtype=np.float64)
        np.add.at(sufficient[:, 0], indices, 1.0)
        np.add.at(sufficient[:, 1], indices, x)
        np.add.at(sufficient[:, 2], indices, y)
        np.add.at(sufficient[:, 3], indices, np.square(x))
        np.add.at(sufficient[:, 4], indices, np.square(y))
        np.add.at(sufficient[:, 5], indices, x * y)
        point = float(np.corrcoef(x, y)[0, 1])
        boot = np.full(repeats, np.nan, dtype=np.float64)
        for offset in range(0, repeats, 500):
            draw = draws[offset : offset + 500].astype(np.float64, copy=False)
            stats = draw @ sufficient
            count, sx, sy, sx2, sy2, sxy = stats.T
            covariance = sxy - np.divide(
                sx * sy, count, out=np.zeros_like(count), where=count > 0
            )
            variance_x = sx2 - np.divide(
                np.square(sx), count, out=np.zeros_like(count), where=count > 0
            )
            variance_y = sy2 - np.divide(
                np.square(sy), count, out=np.zeros_like(count), where=count > 0
            )
            denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
            np.divide(
                covariance,
                denominator,
                out=boot[offset : offset + len(draw)],
                where=(count >= 2) & (denominator > 1e-12),
            )
        low, high, valid_repeats = _finite_interval(boot)
        finite_clusters = int(np.unique(indices).size)
        rows.append(
            {
                **identity,
                "aggregation": "micro_pair_pearson",
                "metric": f"pearson_{x_col}_vs_{y_col}",
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "num_images": finite_clusters,
                "num_images_total": int(len(clusters)),
                "num_rows": int(finite.sum()),
                "num_rows_total": int(len(frame)),
                "num_classes": None,
                "bootstrap_repeats": int(repeats),
                "bootstrap_valid_repeats": valid_repeats,
                "bootstrap_valid_fraction": valid_repeats / repeats,
                "bootstrap_seed": int(draw_seed),
                "bootstrap_base_seed": int(seed),
                "association_unit": (
                    "positive class-pair rows with image-cluster resampling"
                ),
            }
        )
    return rows


def summarize_clustered_macro_class_correlations(
    frame: pd.DataFrame,
    *,
    x_col: str,
    y_cols: Sequence[str],
    identity: dict[str, object],
    repeats: int,
    seed: int,
    cluster_col: str = "image_id",
    class_col: str = "class_id",
) -> list[dict[str, object]]:
    """Equal-class mean of within-focal-class Pearson correlations.

    Every bootstrap replicate samples whole images.  Within each sampled
    replicate, a Pearson correlation is recomputed separately for each focal
    class; finite class correlations are then averaged with equal class
    weight.  Degenerate classes are excluded per metric/per replicate rather
    than assigned a zero correlation.
    """

    requested = [column for column in y_cols if column in frame]
    required = {x_col, cluster_col, class_col}
    if frame.empty or not required.issubset(frame.columns) or not requested:
        return []
    clusters = tuple(sorted(frame[cluster_col].astype(str).unique()))
    if len(clusters) < 2:
        return []
    try:
        class_values = pd.to_numeric(frame[class_col], errors="raise").to_numpy(
            dtype=np.int64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{class_col} must contain integer class IDs") from error
    classes = tuple(sorted(np.unique(class_values).tolist()))
    class_index = {class_id: index for index, class_id in enumerate(classes)}
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    row_classes = np.asarray(
        [class_index[int(value)] for value in class_values], dtype=np.int64
    )
    row_clusters = np.asarray(
        [cluster_index[str(value)] for value in frame[cluster_col]], dtype=np.int64
    )
    x_all = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(dtype=np.float64)
    draw_seed = derived_seed(seed, "image_cluster_draws")
    draws = _cluster_draws(clusters, repeats, draw_seed)

    def correlations(stats: np.ndarray) -> np.ndarray:
        count = stats[..., 0]
        sum_x = stats[..., 1]
        sum_y = stats[..., 2]
        covariance = stats[..., 5] - np.divide(
            sum_x * sum_y,
            count,
            out=np.zeros_like(count),
            where=count > 0,
        )
        variance_x = stats[..., 3] - np.divide(
            np.square(sum_x),
            count,
            out=np.zeros_like(count),
            where=count > 0,
        )
        variance_y = stats[..., 4] - np.divide(
            np.square(sum_y),
            count,
            out=np.zeros_like(count),
            where=count > 0,
        )
        denominator = np.sqrt(np.maximum(variance_x * variance_y, 0.0))
        result = np.full_like(denominator, np.nan, dtype=np.float64)
        np.divide(
            covariance,
            denominator,
            out=result,
            where=(count >= 2) & (denominator > 1e-12),
        )
        return result

    rows: list[dict[str, object]] = []
    for y_col in requested:
        y_all = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(x_all) & np.isfinite(y_all)
        if finite.sum() < 2:
            continue
        sufficient = np.zeros((len(classes), len(clusters), 6), dtype=np.float64)
        class_indices = row_classes[finite]
        cluster_indices = row_clusters[finite]
        x = x_all[finite]
        y = y_all[finite]
        for statistic, values in enumerate(
            (
                np.ones(len(x), dtype=np.float64),
                x,
                y,
                np.square(x),
                np.square(y),
                x * y,
            )
        ):
            np.add.at(
                sufficient[:, :, statistic],
                (class_indices, cluster_indices),
                values,
            )
        point_class = correlations(sufficient.sum(axis=1))
        finite_point_classes = np.isfinite(point_class)
        if not bool(finite_point_classes.any()):
            continue
        point = float(point_class[finite_point_classes].mean())
        boot = np.full(repeats, np.nan, dtype=np.float64)
        for offset in range(0, repeats, 250):
            draw = draws[offset : offset + 250].astype(np.float64, copy=False)
            stats = np.einsum("ri,cij->rcj", draw, sufficient, optimize=True)
            class_correlations = correlations(stats)
            finite_classes = np.isfinite(class_correlations)
            denominator = finite_classes.sum(axis=1)
            np.divide(
                np.nansum(class_correlations, axis=1),
                denominator,
                out=boot[offset : offset + len(draw)],
                where=denominator > 0,
            )
        low, high, valid_repeats = _finite_interval(boot)
        rows.append(
            {
                **identity,
                "aggregation": "macro_class_pearson",
                "metric": f"pearson_{x_col}_vs_{y_col}",
                "estimate": point,
                "ci_low": low,
                "ci_high": high,
                "num_images": int(np.unique(row_clusters[finite]).size),
                "num_images_total": int(len(clusters)),
                "num_rows": int(finite.sum()),
                "num_rows_total": int(len(frame)),
                "num_classes": int(finite_point_classes.sum()),
                "num_classes_total": int(len(classes)),
                "bootstrap_repeats": int(repeats),
                "bootstrap_valid_repeats": valid_repeats,
                "bootstrap_valid_fraction": valid_repeats / repeats,
                "bootstrap_seed": int(draw_seed),
                "bootstrap_base_seed": int(seed),
                "association_unit": (
                    "equal-focal-class mean of within-class positive-pair "
                    "Pearson correlations with image-cluster resampling"
                ),
            }
        )
    return rows


def paired_model_frame(
    frame: pd.DataFrame,
    *,
    key_cols: Sequence[str],
    value_cols: Sequence[str],
    model_col: str = "model",
    baseline: str = "mctformer",
    comparison: str = "mctformer_plus",
) -> pd.DataFrame:
    """Build exact common-key deltas, defined as comparison minus baseline."""

    required = set(key_cols) | set(value_cols) | {model_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing paired-frame columns: {sorted(missing)}")
    if frame.duplicated([model_col, *key_cols]).any():
        raise ValueError("model/key rows must be unique before paired comparison")
    left = frame[frame[model_col] == baseline][[*key_cols, *value_cols]].copy()
    right = frame[frame[model_col] == comparison][[*key_cols, *value_cols]].copy()
    paired = left.merge(
        right,
        on=list(key_cols),
        how="inner",
        suffixes=("_baseline", "_comparison"),
        validate="one_to_one",
    )
    for column in value_cols:
        # Metrics such as target_hit are intentionally stored as booleans in
        # the canonical tables.  Paired effects are arithmetic deltas, so make
        # the conversion explicit instead of relying on pandas dtype coercion.
        comparison_values = pd.to_numeric(
            paired[f"{column}_comparison"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        baseline_values = pd.to_numeric(
            paired[f"{column}_baseline"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        paired[column] = comparison_values - baseline_values
    paired["baseline_model"] = baseline
    paired["comparison_model"] = comparison
    return paired


def add_label_stratum(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "num_positive_classes" not in result:
        raise ValueError("num_positive_classes is required for label strata")
    count = result["num_positive_classes"].to_numpy(dtype=int)
    result["label_stratum"] = np.select(
        [count == 1, count == 2, count >= 3],
        ["single_label", "exactly_2_labels", "3plus_labels"],
        default="invalid",
    )
    return result


def iter_all_and_label_strata(
    frame: pd.DataFrame,
) -> Iterable[tuple[str, pd.DataFrame]]:
    enriched = add_label_stratum(frame)
    yield "all", enriched
    for name in ("single_label", "exactly_2_labels", "3plus_labels"):
        yield name, enriched[enriched["label_stratum"] == name]
