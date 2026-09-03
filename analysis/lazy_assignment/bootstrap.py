"""Image-clustered bootstrap helpers for Experiment 1 analysis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    n_clusters: int
    n_rows: int


def derived_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256(
        "|".join([str(base_seed), *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def draw_cluster_counts(
    num_clusters: int, repeats: int, seed: int
) -> np.ndarray:
    if num_clusters < 1 or repeats < 1:
        raise ValueError("num_clusters and repeats must be positive")
    rng = np.random.default_rng(seed)
    probabilities = np.full(num_clusters, 1.0 / num_clusters, dtype=np.float64)
    return rng.multinomial(num_clusters, probabilities, size=repeats)


def _cluster_arrays(
    frame: pd.DataFrame, cluster_col: str, value_cols: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.empty:
        raise ValueError("bootstrap frame is empty")
    clusters = np.asarray(sorted(frame[cluster_col].astype(str).unique()))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    sums = np.zeros((len(clusters), len(value_cols)), dtype=np.float64)
    counts = np.zeros_like(sums)
    for row in frame[[cluster_col, *value_cols]].itertuples(index=False, name=None):
        index = cluster_index[str(row[0])]
        values = np.asarray(row[1:], dtype=np.float64)
        valid = np.isfinite(values)
        sums[index, valid] += values[valid]
        counts[index, valid] += 1.0
    return clusters, sums, counts


def cluster_bootstrap_means(
    frame: pd.DataFrame,
    cluster_col: str,
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
    chunk_size: int = 500,
) -> dict[str, BootstrapEstimate]:
    """Bootstrap row-weighted means while resampling complete image clusters."""

    clusters, sums, valid_counts = _cluster_arrays(frame, cluster_col, value_cols)
    point_denominator = valid_counts.sum(axis=0)
    if np.any(point_denominator == 0):
        missing = [col for col, count in zip(value_cols, point_denominator) if count == 0]
        raise ValueError(f"no finite observations for {missing}")
    point = sums.sum(axis=0) / point_denominator
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, len(value_cols)), dtype=np.float64)
    probabilities = np.full(len(clusters), 1.0 / len(clusters), dtype=np.float64)
    offset = 0
    while offset < repeats:
        size = min(chunk_size, repeats - offset)
        draws = rng.multinomial(len(clusters), probabilities, size=size)
        numerator = draws @ sums
        denominator = draws @ valid_counts
        np.divide(
            numerator,
            denominator,
            out=boot[offset : offset + size],
            where=denominator > 0,
        )
        boot[offset : offset + size][denominator == 0] = np.nan
        offset += size
    lower = np.nanquantile(boot, 0.025, axis=0)
    upper = np.nanquantile(boot, 0.975, axis=0)
    return {
        column: BootstrapEstimate(
            estimate=float(point[index]),
            ci_low=float(lower[index]),
            ci_high=float(upper[index]),
            n_clusters=len(clusters),
            n_rows=len(frame),
        )
        for index, column in enumerate(value_cols)
    }


def cluster_bootstrap_macro_class_means(
    frame: pd.DataFrame,
    cluster_col: str,
    class_col: str,
    value_cols: Sequence[str],
    repeats: int,
    seed: int,
    chunk_size: int = 250,
) -> dict[str, BootstrapEstimate]:
    """Bootstrap equal-class means while retaining all rows from sampled images."""

    if frame.empty:
        raise ValueError("bootstrap frame is empty")
    clusters = np.asarray(sorted(frame[cluster_col].astype(str).unique()))
    classes = np.asarray(sorted(frame[class_col].astype(int).unique()))
    cluster_index = {value: index for index, value in enumerate(clusters)}
    class_index = {int(value): index for index, value in enumerate(classes)}
    class_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    point_class = np.full((len(classes), len(value_cols)), np.nan, dtype=np.float64)
    for class_value in classes:
        subset = frame[frame[class_col].astype(int) == int(class_value)]
        image_indices = np.asarray(
            [cluster_index[str(value)] for value in subset[cluster_col]], dtype=np.int64
        )
        values = subset[list(value_cols)].to_numpy(dtype=np.float64)
        valid = np.isfinite(values).astype(np.float64)
        clean_values = np.where(np.isfinite(values), values, 0.0)
        category_index = class_index[int(class_value)]
        denominator = valid.sum(axis=0)
        point_class[category_index] = np.divide(
            clean_values.sum(axis=0),
            denominator,
            out=np.full(len(value_cols), np.nan, dtype=np.float64),
            where=denominator > 0,
        )
        class_arrays.append((image_indices, clean_values, valid))
    point = np.nanmean(point_class, axis=0)

    probabilities = np.full(len(clusters), 1.0 / len(clusters), dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty((repeats, len(value_cols)), dtype=np.float64)
    offset = 0
    while offset < repeats:
        size = min(chunk_size, repeats - offset)
        draws = rng.multinomial(len(clusters), probabilities, size=size)
        class_means = np.full(
            (size, len(classes), len(value_cols)), np.nan, dtype=np.float64
        )
        for category_index, (image_indices, values, valid) in enumerate(class_arrays):
            selected = draws[:, image_indices]
            numerator = selected @ values
            denominator = selected @ valid
            class_means[:, category_index, :] = np.divide(
                numerator,
                denominator,
                out=np.full_like(numerator, np.nan),
                where=denominator > 0,
            )
        boot[offset : offset + size] = np.nanmean(class_means, axis=1)
        offset += size
    lower = np.nanquantile(boot, 0.025, axis=0)
    upper = np.nanquantile(boot, 0.975, axis=0)
    return {
        column: BootstrapEstimate(
            estimate=float(point[index]),
            ci_low=float(lower[index]),
            ci_high=float(upper[index]),
            n_clusters=len(clusters),
            n_rows=len(frame),
        )
        for index, column in enumerate(value_cols)
    }


def cluster_standardized_effect(
    frame: pd.DataFrame, cluster_col: str, delta_col: str, epsilon: float = 1e-12
) -> float:
    image_delta = frame.groupby(cluster_col, sort=True)[delta_col].mean()
    if len(image_delta) < 2:
        return float("nan")
    return float(image_delta.mean() / (image_delta.std(ddof=1) + epsilon))
