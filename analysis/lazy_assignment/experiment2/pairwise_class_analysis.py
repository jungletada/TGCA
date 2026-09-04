"""Endpoint-aware helpers for Experiment 2 unordered class-pair tables.

The canonical pair tables store each positive class pair once, with
``class_a < class_b``.  That representation is appropriate for symmetric
pair-level quantities, but it cannot be macro-averaged by class without first
giving both endpoints equal, explicit representation.  This module performs
that deterministic reshaping only; statistical aggregation and bootstrap
resampling deliberately live elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


# These columns jointly identify rows in the three canonical unordered-pair
# tables.  The default validator uses every listed column that is present, so
# it works for shared ownership, pair-map diversity, and class-token pairs.
DEFAULT_PAIR_IDENTITY_COLUMNS = (
    "model",
    "image_id",
    "class_a",
    "class_b",
    "layer",
    "layer_or_stage",
    "stage",
    "signal",
    "rho",
    "topk_ratio",
    "transition",
    "source_signal",
    "destination_signal",
    "previous_layer_or_stage",
    "new_shared_transition",
)

_REQUIRED_PAIR_COLUMNS = ("image_id", "class_a", "class_b")
_ENDPOINT_OUTPUT_COLUMNS = ("focal_endpoint", "class_id", "partner_class_id")
_ENDPOINT_METADATA = (
    ("class_a_name", "class_b_name", "class_name", "partner_class_name"),
    ("class_a_offset", "class_b_offset", "class_offset", "partner_class_offset"),
    (
        "classification_status_a",
        "classification_status_b",
        "classification_status",
        "partner_classification_status",
    ),
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"pair table misses required columns: {missing}")


def _validated_class_ids(frame: pd.DataFrame, column: str) -> np.ndarray:
    try:
        numeric = pd.to_numeric(frame[column], errors="raise").to_numpy(
            dtype=np.float64
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must contain integer class IDs") from error
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{column} must contain finite integer class IDs")
    return numeric.astype(np.int64)


def _identity_columns(
    frame: pd.DataFrame, identity_columns: Sequence[str] | None
) -> tuple[str, ...]:
    if identity_columns is None:
        resolved = tuple(
            column for column in DEFAULT_PAIR_IDENTITY_COLUMNS if column in frame
        )
    else:
        resolved = tuple(identity_columns)
        if len(set(resolved)) != len(resolved):
            raise ValueError("identity_columns contains duplicates")
    required = set(_REQUIRED_PAIR_COLUMNS)
    if not required.issubset(resolved):
        raise ValueError(
            "pair identity must include image_id, class_a, and class_b; got "
            f"{list(resolved)}"
        )
    _require_columns(frame, resolved)
    return resolved


def validate_unordered_pairs(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate canonical ordering and uniqueness, returning identity columns.

    ``class_a < class_b`` is required rather than silently canonicalized.  A
    silent swap would be unsafe for ownership columns whose ``target_a`` and
    ``target_b`` meanings must move with their corresponding class IDs.
    """

    _require_columns(frame, _REQUIRED_PAIR_COLUMNS)
    identities = _identity_columns(frame, identity_columns)
    class_a = _validated_class_ids(frame, "class_a")
    class_b = _validated_class_ids(frame, "class_b")
    invalid = np.flatnonzero(class_a >= class_b)
    if invalid.size:
        index = frame.index[int(invalid[0])]
        raise ValueError(
            "unordered pair rows must satisfy class_a < class_b; "
            f"row index {index!r} has ({class_a[invalid[0]]}, {class_b[invalid[0]]})"
        )
    duplicate = frame.duplicated(list(identities), keep=False)
    if bool(duplicate.any()):
        sample = frame.loc[duplicate, list(identities)].iloc[0].to_dict()
        raise ValueError(f"duplicate unordered pair identity: {sample}")
    return identities


def _numeric_metric(frame: pd.DataFrame, column: str) -> pd.Series:
    try:
        values = pd.to_numeric(frame[column], errors="raise").astype(float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{column} must be numeric") from error
    finite = values.notna()
    if bool(np.isinf(values[finite].to_numpy(dtype=float)).any()):
        raise ValueError(f"{column} contains infinite values")
    return values


def _target_metric_pair(
    frame: pd.DataFrame, stem: str, metric: str
) -> tuple[pd.Series, pd.Series] | None:
    left_name = f"{stem}_target_a_{metric}"
    right_name = f"{stem}_target_b_{metric}"
    present = (left_name in frame, right_name in frame)
    if present == (False, False):
        return None
    if present != (True, True):
        missing = right_name if present[0] else left_name
        raise ValueError(f"paired ownership metric is incomplete: missing {missing}")
    return _numeric_metric(frame, left_name), _numeric_metric(frame, right_name)


def _dominant_target_ids(
    frame: pd.DataFrame, left: pd.Series, right: pd.Series
) -> pd.Series:
    class_a = _validated_class_ids(frame, "class_a")
    class_b = _validated_class_ids(frame, "class_b")
    left_values = left.to_numpy(dtype=float)
    right_values = right.to_numpy(dtype=float)
    ids: list[object] = []
    for index, (left_value, right_value) in enumerate(zip(left_values, right_values)):
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            ids.append(pd.NA)
        elif left_value > right_value:
            ids.append(int(class_a[index]))
        elif right_value > left_value:
            ids.append(int(class_b[index]))
        else:
            ids.append(pd.NA)
    return pd.Series(pd.array(ids, dtype="Int64"), index=frame.index)


def _background_fraction(frame: pd.DataFrame, stem: str) -> pd.Series:
    explicit = f"{stem}_background_fraction"
    alias = f"{stem}_bg_fraction"
    if explicit not in frame and alias not in frame:
        raise ValueError(
            f"shared ownership table requires {explicit} (or its {alias} alias)"
        )
    if explicit not in frame:
        return _numeric_metric(frame, alias)
    values = _numeric_metric(frame, explicit)
    if alias in frame:
        alias_values = _numeric_metric(frame, alias)
        equal = np.isclose(
            values.to_numpy(dtype=float),
            alias_values.to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        if not bool(equal.all()):
            raise ValueError(f"{explicit} and {alias} disagree")
    return values


def _dominant_owner(frame: pd.DataFrame, stem: str) -> pd.Series:
    targets = _target_metric_pair(frame, stem, "fraction")
    if targets is None:
        raise ValueError(f"shared ownership table has no {stem} target fractions")
    left, right = targets
    other = _numeric_metric(frame, f"{stem}_other_fg_fraction")
    background = _background_fraction(frame, stem)
    mixed_void = _numeric_metric(frame, f"{stem}_mixed_void_fraction")
    matrix = np.column_stack(
        (
            np.maximum(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
            other.to_numpy(dtype=float),
            background.to_numpy(dtype=float),
            mixed_void.to_numpy(dtype=float),
        )
    )
    labels = ("target", "other_fg", "background", "mixed_void")
    owners: list[object] = []
    for values in matrix:
        if not np.isfinite(values).all():
            owners.append(pd.NA)
            continue
        maximum = float(values.max())
        winners = np.flatnonzero(values == maximum)
        owners.append(labels[int(winners[0])] if len(winners) == 1 else "tie")
    return pd.Series(pd.array(owners, dtype="string"), index=frame.index)


def add_order_invariant_shared_metrics(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add endpoint-order-invariant target summaries to canonical shared rows.

    For both ``shared`` and (when present) ``new_shared`` supports, the pair
    target fraction is A+B and the dominant-target fraction is max(A,B).
    Dominant-target enrichment, when available, follows the endpoint selected
    by the dominant *fraction*.  A combined pair-target enrichment is not
    synthesized because the canonical rows do not store the combined target
    area needed to compute it exactly.
    """

    validate_unordered_pairs(frame, identity_columns=identity_columns)
    result = frame.copy(deep=True)
    for stem in ("shared", "new_shared"):
        fractions = _target_metric_pair(result, stem, "fraction")
        if fractions is None:
            if stem == "shared":
                raise ValueError(
                    "shared ownership table has no shared target fractions"
                )
            continue
        left, right = fractions
        finite_pair = left.notna() & right.notna()
        result[f"{stem}_pair_target_fraction"] = (left + right).where(finite_pair)
        result[f"{stem}_dominant_target_fraction"] = pd.concat(
            (left, right), axis=1
        ).max(axis=1, skipna=False)
        dominant_ids = _dominant_target_ids(result, left, right)
        result[f"{stem}_dominant_target_class_id"] = dominant_ids

        enrichments = _target_metric_pair(result, stem, "enrichment")
        if enrichments is not None:
            left_enrichment, right_enrichment = enrichments
            class_a = _validated_class_ids(result, "class_a")
            class_b = _validated_class_ids(result, "class_b")
            dominant = dominant_ids.to_numpy(dtype=float, na_value=np.nan)
            selected = np.where(
                dominant == class_a,
                left_enrichment.to_numpy(dtype=float),
                np.where(
                    dominant == class_b,
                    right_enrichment.to_numpy(dtype=float),
                    np.nan,
                ),
            )
            result[f"{stem}_dominant_target_enrichment"] = selected

        result[f"{stem}_dominant_owner"] = _dominant_owner(result, stem)
    return result


def _check_output_collisions(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    collisions = [column for column in columns if column in frame.columns]
    if collisions:
        raise ValueError(f"endpoint output columns already exist: {collisions}")


def _add_endpoint_metadata(
    endpoint: pd.DataFrame, *, focal: str, partner: str
) -> pd.DataFrame:
    for left, right, own_name, partner_name in _ENDPOINT_METADATA:
        present = (left in endpoint, right in endpoint)
        if present == (False, False):
            continue
        if present != (True, True):
            missing = right if present[0] else left
            raise ValueError(f"endpoint metadata pair is incomplete: missing {missing}")
        focal_source = left if focal == "a" else right
        partner_source = right if partner == "b" else left
        endpoint[own_name] = endpoint[focal_source]
        endpoint[partner_name] = endpoint[partner_source]
    return endpoint


def _expand_base(frame: pd.DataFrame) -> pd.DataFrame:
    metadata_outputs = [
        output
        for left, right, own, partner in _ENDPOINT_METADATA
        if left in frame or right in frame
        for output in (own, partner)
    ]
    _check_output_collisions(frame, (*_ENDPOINT_OUTPUT_COLUMNS, *metadata_outputs))
    source = frame.reset_index(drop=True).copy(deep=True)
    source["__source_row"] = np.arange(len(source), dtype=np.int64)

    endpoint_a = source.copy(deep=True)
    endpoint_a["__endpoint_order"] = 0
    endpoint_a["focal_endpoint"] = "a"
    endpoint_a["class_id"] = endpoint_a["class_a"]
    endpoint_a["partner_class_id"] = endpoint_a["class_b"]
    endpoint_a = _add_endpoint_metadata(endpoint_a, focal="a", partner="b")

    endpoint_b = source.copy(deep=True)
    endpoint_b["__endpoint_order"] = 1
    endpoint_b["focal_endpoint"] = "b"
    endpoint_b["class_id"] = endpoint_b["class_b"]
    endpoint_b["partner_class_id"] = endpoint_b["class_a"]
    endpoint_b = _add_endpoint_metadata(endpoint_b, focal="b", partner="a")

    expanded = pd.concat((endpoint_a, endpoint_b), ignore_index=True, sort=False)
    expanded = expanded.sort_values(
        ["__source_row", "__endpoint_order"], kind="stable"
    ).drop(columns=["__source_row", "__endpoint_order"])
    return expanded.reset_index(drop=True)


def validate_endpoint_rows(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Validate that every unordered pair identity has exactly two endpoints."""

    identities = validate_unordered_pairs(
        frame.drop_duplicates(list(_identity_columns(frame, identity_columns))),
        identity_columns=identity_columns,
    )
    _require_columns(frame, (*_ENDPOINT_OUTPUT_COLUMNS,))
    class_a = _validated_class_ids(frame, "class_a")
    class_b = _validated_class_ids(frame, "class_b")
    focal = _validated_class_ids(frame, "class_id")
    partner = _validated_class_ids(frame, "partner_class_id")
    valid_orientation = ((focal == class_a) & (partner == class_b)) | (
        (focal == class_b) & (partner == class_a)
    )
    if not bool(valid_orientation.all()):
        raise ValueError("endpoint class_id/partner_class_id is inconsistent with pair")
    if not frame["focal_endpoint"].isin(("a", "b")).all():
        raise ValueError("focal_endpoint must be either 'a' or 'b'")

    endpoint_keys = [*identities, "class_id", "partner_class_id"]
    if bool(frame.duplicated(endpoint_keys, keep=False).any()):
        raise ValueError("expanded pair table contains duplicate focal endpoints")
    sizes = frame.groupby(list(identities), dropna=False, sort=False).size()
    if not bool((sizes == 2).all()):
        raise ValueError(
            "every unordered pair identity must expand to exactly two rows"
        )
    endpoints = frame.groupby(list(identities), dropna=False, sort=False)[
        "focal_endpoint"
    ].agg(lambda values: frozenset(values))
    if not endpoints.map(lambda values: values == frozenset(("a", "b"))).all():
        raise ValueError("every pair identity must contain unique a and b endpoints")
    return identities


def expand_symmetric_pairs_to_endpoints(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Expand each symmetric unordered pair row to its two focal endpoints.

    Every input column is retained.  ``class_id`` and ``partner_class_id``
    identify the focal direction; available name/offset/classification identity
    fields are additionally exposed in the same focal-relative convention.
    """

    identities = validate_unordered_pairs(frame, identity_columns=identity_columns)
    expanded = _expand_base(frame)
    validate_endpoint_rows(expanded, identity_columns=identities)
    return expanded


def _target_endpoint_columns(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    mappings: list[tuple[str, str, str, str]] = []
    for left in frame.columns:
        if "target_a" not in left or not (
            left.startswith("shared_") or left.startswith("new_shared_")
        ):
            continue
        right = left.replace("target_a", "target_b", 1)
        if right not in frame:
            raise ValueError(f"paired ownership metric is incomplete: missing {right}")
        own = left.replace("target_a", "own_target", 1)
        partner = left.replace("target_a", "partner_target", 1)
        mappings.append((left, right, own, partner))
    return mappings


def expand_shared_pairs_to_endpoints(
    frame: pd.DataFrame,
    *,
    identity_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add invariant summaries and expand shared ownership to focal classes.

    Positional ``target_a``/``target_b`` columns are retained for provenance and
    copied to endpoint-relative ``own_target``/``partner_target`` columns.  This
    includes ``new_shared`` fractions and any enrichment pair present in the
    input schema.
    """

    identities = validate_unordered_pairs(frame, identity_columns=identity_columns)
    enriched = add_order_invariant_shared_metrics(frame, identity_columns=identities)
    mappings = _target_endpoint_columns(enriched)
    output_names = [name for mapping in mappings for name in mapping[2:]]
    _check_output_collisions(enriched, output_names)
    expanded = _expand_base(enriched)
    endpoint_a = expanded["focal_endpoint"].eq("a")
    endpoint_b = expanded["focal_endpoint"].eq("b")
    for left, right, own, partner in mappings:
        expanded[own] = np.where(endpoint_a, expanded[left], expanded[right])
        expanded[partner] = np.where(endpoint_a, expanded[right], expanded[left])
        if not bool((endpoint_a | endpoint_b).all()):  # Defensive schema guard.
            raise RuntimeError("unexpected focal endpoint during ownership remap")
    validate_endpoint_rows(expanded, identity_columns=identities)
    return expanded


__all__ = [
    "DEFAULT_PAIR_IDENTITY_COLUMNS",
    "add_order_invariant_shared_metrics",
    "expand_shared_pairs_to_endpoints",
    "expand_symmetric_pairs_to_endpoints",
    "validate_endpoint_rows",
    "validate_unordered_pairs",
]
