"""Tests for endpoint-expanded Experiment 2 class-pair analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from analysis.lazy_assignment.experiment2.pairwise_class_analysis import (
    add_order_invariant_shared_metrics,
    expand_shared_pairs_to_endpoints,
    expand_symmetric_pairs_to_endpoints,
    validate_endpoint_rows,
    validate_unordered_pairs,
)


SHARED_IDENTITY = (
    "model",
    "image_id",
    "class_a",
    "class_b",
    "layer",
    "layer_or_stage",
    "signal",
    "rho",
    "topk_ratio",
)


def _shared_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model": "mctformer_plus",
        "image_id": "image-1",
        "class_a": 2,
        "class_b": 7,
        "class_a_name": "bird",
        "class_b_name": "cat",
        "class_a_offset": 0,
        "class_b_offset": 1,
        "classification_status_a": "both_positive",
        "classification_status_b": "class_only_positive",
        "num_positive_classes": 2,
        "label_stratum": "exactly_2_labels",
        "layer": 12,
        "layer_or_stage": "L12",
        "signal": "feature_post",
        "rho": 0.5,
        "topk_ratio": 0.10,
        "shared_target_a_fraction": 0.20,
        "shared_target_b_fraction": 0.50,
        "shared_target_a_enrichment": 0.80,
        "shared_target_b_enrichment": 1.25,
        "shared_other_fg_fraction": 0.10,
        "shared_background_fraction": 0.15,
        "shared_bg_fraction": 0.15,
        "shared_mixed_void_fraction": 0.05,
        "new_shared_target_a_fraction": 0.10,
        "new_shared_target_b_fraction": 0.40,
        "new_shared_other_fg_fraction": 0.20,
        "new_shared_background_fraction": 0.20,
        "new_shared_bg_fraction": 0.20,
        "new_shared_mixed_void_fraction": 0.10,
        "shared_set_size": 20,
    }
    row.update(overrides)
    return row


def test_order_invariant_shared_metrics_combine_targets_and_identify_owner() -> None:
    source = pd.DataFrame([_shared_row()])
    result = add_order_invariant_shared_metrics(
        source, identity_columns=SHARED_IDENTITY
    )

    assert result.loc[0, "shared_pair_target_fraction"] == pytest.approx(0.70)
    assert result.loc[0, "shared_dominant_target_fraction"] == pytest.approx(0.50)
    assert result.loc[0, "shared_dominant_target_class_id"] == 7
    assert result.loc[0, "shared_dominant_target_enrichment"] == pytest.approx(1.25)
    assert result.loc[0, "shared_dominant_owner"] == "target"
    assert result.loc[0, "new_shared_pair_target_fraction"] == pytest.approx(0.50)
    assert result.loc[0, "new_shared_dominant_target_fraction"] == pytest.approx(0.40)
    assert result.loc[0, "new_shared_dominant_target_class_id"] == 7
    assert result.loc[0, "new_shared_dominant_owner"] == "target"
    pdt.assert_frame_equal(source, pd.DataFrame([_shared_row()]))


def test_order_invariant_metrics_do_not_depend_on_a_b_value_position() -> None:
    rows = [
        _shared_row(
            image_id="left", shared_target_a_fraction=0.2, shared_target_b_fraction=0.5
        ),
        _shared_row(
            image_id="right", shared_target_a_fraction=0.5, shared_target_b_fraction=0.2
        ),
    ]
    result = add_order_invariant_shared_metrics(
        pd.DataFrame(rows), identity_columns=SHARED_IDENTITY
    )
    assert result["shared_pair_target_fraction"].tolist() == pytest.approx([0.7, 0.7])
    assert result["shared_dominant_target_fraction"].tolist() == pytest.approx(
        [0.5, 0.5]
    )
    assert result["shared_dominant_owner"].tolist() == ["target", "target"]


def test_dominant_owner_marks_cross_category_ties_and_empty_support() -> None:
    tied = _shared_row(
        image_id="tie",
        shared_target_a_fraction=0.30,
        shared_target_b_fraction=0.10,
        shared_other_fg_fraction=0.30,
        shared_background_fraction=0.20,
        shared_bg_fraction=0.20,
        shared_mixed_void_fraction=0.10,
    )
    empty = _shared_row(image_id="empty")
    for stem in ("shared", "new_shared"):
        for region in (
            "target_a",
            "target_b",
            "other_fg",
            "background",
            "bg",
            "mixed_void",
        ):
            name = f"{stem}_{region}_fraction"
            if name in empty:
                empty[name] = np.nan
    result = add_order_invariant_shared_metrics(
        pd.DataFrame([tied, empty]), identity_columns=SHARED_IDENTITY
    )
    assert result.loc[0, "shared_dominant_owner"] == "tie"
    assert pd.isna(result.loc[1, "shared_dominant_owner"])
    assert pd.isna(result.loc[1, "shared_dominant_target_class_id"])
    assert pd.isna(result.loc[1, "new_shared_dominant_owner"])


@pytest.mark.parametrize(
    ("other", "background", "mixed_void", "expected"),
    (
        (0.60, 0.10, 0.10, "other_fg"),
        (0.10, 0.60, 0.10, "background"),
        (0.10, 0.10, 0.60, "mixed_void"),
    ),
)
def test_dominant_owner_reports_each_non_target_category(
    other: float, background: float, mixed_void: float, expected: str
) -> None:
    source = pd.DataFrame(
        [
            _shared_row(
                shared_target_a_fraction=0.10,
                shared_target_b_fraction=0.10,
                shared_other_fg_fraction=other,
                shared_background_fraction=background,
                shared_bg_fraction=background,
                shared_mixed_void_fraction=mixed_void,
            )
        ]
    )
    result = add_order_invariant_shared_metrics(
        source, identity_columns=SHARED_IDENTITY
    )
    assert result.loc[0, "shared_dominant_owner"] == expected


def test_shared_pair_expansion_remaps_all_endpoint_relative_target_fields() -> None:
    source = pd.DataFrame([_shared_row()])
    result = expand_shared_pairs_to_endpoints(source, identity_columns=SHARED_IDENTITY)

    assert result["focal_endpoint"].tolist() == ["a", "b"]
    assert result["class_id"].tolist() == [2, 7]
    assert result["partner_class_id"].tolist() == [7, 2]
    assert result["class_name"].tolist() == ["bird", "cat"]
    assert result["partner_class_name"].tolist() == ["cat", "bird"]
    assert result["class_offset"].tolist() == [0, 1]
    assert result["partner_class_offset"].tolist() == [1, 0]
    assert result["classification_status"].tolist() == [
        "both_positive",
        "class_only_positive",
    ]
    assert result["partner_classification_status"].tolist() == [
        "class_only_positive",
        "both_positive",
    ]

    assert result["shared_own_target_fraction"].tolist() == pytest.approx([0.2, 0.5])
    assert result["shared_partner_target_fraction"].tolist() == pytest.approx(
        [0.5, 0.2]
    )
    assert result["shared_own_target_enrichment"].tolist() == pytest.approx([0.8, 1.25])
    assert result["shared_partner_target_enrichment"].tolist() == pytest.approx(
        [1.25, 0.8]
    )
    assert result["new_shared_own_target_fraction"].tolist() == pytest.approx(
        [0.1, 0.4]
    )
    assert result["new_shared_partner_target_fraction"].tolist() == pytest.approx(
        [0.4, 0.1]
    )
    assert result["shared_pair_target_fraction"].tolist() == pytest.approx([0.7, 0.7])
    assert set(source.columns).issubset(result.columns)
    validate_endpoint_rows(result, identity_columns=SHARED_IDENTITY)


def test_symmetric_pair_expansion_is_interleaved_and_preserves_input_columns() -> None:
    source = pd.DataFrame(
        [
            {
                "model": "mctformer",
                "image_id": "i",
                "class_a": 2,
                "class_b": 7,
                "class_a_name": "bird",
                "class_b_name": "cat",
                "layer": layer,
                "class_token_cosine": value,
                "feature_post_spearman": value / 2,
            }
            for layer, value in ((9, 0.4), (10, 0.6))
        ]
    )
    identities = ("model", "image_id", "class_a", "class_b", "layer")
    result = expand_symmetric_pairs_to_endpoints(source, identity_columns=identities)

    assert result["layer"].tolist() == [9, 9, 10, 10]
    assert result["class_id"].tolist() == [2, 7, 2, 7]
    assert result["partner_class_id"].tolist() == [7, 2, 7, 2]
    assert result["class_token_cosine"].tolist() == [0.4, 0.4, 0.6, 0.6]
    assert set(source.columns).issubset(result.columns)
    validate_endpoint_rows(result, identity_columns=identities)


@pytest.mark.parametrize(
    ("class_a", "class_b"),
    ((7, 2), (2, 2)),
)
def test_unordered_pair_validation_rejects_reversed_or_equal_ids(
    class_a: int, class_b: int
) -> None:
    frame = pd.DataFrame([_shared_row(class_a=class_a, class_b=class_b)])
    with pytest.raises(ValueError, match="class_a < class_b"):
        validate_unordered_pairs(frame, identity_columns=SHARED_IDENTITY)


def test_unordered_pair_validation_rejects_duplicate_identity() -> None:
    frame = pd.DataFrame([_shared_row(), _shared_row(shared_set_size=99)])
    with pytest.raises(ValueError, match="duplicate unordered pair identity"):
        validate_unordered_pairs(frame, identity_columns=SHARED_IDENTITY)


def test_shared_metrics_require_both_endpoint_columns_and_consistent_bg_alias() -> None:
    missing_partner = pd.DataFrame([_shared_row()]).drop(
        columns="shared_target_b_enrichment"
    )
    with pytest.raises(ValueError, match="shared_target_b_enrichment"):
        add_order_invariant_shared_metrics(
            missing_partner, identity_columns=SHARED_IDENTITY
        )

    mismatched_alias = pd.DataFrame([_shared_row(shared_bg_fraction=0.9)])
    with pytest.raises(ValueError, match="disagree"):
        add_order_invariant_shared_metrics(
            mismatched_alias, identity_columns=SHARED_IDENTITY
        )


def test_endpoint_validation_rejects_a_missing_or_duplicate_endpoint() -> None:
    identities = ("model", "image_id", "class_a", "class_b", "layer")
    source = pd.DataFrame(
        [
            {
                "model": "mctformer",
                "image_id": "i",
                "class_a": 2,
                "class_b": 7,
                "layer": 12,
                "class_token_cosine": 0.5,
            }
        ]
    )
    valid = expand_symmetric_pairs_to_endpoints(source, identity_columns=identities)
    with pytest.raises(ValueError, match="exactly two"):
        validate_endpoint_rows(valid.iloc[:1].copy(), identity_columns=identities)
    duplicated = pd.concat((valid.iloc[:1], valid.iloc[:1]), ignore_index=True)
    with pytest.raises(ValueError, match="duplicate focal endpoints"):
        validate_endpoint_rows(duplicated, identity_columns=identities)
