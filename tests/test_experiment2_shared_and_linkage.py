import numpy as np
import pytest

from analysis.lazy_assignment.experiment2.metrics_shared_ownership import (
    pairwise_cosine,
    shared_support_metrics,
)
from analysis.lazy_assignment.experiment2.metrics_stage_linkage import (
    stage_transition_metrics,
)


def test_shared_support_composition_and_new_membership():
    # Pair codes: target_a, target_b, other, bg, mixed, void.
    regions = np.asarray([0, 1, 2, 3, 3, 4, 5, 3, 0, 1])
    old_a = np.asarray([9, 1, 8, 0, 0, 0, 99, 0, 0, 0], dtype=float)
    old_b = np.asarray([9, 1, 8, 0, 0, 0, 99, 0, 0, 0], dtype=float)
    new_a = np.asarray([9, 1, 0, 8, 0, 0, 99, 0, 0, 0], dtype=float)
    new_b = np.asarray([9, 1, 0, 8, 0, 0, 99, 0, 0, 0], dtype=float)
    result = shared_support_metrics(
        new_a,
        new_b,
        regions,
        ratio=0.3,
        previous_scores_a=old_a,
        previous_scores_b=old_b,
    )
    assert result["shared_set_size"] == 3
    assert result["new_shared_from_previous_layer"] == 1
    assert result["new_shared_background_fraction"] == 1.0
    assert result["shared_background_fraction"] == 1 / 3


def test_pairwise_cosine():
    values = pairwise_cosine(np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]))
    np.testing.assert_allclose(np.diag(values), 1.0)
    assert values[0, 1] == 0.0
    assert values[0, 2] == 1.0


def test_stage_transition_semantics():
    regions = np.asarray([0, 0, 2, 2, 1, 1, 3, 4])
    source = np.asarray([8, 7, 1, 0, 6, 2, 3, 100], dtype=float)
    destination = np.asarray([8, 1, 7, 0, 6, 2, 3, 100], dtype=float)
    result = stage_transition_metrics(source, destination, regions, ratio=0.5)
    assert result["source_topk_size"] == 4
    assert result["destination_topk_size"] == 4
    assert result["introduced_size"] == 1
    assert result["removed_size"] == 1
    assert result["introduced_background_fraction"] == pytest.approx(0.25)
    assert result["removed_target_fraction"] == pytest.approx(0.25)
