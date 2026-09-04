import numpy as np

from analysis.lazy_assignment.experiment2.metrics_region import (
    map_overlap_metrics,
    region_map_metrics,
    stable_topk_mask,
)


def test_stable_topk_excludes_void_and_breaks_ties_by_index():
    scores = np.ones(6)
    eligible = np.asarray([False, True, True, True, True, True])
    selected = stable_topk_mask(scores, 0.4, eligible).reshape(-1)
    assert np.flatnonzero(selected).tolist() == [1, 2]


def test_region_metrics_signed_auc_and_area_enrichment():
    # target=0, other=1, bg=2, mixed=3, void=4
    regions = np.asarray([0, 0, 1, 2, 2, 2, 3, 4])
    scores = np.asarray([8.0, 7.0, 3.0, 1.0, 0.0, -1.0, 2.0, 100.0])
    result = region_map_metrics(scores, regions, grid_h=2, grid_w=4)
    assert result["top1_region"] == "target"
    assert result["auc_target_bg"] == 1.0
    assert result["auc_target_other"] == 1.0
    assert result["num_void"] == 1
    assert result["target_top20_fraction"] == 1.0
    assert result["target_tail_enrich_20"] > 1.0
    assert np.isnan(result["conditional_bg_mass"])


def test_auc_is_not_flipped_and_undefined_target_is_nan():
    regions = np.asarray([0, 0, 2, 2])
    reversed_result = region_map_metrics(
        np.asarray([0.0, 1.0, 3.0, 2.0]), regions, grid_h=2, grid_w=2
    )
    assert reversed_result["auc_target_bg"] == 0.0

    no_target = region_map_metrics(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([1, 2, 2, 3]),
        grid_h=2,
        grid_w=2,
    )
    assert np.isnan(no_target["auc_target_bg"])
    assert np.isnan(no_target["ap_target_bg"])


def test_zero_cam_has_no_arbitrary_top1_and_mass_is_undefined():
    result = region_map_metrics(
        np.zeros(4),
        np.asarray([0, 1, 2, 3]),
        grid_h=2,
        grid_w=2,
        nonnegative_mass=True,
    )
    assert result["degenerate_map"]
    assert result["top1_region"] == "degenerate"
    assert np.isnan(result["conditional_bg_mass"])


def test_map_overlap_metrics_are_exact_for_equal_maps():
    values = np.asarray([3.0, 1.0, 2.0, 0.0])
    result = map_overlap_metrics(values, values)
    assert result == {
        "spearman": 1.0,
        "topk_jaccard": 1.0,
        "topk_overlap_coefficient": 1.0,
    }
