import numpy as np
import pytest
import torch
import torch.nn.functional as F

from analysis.lazy_assignment.score_utils import (
    LayerScoreSummary,
    class_specific_patch_score,
    infer_patch_grid,
)


def test_class_specific_patch_score_matches_expanded_cosine_similarity():
    torch.manual_seed(2027)
    class_tokens = torch.randn(2, 3, 7, dtype=torch.float64)
    patch_tokens = torch.randn(2, 5, 7, dtype=torch.float64)
    actual = class_specific_patch_score(class_tokens, patch_tokens)
    expected = F.cosine_similarity(
        class_tokens.float().unsqueeze(2).expand(-1, -1, 5, -1),
        patch_tokens.float().unsqueeze(1).expand(-1, 3, -1, -1),
        dim=-1,
    )
    assert actual.shape == (2, 3, 5)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    assert float(actual.min()) >= -1.0 - 1e-6
    assert float(actual.max()) <= 1.0 + 1e-6


def test_class_specific_patch_score_is_float32_for_half_inputs():
    class_tokens = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float16)
    patch_tokens = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]], dtype=torch.float16)
    scores = class_specific_patch_score(class_tokens, patch_tokens)
    assert scores.dtype == torch.float32
    torch.testing.assert_close(scores[0, :, 0], torch.tensor([1.0, 0.0]))


def test_infer_patch_grid_preserves_rectangular_axes():
    assert infer_patch_grid((2, 3, 320, 448), (16, 16), 20 * 28) == (20, 28)
    with pytest.raises(ValueError, match="implies"):
        infer_patch_grid((1, 3, 320, 448), (16, 16), 561)


def test_layer_summary_reports_positive_map_counts_and_exact_quantiles():
    summary = LayerScoreSummary(depth=2)
    scores = np.asarray(
        [
            [[-1.0, 0.0, 1.0]],
            [[0.0, 0.5, 1.0]],
        ],
        dtype=np.float32,
    )
    summary.add_image(scores)
    rows = summary.finish("mctformerplus")
    assert len(rows) == 2
    assert rows[0]["num_images"] == 1
    assert rows[0]["num_positive_class_maps"] == 1
    assert rows[0]["score_min"] == -1.0
    assert rows[0]["score_max"] == 1.0
    assert rows[0]["score_q50"] == 0.0
    assert rows[0]["nan_count"] == 0
    assert rows[0]["inf_count"] == 0
