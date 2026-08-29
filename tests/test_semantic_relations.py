import numpy as np
import torch

from analysis.semantic_relations import (
    cam_prediction,
    class_permutation_control,
    conditional_relations,
    conservative_diagnostic_gates,
    confusion_matrix,
    confusion_summary,
    four_region_masks,
    mutual_relation,
    present_class_relation,
    region_composition,
    spatial_minmax,
)
from models.vit import Attention


def test_conditional_relations_have_the_declared_axes():
    raw = torch.tensor(
        [[[3.0, 1.0, -1.0], [0.0, 2.0, 1.0]]], dtype=torch.float32
    )
    class_to_patch, patch_to_class = conditional_relations(
        raw, raw.transpose(1, 2)
    )
    torch.testing.assert_close(
        class_to_patch.sum(dim=-1), torch.ones(1, 2), rtol=0, atol=1e-6
    )
    torch.testing.assert_close(
        patch_to_class.sum(dim=-1), torch.ones(1, 3), rtol=0, atol=1e-6
    )


def test_present_class_relation_masks_absent_classes_before_softmax():
    logits = torch.tensor([[[10.0, 2.0, 1.0], [9.0, -1.0, 3.0]]])
    probability = present_class_relation(logits, [False, True, True])
    assert torch.count_nonzero(probability[..., 0]) == 0
    torch.testing.assert_close(
        probability.sum(dim=-1), torch.ones(1, 2), rtol=0, atol=1e-7
    )
    assert probability[0, 0, 1] > probability[0, 0, 2]
    assert probability[0, 1, 2] > probability[0, 1, 1]


def test_mutual_relation_and_spatial_normalization_are_finite():
    class_to_patch = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.2, 0.1]])
    patch_to_class = torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
    mutual = mutual_relation(class_to_patch, patch_to_class)
    expected = torch.sqrt(class_to_patch * patch_to_class.transpose(0, 1))
    torch.testing.assert_close(mutual, expected)
    normalized = spatial_minmax(mutual)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(normalized.amin(-1), torch.zeros(2))
    torch.testing.assert_close(normalized.amax(-1), torch.ones(2))


def test_cam_prediction_uses_only_image_level_present_classes():
    scores = np.asarray(
        [[0.8, 0.2, 0.9], [0.99, 0.7, 0.1], [0.1, 0.6, 0.4]]
    )
    prediction = cam_prediction(scores, [True, False, True], threshold=0.5)
    np.testing.assert_array_equal(prediction, [1, 3, 1])


def test_region_c_is_semantically_enriched_in_deterministic_example():
    class_score = np.asarray([0.9, 0.2, 0.1, 0.1])
    semantic_probability = np.asarray([0.9, 0.8, 0.1, 0.1])
    target = np.asarray([1, 1, 0, 0])
    regions = four_region_masks(class_score, semantic_probability, threshold=0.5)
    np.testing.assert_array_equal(regions["A"], [True, False, False, False])
    np.testing.assert_array_equal(regions["C"], [False, True, False, False])
    composition = region_composition(regions["C"], target, class_id=1)
    np.testing.assert_array_equal(composition, [1, 0, 0, 0])
    low = class_score <= 0.5
    assert composition[0] / composition[:3].sum() > (target[low] == 1).mean()


def test_foreground_restricted_confusion_does_not_treat_background_as_class():
    target = np.asarray([-1, 0, 1, 1])
    prediction = np.asarray([1, 0, 0, 1])
    valid = target >= 0
    confusion = confusion_matrix(target, prediction, num_classes=2, valid=valid)
    np.testing.assert_array_equal(confusion, [[1, 0], [1, 1]])
    summary = confusion_summary(confusion)
    assert summary["accuracy"] == 2 / 3
    assert np.isfinite(summary["mean_iou"])


def test_normalizer_hook_observes_exact_raw_logits_without_changing_output():
    torch.manual_seed(71)
    module = Attention(dim=24, num_heads=3, qkv_bias=True, num_classes=2).eval()
    inputs = torch.randn(1, 7, 24)
    captured = {}

    def hook(_module, hook_inputs, output):
        captured["raw"] = hook_inputs[0].detach().clone()
        captured["post"] = output.detach().clone()

    handle = module.normalizer.register_forward_hook(hook)
    hooked_output, hooked_attention = module(inputs)
    handle.remove()
    plain_output, plain_attention = module(inputs)

    qkv = module.qkv(inputs).reshape(1, 7, 3, 3, 8).permute(2, 0, 3, 1, 4)
    query, key, _ = qkv.unbind(0)
    expected_raw = (query @ key.transpose(-2, -1)) * module.scale
    torch.testing.assert_close(captured["raw"], expected_raw, rtol=0, atol=1e-7)
    torch.testing.assert_close(captured["post"], torch.softmax(expected_raw, -1))
    torch.testing.assert_close(hooked_output, plain_output, rtol=0, atol=0)
    torch.testing.assert_close(hooked_attention, plain_attention, rtol=0, atol=0)


def test_conservative_gates_reject_tiny_region_c_support():
    gates = conservative_diagnostic_gates(
        pc_accuracy_ci_lower=0.58,
        random_accuracy=0.05,
        maximum_recovery_recall=0.00002,
        region_c_ci_lower=0.7,
        region_c_images=2,
        total_images=1464,
    )
    assert gates["pc_all_above_uniform_random"]
    assert not gates["pc_all_recovers_cp_missed_foreground"]
    assert not gates["region_c_enriched_over_cp_low_reference"]
    assert gates["region_c_minimum_images"] == 74


def test_conservative_region_c_gate_requires_coverage_and_recovery():
    gates = conservative_diagnostic_gates(
        pc_accuracy_ci_lower=0.04,
        random_accuracy=0.05,
        maximum_recovery_recall=0.12,
        region_c_ci_lower=0.08,
        region_c_images=100,
        total_images=1464,
    )
    assert not gates["pc_all_above_uniform_random"]
    assert gates["pc_all_recovers_cp_missed_foreground"]
    assert gates["region_c_enriched_over_cp_low_reference"]


def test_class_permutation_control_rejects_wrong_class_identities():
    confusion = np.full((10, 10), 1, dtype=np.int64)
    np.fill_diagonal(confusion, 91)
    result = class_permutation_control(confusion, resamples=1000, seed=2027)
    assert result["observed_accuracy"] > 0.9
    assert result["permuted_accuracy_ci95"][1] < result["observed_accuracy"]
    assert result["empirical_p_greater_equal"] <= 0.01
    assert result["seed"] == 2027
