import torch

from analysis.lazy_assignment.score_utils import class_specific_patch_score
from analysis.lazy_assignment.token_collector import BlockTokenCollector
from models.mctformer import MCTformerV2Cam
from models.mctformer_plus import MCTformerPlusCam


def _assert_forward_equal(left, right):
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        return
    if isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_forward_equal(left_item, right_item)
        return
    assert left == right


def test_mctformerplus_collector_captures_all_blocks_without_numerical_change():
    torch.manual_seed(17)
    model = MCTformerPlusCam(
        num_classes=3,
        input_size=32,
        attention_normalization="vanilla",
    ).eval()
    image = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        baseline = model.forward_features(image)
        with BlockTokenCollector(model, num_classes=3) as collector:
            collector.clear(expected_num_patches=4)
            instrumented = model.forward_features(image)
            capture = collector.consume()

    _assert_forward_equal(baseline, instrumented)
    assert capture.scores.shape == (12, 1, 3, 4)
    torch.testing.assert_close(capture.last_class_tokens, instrumented[0], rtol=0, atol=0)
    torch.testing.assert_close(capture.last_patch_tokens, instrumented[1], rtol=0, atol=0)
    torch.testing.assert_close(
        capture.scores[-1],
        class_specific_patch_score(instrumented[0], instrumented[1]),
        rtol=0,
        atol=1e-6,
    )
    assert all(not block._forward_hooks for block in model.blocks)


def test_mctformerv2_collector_captures_all_blocks_without_numerical_change():
    torch.manual_seed(1701)
    model = MCTformerV2Cam(num_classes=3, input_size=32).eval()
    image = torch.randn(1, 3, 32, 32)

    with torch.inference_mode():
        baseline = model.forward_features(image)
        with BlockTokenCollector(model, num_classes=3) as collector:
            collector.clear(expected_num_patches=4)
            instrumented = model.forward_features(image)
            capture = collector.consume()

    _assert_forward_equal(baseline, instrumented)
    assert capture.scores.shape == (12, 1, 3, 4)
    torch.testing.assert_close(capture.last_class_tokens, instrumented[0], rtol=0, atol=0)
    torch.testing.assert_close(capture.last_patch_tokens, instrumented[1], rtol=0, atol=0)
    torch.testing.assert_close(
        capture.scores[-1],
        class_specific_patch_score(instrumented[0], instrumented[1]),
        rtol=0,
        atol=1e-6,
    )
    assert all(not block._forward_hooks for block in model.blocks)


def test_collector_can_be_reused_without_duplicate_hooks_or_stale_scores():
    torch.manual_seed(29)
    model = MCTformerPlusCam(num_classes=2, input_size=32).eval()
    first = torch.randn(1, 3, 32, 32)
    second = torch.randn(1, 3, 32, 32)
    with torch.inference_mode(), BlockTokenCollector(model, 2) as collector:
        collector.clear(expected_num_patches=4)
        model.forward_features(first)
        first_capture = collector.consume()
        collector.clear(expected_num_patches=4)
        model.forward_features(second)
        second_capture = collector.consume()
    assert first_capture.scores.shape == second_capture.scores.shape
    assert not torch.equal(first_capture.scores, second_capture.scores)
    assert all(not block._forward_hooks for block in model.blocks)
