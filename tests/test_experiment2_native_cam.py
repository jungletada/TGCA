from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from analysis.lazy_assignment.experiment2.native_cam_stages import (
    assert_native_cam_equivalent,
    decompose_native_cam,
    decompose_native_cam_reduced,
    patch_logits_from_tokens,
)
from models.mctformer import MCTformerV2Cam
from models.mctformer_plus import MCTformerPlusCam


def attention_fixture(layers=4, batch=2, heads=3, classes=20, patches=4):
    torch.manual_seed(233)
    total = classes + patches
    logits = torch.randn(layers, batch, heads, total, total)
    head_attention = torch.softmax(logits, dim=-1)
    return head_attention, head_attention.mean(dim=2)


class PlusNativeStub:
    def __init__(self, num_classes=20):
        self.num_classes = num_classes
        self.num_class_tokens = num_classes
        self.cti_bgt = False
        self.n_layers = 3
        self.psl_spec = SimpleNamespace(enabled=False)

    def _foreground_slice(self):
        return slice(0, self.num_classes)

    def _patch_slice(self, patch_count):
        return slice(self.num_classes, self.num_classes + patch_count)

    def _attention_patch_slice(self, patch_count):
        return self._patch_slice(patch_count)


def test_mctformer_decomposition_matches_native_forward_attention():
    heads, head_mean = attention_fixture()
    patch_logits = torch.randn(2, 20, 2, 2)
    native_stub = SimpleNamespace(num_classes=20)
    native = MCTformerV2Cam.forward_attention(
        native_stub,
        patch_logits,
        [heads[layer] for layer in range(heads.shape[0])],
        fuse_layers=3,
    )
    stages = decompose_native_cam(
        "mctformerv2", patch_logits, head_mean, num_classes=20
    )

    assert assert_native_cam_equivalent(stages, native, tolerance=1e-6) < 1e-6
    expected_patch = torch.relu(patch_logits)
    expected_c2p = head_mean[-3:, :, :20, 20:].sum(dim=0)
    expected_c1 = expected_c2p * expected_patch.flatten(2)
    torch.testing.assert_close(stages["patch_cam"], expected_patch)
    torch.testing.assert_close(stages["official_c2p_flat"], expected_c2p)
    torch.testing.assert_close(
        stages["class_attention_cam"], expected_c1.reshape(2, 20, 2, 2)
    )


def test_mctformerplus_decomposition_matches_native_get_cam():
    _, head_mean = attention_fixture()
    patch_logits = torch.randn(2, 20, 2, 2)
    native = MCTformerPlusCam.get_cam(
        PlusNativeStub(), patch_logits, head_mean, auxiliary=None
    )
    stages = decompose_native_cam(
        "MCTformerPlusCam", patch_logits, head_mean, num_classes=20
    )

    assert assert_native_cam_equivalent(stages, native, tolerance=1e-6) < 1e-6
    expected_c2p = head_mean[-3:, :, :20, 20:].mean(dim=0)
    expected_c1 = torch.sqrt(expected_c2p * torch.relu(patch_logits).flatten(2))
    torch.testing.assert_close(stages["official_c2p_flat"], expected_c2p)
    torch.testing.assert_close(
        stages["class_attention_cam"], expected_c1.reshape(2, 20, 2, 2)
    )


@pytest.mark.parametrize("model_name", ["mctformer", "mctformerplus"])
def test_reduced_decomposition_equals_full_attention_decomposition(model_name):
    _, head_mean = attention_fixture()
    patch_logits = torch.randn(2, 20, 2, 2)
    full = decompose_native_cam(model_name, patch_logits, head_mean)
    reduced = decompose_native_cam_reduced(
        model_name,
        patch_logits,
        head_mean[:, :, :20, 20:],
        head_mean[:, :, 20:, 20:].sum(dim=0),
    )
    for key in (
        "patch_logits",
        "patch_cam",
        "official_c2p_flat",
        "class_attention_cam",
        "official_p2p",
        "final_cam",
    ):
        torch.testing.assert_close(reduced[key], full[key], rtol=0, atol=0)


def test_patch_logits_helper_preserves_row_major_patch_grid():
    head = nn.Conv2d(3, 20, kernel_size=3, padding=1)
    model = SimpleNamespace(head=head)
    patches = torch.arange(1 * 6 * 3, dtype=torch.float32).reshape(1, 6, 3)
    expected_grid = patches.reshape(1, 2, 3, 3).permute(0, 3, 1, 2).contiguous()
    expected = head(expected_grid)
    actual = patch_logits_from_tokens(model, patches, grid_size=(2, 3))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_decomposition_rejects_any_extra_token():
    patch_logits = torch.randn(1, 20, 2, 2)
    attention_with_extra = torch.softmax(torch.randn(3, 1, 25, 25), dim=-1)
    with pytest.raises(ValueError, match="extra tokens are not accepted"):
        decompose_native_cam("mctformerplus", patch_logits, attention_with_extra)
