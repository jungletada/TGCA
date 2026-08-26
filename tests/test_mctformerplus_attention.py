import torch

from models.vit import Attention


def test_default_vanilla_attention_matches_preintegration_equations():
    torch.manual_seed(29)
    module = Attention(dim=24, num_heads=3, qkv_bias=True, num_classes=2)
    module.eval()
    inputs = torch.randn(2, 7, 24, requires_grad=True)
    actual_output, actual_weights = module(inputs)

    batch, tokens, channels = inputs.shape
    qkv = module.qkv(inputs).reshape(batch, tokens, 3, 3, 8).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)
    expected_weights = torch.softmax(
        (query @ key.transpose(-2, -1)) * module.scale, dim=-1
    )
    expected_output = (expected_weights @ value).transpose(1, 2).reshape(
        batch, tokens, channels
    )
    expected_output = module.proj_drop(module.proj(expected_output))

    torch.testing.assert_close(actual_weights, expected_weights, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=1e-7)


def test_all_modes_preserve_attention_shape_and_expected_row_mass():
    inputs = torch.randn(1, 10, 24)
    for mode, expected_mass in (
        ("vanilla", 1.0),
        ("split_11", 2.0),
        ("split_05", 1.0),
        ("tgca", 1.0),
        ("tgca_bias", 1.0),
    ):
        module = Attention(
            dim=24,
            num_heads=3,
            qkv_bias=True,
            num_classes=2,
            attention_normalization=mode,
        )
        output, weights = module(inputs)
        assert output.shape == inputs.shape
        assert weights.shape == (1, 3, 10, 10)
        torch.testing.assert_close(
            weights.sum(-1),
            torch.full_like(weights[..., 0], expected_mass),
            atol=1e-6,
            rtol=0,
        )
