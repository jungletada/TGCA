import pytest
import torch

from models.tgca import TokenGroupNormalizer, token_group_normalize


def test_vanilla_matches_torch_softmax():
    torch.manual_seed(7)
    logits = torch.randn(2, 3, 4, 9, dtype=torch.float32)
    groups = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1])
    actual = token_group_normalize(logits, groups, mode="vanilla")
    expected = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


@pytest.mark.parametrize("mode", ["vanilla", "split_05", "tgca", "tgca_bias"])
def test_unit_row_sum_modes(mode):
    torch.manual_seed(11)
    logits = torch.randn(2, 2, 5, 7)
    key_groups = torch.tensor([0, 0, 1, 1, 1, 1, 1])
    query_groups = torch.tensor([0, 0, 1, 1, 1])
    bias = torch.zeros(2, 2, 2) if mode == "tgca_bias" else None
    attention = token_group_normalize(
        logits,
        key_groups,
        query_groups,
        mode=mode,
        relation_bias=bias,
    )
    torch.testing.assert_close(
        attention.sum(dim=-1), torch.ones_like(attention[..., 0]), rtol=0, atol=1e-6
    )


def test_split_11_row_sum_is_two():
    logits = torch.zeros(1, 1, 3, 6)
    groups = torch.tensor([0, 0, 1, 1, 1, 1])
    attention = token_group_normalize(logits, groups, mode="split_11")
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.full_like(attention[..., 0], 2.0),
        rtol=0,
        atol=1e-6,
    )


def test_one_group_tgca_reduces_to_vanilla():
    torch.manual_seed(3)
    logits = torch.randn(2, 3, 4, 8)
    groups = torch.zeros(8, dtype=torch.long)
    actual = token_group_normalize(logits, groups, mode="tgca")
    expected = torch.softmax(logits, dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-7)


def test_mask_uses_per_sample_valid_group_counts():
    logits = torch.zeros(2, 1, 1, 6)
    groups = torch.tensor([0, 0, 1, 1, 1, 1])
    valid = torch.tensor(
        [[True, True, True, True, False, False], [True, False, True, True, True, True]]
    )
    attention = token_group_normalize(
        logits, groups, key_valid_mask=valid, mode="tgca"
    )
    assert torch.count_nonzero(attention[~valid[:, None, None, :]]) == 0
    group0 = attention[..., :2].sum(dim=-1)
    group1 = attention[..., 2:].sum(dim=-1)
    torch.testing.assert_close(group0, torch.full_like(group0, 0.5), atol=1e-6, rtol=0)
    torch.testing.assert_close(group1, torch.full_like(group1, 0.5), atol=1e-6, rtol=0)


def test_fully_masked_row_fails_loudly():
    logits = torch.zeros(1, 1, 1, 4)
    with pytest.raises(ValueError, match="at least one valid key"):
        token_group_normalize(
            logits,
            torch.tensor([0, 0, 1, 1]),
            key_valid_mask=torch.zeros(4, dtype=torch.bool),
            mode="tgca",
        )


def test_relation_bias_lookup_and_zero_bias_reduction():
    logits = torch.zeros(1, 1, 2, 2)
    key_groups = torch.tensor([0, 1])
    query_groups = torch.tensor([0, 1])
    zero = torch.zeros(1, 2, 2)
    tgca = token_group_normalize(
        logits, key_groups, query_groups, mode="tgca"
    )
    tgca_zero_bias = token_group_normalize(
        logits,
        key_groups,
        query_groups,
        mode="tgca_bias",
        relation_bias=zero,
    )
    torch.testing.assert_close(tgca_zero_bias, tgca, rtol=0, atol=0)

    bias = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]])
    attention = token_group_normalize(
        logits,
        key_groups,
        query_groups,
        mode="tgca_bias",
        relation_bias=bias,
    )
    expected = torch.softmax(torch.tensor([2.0, 0.0]), dim=0)
    torch.testing.assert_close(attention[0, 0, 0], expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(attention[0, 0, 1], expected.flip(0), atol=1e-6, rtol=0)


def test_rectangular_attention_and_gradients():
    torch.manual_seed(17)
    logits = torch.randn(2, 3, 4, 7, requires_grad=True)
    relation_bias = torch.zeros(3, 2, 2, requires_grad=True)
    attention = token_group_normalize(
        logits,
        torch.tensor([0, 0, 1, 1, 1, 1, 1]),
        torch.tensor([0, 0, 1, 1]),
        mode="tgca_bias",
        relation_bias=relation_bias,
    )
    values = torch.randn(2, 3, 7, 5, requires_grad=True)
    output = attention @ values
    output.square().mean().backward()
    for tensor in (logits.grad, values.grad, relation_bias.grad):
        assert tensor is not None
        assert torch.isfinite(tensor).all()


def test_inputs_are_not_mutated():
    logits = torch.randn(1, 2, 3, 5)
    groups = torch.tensor([0, 0, 1, 1, 1])
    mask = torch.tensor([True, True, True, False, True])
    originals = (logits.clone(), groups.clone(), mask.clone())
    token_group_normalize(logits, groups, key_valid_mask=mask, mode="tgca")
    for actual, expected in zip((logits, groups, mask), originals):
        assert torch.equal(actual, expected)


def test_module_relation_parameter_contract():
    count_only = TokenGroupNormalizer(6, 2, 2, mode="tgca")
    with_bias = TokenGroupNormalizer(6, 2, 2, mode="tgca_bias")
    assert sum(parameter.numel() for parameter in count_only.parameters()) == 0
    assert sum(parameter.numel() for parameter in with_bias.parameters()) == 24
    assert torch.count_nonzero(with_bias.relation_bias) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_mixed_precision(dtype):
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")
    logits = torch.randn(2, 2, 4, 9, device="cuda", dtype=dtype, requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1, 1], device="cuda")
    attention = token_group_normalize(logits, groups, mode="tgca")
    assert attention.dtype == dtype
    torch.testing.assert_close(
        attention.float().sum(dim=-1),
        torch.ones_like(attention[..., 0], dtype=torch.float32),
        atol=2e-3,
        rtol=0,
    )
    attention.float().square().mean().backward()
    assert torch.isfinite(logits.grad).all()
