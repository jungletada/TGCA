import torch

from models.tgca import token_group_normalize


def _group_mass(attention, groups, group_id):
    return attention[..., groups == group_id].sum(dim=-1)


def test_tgca_group_and_output_replication_invariance():
    torch.manual_seed(23)
    logits = torch.randn(2, 3, 4, 7)
    values = torch.randn(2, 3, 7, 5)
    groups = torch.tensor([0, 0, 1, 1, 1, 1, 1])

    original_attention = token_group_normalize(logits, groups, mode="tgca")
    original_output = original_attention @ values

    replicated_logits = torch.cat((logits[..., :2], logits[..., 2:].repeat_interleave(4, -1)), -1)
    replicated_values = torch.cat(
        (values[..., :2, :], values[..., 2:, :].repeat_interleave(4, -2)), -2
    )
    replicated_groups = torch.cat(
        (groups[:2], groups[2:].repeat_interleave(4)), dim=0
    )
    replicated_attention = token_group_normalize(
        replicated_logits, replicated_groups, mode="tgca"
    )

    torch.testing.assert_close(
        _group_mass(original_attention, groups, 1),
        _group_mass(replicated_attention, replicated_groups, 1),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        replicated_attention @ replicated_values,
        original_output,
        atol=1e-5,
        rtol=0,
    )


def test_vanilla_is_not_replication_invariant_for_equal_logits():
    logits = torch.zeros(1, 1, 1, 4)
    groups = torch.tensor([0, 0, 1, 1])
    original = token_group_normalize(logits, groups, mode="vanilla")

    replicated_logits = torch.cat((logits[..., :2], logits[..., 2:].repeat_interleave(8, -1)), -1)
    replicated_groups = torch.cat((groups[:2], groups[2:].repeat_interleave(8)))
    replicated = token_group_normalize(
        replicated_logits, replicated_groups, mode="vanilla"
    )
    assert _group_mass(replicated, replicated_groups, 1).item() > _group_mass(
        original, groups, 1
    ).item()
