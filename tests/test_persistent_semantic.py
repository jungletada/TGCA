import pytest
import torch

from models.mctformer_plus import MCTformerPlus, MCTformerPlusCam
from models.persistent_semantic import (
    PSL_VARIANTS,
    SemanticReadWrite,
    parse_interaction_layers,
    validate_psl_variant,
)


def small_model(model_type=MCTformerPlus, variant="read_write"):
    return model_type(
        input_size=32,
        img_size=32,
        patch_size=16,
        embed_dim=24,
        depth=2,
        num_heads=3,
        mlp_ratio=2,
        num_classes=3,
        drop_path_rate=0.0,
        psl_variant=variant,
        psl_interaction_layers=(1,),
        psl_relation_dim=24,
    )


def test_variant_and_layer_validation_is_explicit():
    assert tuple(validate_psl_variant(name) for name in PSL_VARIANTS)
    assert parse_interaction_layers("1,3,5") == (1, 3, 5)
    with pytest.raises(ValueError):
        validate_psl_variant("unknown")
    with pytest.raises(ValueError):
        parse_interaction_layers("3,1")


def test_shared_relation_normalizes_both_conditional_directions():
    module = SemanticReadWrite(dim=12, relation_dim=12, read=True, write=True)
    semantic = torch.randn(2, 4, 12)
    patches = torch.randn(2, 7, 12)
    _, _, relation = module(semantic, patches)
    torch.testing.assert_close(
        relation["read_attention"].sum(-1), torch.ones(2, 4), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        relation["write_attention"].sum(-1), torch.ones(2, 7), atol=1e-6, rtol=0
    )


def test_zero_write_gate_preserves_patch_stream_and_backpropagates_to_gate():
    module = SemanticReadWrite(dim=12, relation_dim=12, read=True, write=True)
    semantic = torch.randn(2, 4, 12, requires_grad=True)
    patches = torch.randn(2, 7, 12, requires_grad=True)
    updated_semantic, updated_patches, _ = module(semantic, patches)
    assert torch.equal(updated_patches, patches)
    (updated_semantic.square().mean() + updated_patches.square().mean()).backward()
    assert module.write_gate.grad is not None
    assert torch.isfinite(module.write_gate.grad)


def test_zero_write_gate_makes_read_write_patch_path_match_read_only():
    read_only = small_model(variant="read_only").eval()
    read_write = small_model(variant="read_write").eval()
    read_write.load_state_dict(read_only.state_dict(), strict=True)
    inputs = torch.randn(2, 3, 32, 32)
    read_classes, read_patches, _, _ = read_only.forward_features(inputs)
    rw_classes, rw_patches, _, _ = read_write.forward_features(inputs)
    torch.testing.assert_close(rw_classes, read_classes, atol=0, rtol=0)
    torch.testing.assert_close(rw_patches, read_patches, atol=0, rtol=0)


def test_read_write_uses_one_background_latent_outside_patch_attention():
    model = small_model()
    inputs = torch.randn(2, 3, 32, 32)
    classes, patches, attentions, _, auxiliary = model.forward_features(
        inputs, return_aux=True
    )
    assert classes.shape == (2, 3, 24)
    assert patches.shape == (2, 4, 24)
    assert attentions[0].shape == (2, 3, 4, 4)
    assert auxiliary["semantic_latents"].shape == (2, 4, 24)
    relation = auxiliary["psl_relations"][0]
    assert relation["relation"].shape == (2, 4, 4)
    assert relation["write_attention"].shape == (2, 4, 4)


def test_parameter_matched_variants_have_the_same_state_shapes():
    state_shapes = []
    for variant in ("read_only", "write_only", "read_write"):
        state_shapes.append({key: value.shape for key, value in small_model(
            variant=variant).state_dict().items()})
    assert state_shapes[0] == state_shapes[1] == state_shapes[2]


def test_backbone_initialization_copies_qkv_and_keeps_write_gate_zero():
    model = small_model()
    model.initialize_psl_from_backbone()
    interaction = model.semantic_interactions["1"]
    query, key, value = model.blocks[1].attn.qkv.weight.chunk(3, dim=0)
    torch.testing.assert_close(interaction.semantic_query.weight, query)
    torch.testing.assert_close(interaction.patch_key.weight, key)
    torch.testing.assert_close(interaction.patch_value.weight, value)
    torch.testing.assert_close(interaction.write_gate, torch.zeros(()))


def test_phase2_cam_and_diagnostic_paths_have_stable_shapes():
    model = small_model(MCTformerPlusCam).eval()
    diagnostics = model(torch.randn(1, 3, 32, 32), return_diagnostics=True)
    assert diagnostics["final_cam"].shape == (1, 3, 2, 2)
    assert diagnostics["class_to_patch_layers"].shape == (1, 1, 3, 4)
    assert diagnostics["patch_to_class_layers"].shape == (1, 1, 4, 3)
    assert diagnostics["patch_to_background_layers"].shape == (1, 1, 4, 1)
    assert diagnostics["semantic_latents"].shape == (1, 4, 24)
