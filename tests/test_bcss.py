import pytest
import torch

from models.bcss import (
    SemanticSlotDecoder,
    bcss_schedule,
    ownership_calibrate_attention,
    semantic_slot_losses,
    validate_bcss_variant,
)
from models.mctformer_plus import MCTformerPlus, MCTformerPlusCam


def small_model(model_type=MCTformerPlus, variant="e0", **kwargs):
    return model_type(
        input_size=32,
        img_size=32,
        patch_size=16,
        embed_dim=24,
        depth=2,
        num_heads=3,
        mlp_ratio=2,
        num_classes=3,
        bcss_variant=variant,
        drop_path_rate=0.0,
        **kwargs,
    )


def test_schedule_preserves_warmup_and_reaches_final_values():
    assert bcss_schedule(2, 0.5, 0.5) == {
        "progress": 0.0, "tau": 1.0, "beta": 0.0, "refinement_strength": 0.0}
    assert bcss_schedule(8, 0.5, 0.5) == {
        "progress": 1.0, "tau": 0.5, "beta": 0.5, "refinement_strength": 1.0}


@pytest.mark.parametrize("kwargs", (
    {"bcss_tau": 0.0}, {"bcss_beta": -0.1}, {"bcss_beta": 1.1},
    {"bcss_lambda_fg": -0.1}, {"bcss_lambda_bg": -0.1},
    {"bcss_semantic_temperature": 0.0},
))
def test_invalid_screening_hyperparameters_fail_fast(kwargs):
    with pytest.raises(ValueError):
        small_model(**kwargs)


def test_ownership_is_patchwise_normalized_and_inactive_classes_are_zero():
    decoder = SemanticSlotDecoder(dim=12, num_classes=3)
    output = decoder(
        torch.randn(2, 3, 12),
        torch.randn(2, 5, 12),
        torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.bool),
        tau=0.5,
        competitive=True,
    )
    torch.testing.assert_close(
        output["ownership"].sum(dim=1), torch.ones(2, 5), atol=1e-6, rtol=0)
    assert torch.count_nonzero(output["class_ownership"][0, 1]) == 0
    assert torch.count_nonzero(output["class_ownership"][1, (0, 2)]) == 0


def test_beta_zero_is_exact_and_nonzero_beta_preserves_c2p_mass():
    attention = torch.rand(2, 3, 7)
    ownership = torch.rand_like(attention)
    assert torch.equal(ownership_calibrate_attention(attention, ownership, 0), attention)
    calibrated = ownership_calibrate_attention(attention, ownership, 0.5)
    torch.testing.assert_close(
        calibrated.sum(-1), attention.sum(-1), atol=1e-6, rtol=1e-6)


def test_slot_losses_backpropagate_and_null_loss_is_optional():
    decoder = SemanticSlotDecoder(dim=12, num_classes=3)
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    auxiliary = decoder(
        torch.randn(1, 3, 12, requires_grad=True),
        torch.randn(1, 4, 12, requires_grad=True),
        targets.bool(), tau=1.0, competitive=True)
    losses = semantic_slot_losses(
        auxiliary, torch.randn(3, 12), torch.randn(3), targets,
        use_foreground_anchor=True, use_background_null=False)
    assert set(losses) == {"foreground_anchor"}
    losses["foreground_anchor"].backward()
    assert decoder.q_proj.weight.grad is not None


def test_mass_anchor_penalizes_epsilon_foreground_ownership():
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    class_aggregate = torch.eye(3).unsqueeze(0)
    healthy_ownership = torch.zeros(1, 3, 4)
    healthy_ownership[:, 0] = 0.5
    epsilon_ownership = healthy_ownership.clone()
    epsilon_ownership[:, 0] = 1e-6
    classifier_weight = 8.0 * torch.eye(3)

    def foreground_loss(ownership, retain_mass):
        return semantic_slot_losses(
            {
                "class_aggregate": class_aggregate,
                "class_ownership": ownership,
            },
            classifier_weight,
            None,
            targets,
            use_foreground_anchor=True,
            use_background_null=False,
            retain_foreground_ownership_mass=retain_mass,
        )["foreground_anchor"]

    torch.testing.assert_close(
        foreground_loss(healthy_ownership, False),
        foreground_loss(epsilon_ownership, False),
    )
    healthy_loss = foreground_loss(healthy_ownership, True)
    epsilon_loss = foreground_loss(epsilon_ownership, True)
    assert epsilon_loss > healthy_loss + 1.0
    torch.testing.assert_close(
        epsilon_loss, torch.log(torch.tensor(3.0)), atol=1e-5, rtol=0)


def test_mass_anchor_backpropagates_to_ownership_magnitude():
    ownership = torch.full((1, 3, 4), 1e-3, requires_grad=True)
    loss = semantic_slot_losses(
        {
            "class_aggregate": torch.eye(3).unsqueeze(0),
            "class_ownership": ownership,
        },
        8.0 * torch.eye(3),
        None,
        torch.tensor([[1.0, 0.0, 0.0]]),
        use_foreground_anchor=True,
        use_background_null=False,
        retain_foreground_ownership_mass=True,
    )["foreground_anchor"]
    loss.backward()
    assert torch.isfinite(ownership.grad).all()
    assert torch.all(ownership.grad[:, 0] < 0)


def test_e4_mass_changes_only_the_foreground_anchor_contract():
    e4 = validate_bcss_variant("e4")
    e4_mass = validate_bcss_variant("e4_mass")
    assert not e4.foreground_mass_anchor
    assert e4_mass.foreground_mass_anchor
    assert {
        **e4.__dict__, "foreground_mass_anchor": True
    } == e4_mass.__dict__


def test_e4_mass_preserves_e4_forward_state_before_auxiliary_loss():
    torch.manual_seed(17)
    e4 = small_model(variant="e4")
    torch.manual_seed(17)
    e4_mass = small_model(variant="e4_mass")
    for key, value in e4.state_dict().items():
        torch.testing.assert_close(value, e4_mass.state_dict()[key], rtol=0, atol=0)

    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    e4_outputs = e4(inputs, active_labels=targets)
    mass_outputs = e4_mass(inputs, active_labels=targets)
    for index in range(3):
        torch.testing.assert_close(e4_outputs[index], mass_outputs[index], rtol=0, atol=0)
    for key in ("ownership", "class_aggregate", "background_aggregate"):
        torch.testing.assert_close(
            e4_outputs[3][key], mass_outputs[3][key], rtol=0, atol=0)


def test_e0_has_no_new_state_and_preserves_original_output_contract():
    model = small_model(variant="e0")
    assert not any(
        "semantic_slot" in key or "register_token" in key or "background_token" in key
        for key in model.state_dict())
    outputs = model(torch.randn(2, 3, 32, 32))
    assert len(outputs) == 3
    assert outputs[0].shape == (2, 3)


@pytest.mark.parametrize("variant", ("e1", "e2", "e4", "e4_mass", "e5", "e6"))
def test_screening_variants_have_stable_forward_contract(variant):
    model = small_model(variant=variant)
    inputs = torch.randn(2, 3, 32, 32)
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    outputs = model(inputs, active_labels=targets)
    assert len(outputs) == (4 if variant in ("e4", "e4_mass", "e5", "e6") else 3)
    assert outputs[0].shape == (2, 3)


@pytest.mark.parametrize("variant,parameter", (
    ("e1", "register_token"), ("e2", "background_token")))
def test_backbone_control_token_receives_base_loss_gradient(variant, parameter):
    model = small_model(variant=variant)
    outputs = model(torch.randn(2, 3, 32, 32))
    (outputs[0].sum() + outputs[2].sum()).backward()
    gradient = getattr(model, parameter).grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()


def test_cam_diagnostics_export_three_baseline_maps_and_ownership():
    model = small_model(MCTformerPlusCam, variant="e6")
    model.set_bcss_epoch(8)
    diagnostics = model(
        torch.randn(1, 3, 32, 32),
        active_labels=torch.tensor([[1.0, 0.0, 1.0]]),
        return_diagnostics=True,
    )
    for key in ("patch_cam", "class_to_patch", "final_cam", "class_ownership",
                "background_ownership", "patch_feature_norm"):
        assert key in diagnostics
    torch.testing.assert_close(
        diagnostics["ownership"].sum(dim=1),
        torch.ones(1, 2, 2), atol=1e-6, rtol=0)


def test_e2_exports_independent_background_attention_directions():
    model = small_model(MCTformerPlusCam, variant="e2")
    model.set_bcss_epoch(8)
    diagnostics = model(torch.randn(1, 3, 32, 32), return_diagnostics=True)
    assert diagnostics["background_attention"].shape == (1, 1, 2, 2)
    assert diagnostics["background_to_patch"].shape[-1] == 4
    assert diagnostics["patch_to_background"].shape[-1] == 4
