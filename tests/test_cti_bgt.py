"""BGT integration, official-loss numerics and frozen-host regression tests."""
import ast
import subprocess
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from models.cti_bgt import (
    adapt_cti_bgt_finetune, cti_bcam_loss, cti_bgt_maps, cti_max_norm,
    validate_cti_bgt_checkpoint,
)
from models.mctformer_plus import MCTformerPlus, MCTformerPlusCam

ROOT = Path(__file__).resolve().parents[1]


def small_model(kind=MCTformerPlus, enabled=True, **overrides):
    config = dict(input_size=32, img_size=32, patch_size=16, embed_dim=24,
                  depth=3, num_heads=3, mlp_ratio=2, num_classes=3,
                  drop_path_rate=0., cti_bgt=enabled, cti_bgt_n_layers=2,
                  cti_bgt_affinity_start=1)
    config.update(overrides)
    return kind(**config)


def test_default_is_bitwise_equal_to_frozen_pre_port_model_and_cam():
    # Read the immutable pre-port host, not another call to the changed path.
    source = subprocess.check_output(
        ['git', 'show', '5a68992:models/mctformer_plus.py'], cwd=ROOT, text=True)
    tree = ast.parse(source)
    tree.body = [node for node in tree.body
                 if not (isinstance(node, ast.FunctionDef) and node.name == 'mctformerplus')]
    namespace = {'__name__': 'frozen_mctformerplus'}
    exec(compile(tree, '<frozen-pre-bgt-host>', 'exec'), namespace)
    config = dict(input_size=32, img_size=32, patch_size=16, embed_dim=24,
                  depth=3, num_heads=3, mlp_ratio=2, num_classes=3)
    for kind in (MCTformerPlus, MCTformerPlusCam):
        torch.manual_seed(107)
        before = namespace[kind.__name__](**config).eval()
        torch.manual_seed(107)
        after = kind(**config).eval()  # Truly omitted flag, not an opt-out override.
        assert before.state_dict().keys() == after.state_dict().keys()
        for key in before.state_dict():
            torch.testing.assert_close(before.state_dict()[key], after.state_dict()[key],
                                       rtol=0, atol=0)
        after.load_state_dict(before.state_dict(), strict=True)
        inputs = torch.randn(2, 3, 32, 48)
        expected, actual = before(inputs), after(inputs)
        if isinstance(expected, list):
            for old, new in zip(expected, actual):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            sum(x.square().mean() for x in expected).backward()
            sum(x.square().mean() for x in actual).backward()
            for (_, old), (_, new) in zip(before.named_parameters(), after.named_parameters()):
                if old.grad is None:
                    assert new.grad is None
                else:
                    torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
        else:
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_bgt_token_order_attention_boundaries_and_foreground_logits():
    torch.manual_seed(3)
    model = small_model()
    inputs = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    block_output = []
    handle = model.blocks[-1].register_forward_hook(
        lambda _m, _i, out: block_output.append(out[0]))
    output = model(inputs, active_labels=labels)
    handle.remove()
    assert model.num_classes == 3 and model.num_class_tokens == 4
    assert model.head.out_channels == 4
    assert torch.count_nonzero(model.bg_token) == 0
    assert torch.count_nonzero(model.pos_embed_bg) == 0
    assert all(block.attn.num_classes == 4 for block in model.blocks)
    assert model._patch_slice(4) == slice(4, 8)
    assert output[0].shape == output[2].shape == (2, 3)
    assert output[1].shape == (3, 2, 3, 24)  # BG excluded from token regularizer.
    torch.testing.assert_close(output[0], block_output[-1][:, 1:4].mean(-1))
    maps = output[3]['cti_bgt']
    assert maps['class_to_patch'].shape == (2, 4, 4)
    assert maps['patch_affinity'].shape == (2, 4, 4)
    assert maps['refined_cam'].shape == (2, 4, 2, 2)
    assert maps['background'].shape == maps['foreground_union'].shape == (2, 1, 2, 2)
    patch_logits = model.gwrp(model.head(block_output[-1][:, 4:].transpose(
        1, 2).reshape(2, 24, 2, 2))[:, 1:])
    torch.testing.assert_close(output[2], patch_logits)


def test_image_classification_does_not_supervise_background_head():
    model = small_model()
    output = model(torch.randn(2, 3, 32, 32))
    labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    (F.multilabel_soft_margin_loss(output[0], labels) +
     F.multilabel_soft_margin_loss(output[2], labels)).backward()
    assert torch.count_nonzero(model.head.weight.grad[0]) == 0
    assert torch.count_nonzero(model.head.bias.grad[0]) == 0
    assert model.head.weight.grad[1:].abs().sum() > 0


def test_bcam_backward_reaches_bg_token_bg_head_foreground_and_affinity():
    torch.manual_seed(101)
    model = small_model()
    with torch.no_grad():
        model.head.bias.fill_(1.)  # Keep the tiny random head in ReLU support.
    output = model(torch.randn(2, 3, 32, 32), active_labels=torch.ones(2, 3))
    maps = output[3]['cti_bgt']
    maps['patch_affinity'].retain_grad()
    cti_bcam_loss(maps).backward()
    for grad in (model.bg_token.grad, model.head.weight.grad[:1],
                 model.head.weight.grad[1:], maps['patch_affinity'].grad):
        assert grad is not None and torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


@pytest.mark.parametrize('shape', [(32, 32), (32, 48), (48, 32), (48, 48)])
def test_cam_paths_keep_c_channels_and_strict_training_checkpoint(shape):
    torch.manual_seed(5)
    train = small_model()
    cam = small_model(MCTformerPlusCam).eval()
    cam.load_state_dict(train.state_dict(), strict=True)
    x = torch.randn(1, 3, *shape)
    labels = torch.ones(1, 3)
    result = cam(x, active_labels=labels, return_diagnostics=True)
    grid = (shape[0] // 16, shape[1] // 16)
    assert result['final_cam'].shape == (1, 3, *grid)
    assert result['background_cam'].shape == (1, 1, *grid)
    assert result['cti_bgt']['refined_cam'].shape == (1, 4, *grid)
    attention = cam(x, return_attn=True)
    expected = attention[-3:, :, 1:4, 4:].mean(0).reshape(1, 3, *grid)
    torch.testing.assert_close(result['class_to_patch'], expected)
    for output in (cam(x), cam.forward_ablation(x, return_type='cam'), cam.forward_with_label(x)[-1]):
        torch.testing.assert_close(output, result['final_cam'])
    assert cam.forward_with_label(x)[0].shape == cam.forward_with_label(x)[1].shape == (1, 3)


def test_official_slices_and_loss_match_independent_fp64_reference():
    torch.manual_seed(7)
    cam = torch.randn(2, 4, 2, 3, requires_grad=True)
    attention = torch.rand(12, 2, 2, 10, 10, requires_grad=True)
    labels = torch.tensor([[1., 0., 1.], [0., 1., 0.]])
    maps = cti_bgt_maps(cam, attention, labels)
    # Independent scalar/spatial construction; importantly max BEFORE A @ union.
    heads = attention.double().mean(2)
    cp = heads[-6:, :, :4, 4:].sum(0)
    pp = heads[4:, :, 4:, 4:].sum(0)
    fused = cam.double().relu().flatten(2) * cp
    fg = (fused[:, 1:] * labels[..., None]).max(1).values
    bg = fused[:, 0]
    ref_fg = torch.stack([pp[b] @ fg[b] for b in range(2)]).reshape(2, 1, 2, 3)
    ref_bg = torch.stack([pp[b] @ bg[b] for b in range(2)]).reshape(2, 1, 2, 3)
    def official_norm(x):
        x = x.relu()
        hi = x.flatten(2).max(-1).values[..., None, None]
        lo = x.flatten(2).min(-1).values[..., None, None]
        return (x - lo - 1e-5).relu() / (hi - lo + 1e-5)
    expected = ((1 - official_norm(ref_fg)) - official_norm(ref_bg)).abs().mean()
    actual = cti_bcam_loss(maps)
    torch.testing.assert_close(maps['class_to_patch'].double(), cp)
    torch.testing.assert_close(maps['patch_affinity'].double(), pp)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    gradients = torch.autograd.grad(actual, (cam, attention), retain_graph=True)
    reference_gradients = torch.autograd.grad(expected, (cam, attention))
    for got, ref in zip(gradients, reference_gradients):
        torch.testing.assert_close(got, ref, rtol=1e-4, atol=2e-6)
    assert torch.count_nonzero(gradients[0][:, 2][labels[:, 1] == 0]) == 0


def test_label_mask_and_union_before_propagation():
    cam = torch.tensor([[[[1., 2.]], [[2., 0.]], [[0., 2.]], [[100., 100.]]]])
    attn = torch.zeros(1, 1, 1, 6, 6)
    attn[..., :4, 4:] = 1
    attn[..., 4:, 4:] = torch.tensor([[.5, .5], [1., 0.]])
    result = cti_bgt_maps(cam, attn, torch.tensor([[1., 1., 0.]]), 1, 0)
    # Union [2,2] -> [2,2] -> constant normalizes to zero.
    # Reducing after propagation would give [1,2], which is not constant.
    assert torch.count_nonzero(result['foreground_union']) == 0


def test_complementary_maps_have_near_zero_loss_and_same_maps_do_not():
    fg = torch.tensor([[[[0., 1.]]]])
    bg = 1 - fg
    assert cti_bcam_loss({'foreground_union': fg, 'background': bg}) == 0
    assert cti_bcam_loss({'foreground_union': fg, 'background': fg}) == 1
    torch.testing.assert_close(cti_max_norm(torch.tensor([[[[-2., 1., 3.]]]])),
                               torch.tensor([[[[0., (1-1e-5)/(3+1e-5), (3-1e-5)/(3+1e-5)]]]]))


@pytest.mark.parametrize('value', [0., -1., 3.])
def test_constant_maps_empty_labels_finite_and_do_not_hide_unit_penalty(value):
    cam = torch.full((1, 3, 1, 2), value, requires_grad=True)
    attention = torch.ones(1, 1, 1, 5, 5, requires_grad=True)
    result = cti_bgt_maps(cam, attention, torch.zeros(1, 2), 1, 0)
    loss = cti_bcam_loss(result)
    assert loss.item() == 1.
    loss.backward()
    assert torch.isfinite(cam.grad).all() and torch.isfinite(attention.grad).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_low_precision_maps_accumulate_in_float32_with_finite_backward(dtype):
    torch.manual_seed(9)
    cam = torch.rand(1, 3, 2, 2, dtype=dtype, requires_grad=True)
    attention = torch.rand(2, 1, 2, 7, 7, dtype=dtype, requires_grad=True)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        maps = cti_bgt_maps(cam, attention, torch.ones(1, 2), 2, 0)
        loss = cti_bcam_loss(maps)
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(cam.grad).all() and torch.isfinite(attention.grad).all()


@pytest.mark.parametrize('override', [dict(bcss_variant='e1'), dict(psl_variant='read_only'),
    dict(attention_normalization='tgca'), dict(cti_bgt_weight=-1),
    dict(cti_bgt_weight=float('nan')), dict(cti_bgt_n_layers=0),
    dict(cti_bgt_n_layers=4), dict(cti_bgt_affinity_start=3)])
def test_invalid_or_confounded_configurations_fail(override):
    with pytest.raises(ValueError):
        small_model(**override)


def test_invalid_map_shapes_and_missing_labels_fail():
    cam = torch.ones(1, 3, 2, 2)
    attention = torch.ones(2, 1, 1, 7, 7)
    with pytest.raises(ValueError, match='labels'):
        cti_bgt_maps(cam, attention, torch.ones(1, 3), 2, 0)
    with pytest.raises(ValueError, match='attention shape'):
        cti_bgt_maps(cam, attention[..., :-1, :-1], None, 2, 0)
    with pytest.raises(ValueError, match='labels'):
        cti_bcam_loss(cti_bgt_maps(cam, attention, None, 2, 0))


def test_finetune_preserves_foreground_checkpoint_and_new_zero_bg():
    baseline, bgt = small_model(enabled=False), small_model()
    state = adapt_cti_bgt_finetune({'model': baseline.state_dict()}, bgt)
    missing = bgt.load_state_dict(state, strict=False)
    assert set(missing.missing_keys) == {'bg_token', 'pos_embed_bg'}
    assert not missing.unexpected_keys
    torch.testing.assert_close(bgt.head.weight[1:], baseline.head.weight)
    torch.testing.assert_close(bgt.head.bias[1:], baseline.head.bias)
    torch.testing.assert_close(bgt.cls_token, baseline.cls_token)
    assert torch.count_nonzero(bgt.bg_token) == 0
    cam = small_model(MCTformerPlusCam)
    cam.load_state_dict(adapt_cti_bgt_finetune({'model': bgt.state_dict()}, cam), strict=True)


def test_local_deit_finetune_repeats_only_foreground_tokens():
    model = small_model()
    state = dict(model.state_dict())
    for key in ('bg_token', 'pos_embed_bg', 'pos_embed_cls', 'pos_embed_pat'):
        state.pop(key)
    state['cls_token'] = torch.randn(1, 1, 24)
    state['head.weight'] = torch.randn(1000, 24)
    state['head.bias'] = torch.randn(1000)
    adapted = adapt_cti_bgt_finetune({'model': state}, model)
    assert adapted['cls_token'].shape == (1, 3, 24)
    assert adapted['pos_embed_cls'].shape == (1, 3, 24)
    assert adapted['pos_embed_pat'].shape == (1, 4, 24)
    assert 'head.weight' not in adapted and 'bg_token' not in adapted


def test_checkpoint_configuration_is_enforced():
    baseline, bgt = small_model(enabled=False), small_model()
    validate_cti_bgt_checkpoint({}, baseline)
    checkpoint = {'cti_bgt': bgt.cti_bgt_configuration()}
    validate_cti_bgt_checkpoint(checkpoint, bgt)
    with pytest.raises(ValueError, match='mismatch'):
        validate_cti_bgt_checkpoint(checkpoint, baseline)
    with pytest.raises(ValueError, match='mismatch'):
        validate_cti_bgt_checkpoint({}, bgt)
    checkpoint['cti_bgt']['n_layers'] = 1
    with pytest.raises(ValueError, match='mismatch'):
        validate_cti_bgt_checkpoint(checkpoint, bgt)


def test_training_and_cam_cli_default_off(monkeypatch):
    from train_model_v2 import get_args_parser as train_parser
    from make_cam import get_args_parser as cam_parser
    parser = train_parser()
    assert parser.parse_args([]).cti_bgt is False
    args = parser.parse_args(['--cti-bgt', '--cti-bgt-weight', '0'])
    assert args.cti_bgt and args.cti_bgt_weight == 0
    # Legacy CAM helper returns the parsed Namespace, not the parser.
    monkeypatch.setattr('sys.argv', ['make_cam.py'])
    assert cam_parser().cti_bgt is False
    monkeypatch.setattr('sys.argv', ['make_cam.py', '--cti-bgt', '--cti-bgt-weight', '0'])
    args = cam_parser()
    assert args.cti_bgt and args.cti_bgt_weight == 0


@pytest.mark.parametrize('weight', [0., .1])
def test_real_training_loop_adds_only_weighted_bcam(weight, monkeypatch):
    from engine import train_one_epoch_mctplus
    torch.manual_seed(113)
    model = small_model(cti_bgt_weight=weight)
    batch = (torch.randn(2, 3, 32, 32), torch.ones(2, 3))
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    def scaler(loss, optimizer, **kwargs):
        loss.backward()
        assert torch.isfinite(model.bg_token.grad).all()
        optimizer.step()
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    # The existing host loop unconditionally enables CUDA autocast; this CPU
    # integration test disables it without changing the production baseline.
    monkeypatch.setattr(torch.cuda.amp, 'autocast', lambda: torch.autocast('cpu', enabled=False))
    stats = train_one_epoch_mctplus(model, [batch], optimizer, torch.device('cpu'),
                                   0, scaler)
    expected = stats['mct_loss'] + stats['attn_loss'] + stats['pat_loss'] + weight * stats['cti_bcam_loss']
    assert stats['loss'] == pytest.approx(expected, abs=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_autocast_forward_backward(dtype):
    torch.manual_seed(29)
    model = small_model().cuda()
    with torch.no_grad():
        model.head.bias.fill_(1.)
    with torch.autocast('cuda', dtype=dtype, enabled=dtype != torch.float32):
        output = model(torch.randn(2, 3, 32, 32, device='cuda'),
                       active_labels=torch.ones(2, 3, device='cuda'))
        loss = cti_bcam_loss(output[3]['cti_bgt'])
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert model.bg_token.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('classes', [1, 20, 80])
def test_foreground_class_count_is_independent_of_token_count(classes):
    model = small_model(num_classes=classes)
    output = model(torch.randn(1, 3, 32, 32), active_labels=torch.ones(1, classes))
    assert output[0].shape == output[2].shape == (1, classes)
    assert output[1].shape[2] == classes
    assert output[3]['cti_bgt']['fused_cam'].shape == (1, classes + 1, 2, 2)
    baseline = small_model(num_classes=classes, enabled=False)
    # One token, one position vector, and one 3x3 convolution row + bias.
    delta = sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters())
    assert delta == 2 * model.embed_dim + 9 * model.embed_dim + 1
