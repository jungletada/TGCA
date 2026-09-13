import pytest
import torch
from models.mctformer_plus import (
    build_mctformerplus, model_spec_from_instance,
    validate_mctformerplus_patch_first_checkpoint,
)


def pair(cam=False, device='cpu'):
    torch.manual_seed(7)
    a = build_mctformerplus('small', cam=cam, input_size=32, num_classes=20).to(device)
    torch.manual_seed(7)
    b = build_mctformerplus('small', cam=cam, input_size=32, num_classes=20,
                           patch_first=True).to(device)
    assert a.state_dict().keys() == b.state_dict().keys()
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    return a, b


def test_disabled_flag_exact_baseline():
    a, b = pair()
    b.patch_first = False
    a.eval(); b.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        for first, second in zip(a(x), b(x)):
            torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_block_layout_raw_cct_and_feature_equivariance():
    a, b = pair()
    a.eval(); b.eval()
    x = torch.randn(2, 3, 32, 32)
    before, after = [], []
    handles = []
    for block in b.blocks:
        handles.append(block.register_forward_pre_hook(lambda m, args: before.append(args[0].detach())))
        handles.append(block.register_forward_hook(lambda m, args, out: after.append(out[0].detach())))
    with torch.no_grad():
        fa, fb = a.forward_features(x), b.forward_features(x)
    for h in handles:
        h.remove()
    patch_input = b.patch_embed(x) + b.interpolate_pos_encoding(b.patch_embed(x), 32, 32)
    torch.testing.assert_close(before[0][:, :4], patch_input, rtol=0, atol=0)
    torch.testing.assert_close(before[0][:, 4:], (b.cls_token + b.pos_embed_cls).expand(2, -1, -1))
    for i in range(12):
        torch.testing.assert_close(fb[3][i], after[i][:, 4:], rtol=0, atol=0)
        if i:
            torch.testing.assert_close(before[i], after[i-1], rtol=0, atol=0)
        torch.testing.assert_close(fa[3][i], fb[3][i], atol=3e-6, rtol=3e-5)
    for i in (0, 1):
        torch.testing.assert_close(fa[i], fb[i], atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('shape', [(32, 32), (32, 48)])
def test_native_cam_and_attention_equivalence(shape):
    a, b = pair(cam=True)
    a.eval(); b.eval()
    x = torch.randn(2, 3, *shape)
    for kwargs in ({}, {'return_attn': True}):
        torch.testing.assert_close(a(x, **kwargs), b(x, **kwargs), atol=3e-5, rtol=5e-5)
    for left, right in zip(a.forward_with_label(x), b.forward_with_label(x)):
        torch.testing.assert_close(left, right, atol=3e-5, rtol=5e-5)


def test_gradient_equivariance():
    a, b = pair()
    x = torch.randn(2, 3, 32, 32)
    for model in (a, b):
        values = model(x)
        sum(v.square().mean() for v in values).backward()
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        if p.grad is not None:
            assert q.grad is not None and torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_amp_forward_backward():
    model = build_mctformerplus('small', input_size=448, num_classes=20,
                               patch_first=True).cuda().train()
    with torch.autocast('cuda', dtype=torch.float16):
        output = model(torch.randn(2, 3, 448, 448, device='cuda'))
        loss = sum(t.float().square().mean() for t in output)
    loss.backward()
    assert torch.isfinite(loss)
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


def test_metadata_and_incompatible_options():
    a, b = pair()
    validate_mctformerplus_patch_first_checkpoint({}, False)
    validate_mctformerplus_patch_first_checkpoint({'model_spec': model_spec_from_instance(b)}, True)
    with pytest.raises(ValueError, match='patch_first'):
        validate_mctformerplus_patch_first_checkpoint({'model_spec': model_spec_from_instance(a)}, True)
    with pytest.raises(ValueError, match='patch_first'):
        build_mctformerplus('small', patch_first=True, patch_final_norm=True)
