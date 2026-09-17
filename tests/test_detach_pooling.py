import pytest
import torch
from models.mctformer_plus import build_mctformerplus
from analysis.diagnostics.metrics import gwrp_weights


def model(detach=False, cam=False):
    return build_mctformerplus('small', cam=cam, input_size=32, num_classes=20,
                              patch_pooling='c2p', c2p_pooling_layers='all',
                              c2p_pooling_reduction='product', detach_weights=detach)


def test_detach_exact_forward_and_weight_path_gradients():
    m = model()
    records = [torch.rand(2,6,24,24,requires_grad=True) for _ in range(12)]
    logits = torch.randn(2,20,2,2,requires_grad=True)
    actual = m.c2p_pool(logits,records)
    weights = m.c2p_spatial_weights(records,4)
    torch.testing.assert_close(actual,(weights*logits.flatten(2)).sum(-1),atol=0,rtol=0)
    actual.square().sum().backward()
    assert all(a.grad is not None and a.grad.abs().sum()>0 for a in records)
    for a in records: a.grad=None
    logits.grad=None
    m.detach_weights=True
    detached=m.c2p_pool(logits,records)
    torch.testing.assert_close(actual,detached,atol=0,rtol=0)
    detached.square().sum().backward()
    assert all(a.grad is None for a in records)
    assert logits.grad.abs().sum()>0


def test_gwrp_weight_path_is_zero_but_value_path_differentiable():
    m=build_mctformerplus('small',input_size=32,num_classes=20)
    logits=torch.randn(2,20,2,2,requires_grad=True)
    weights=gwrp_weights(logits.flatten(2),m.decay_parameter)
    assert not weights.requires_grad and weights.grad_fn is None
    native=m.gwrp(logits)
    equivalent=(weights*logits.flatten(2)).sum(-1)
    torch.testing.assert_close(native,equivalent,atol=1e-7,rtol=1e-6)
    native.sum().backward()
    torch.testing.assert_close(logits.grad.flatten(2),weights,atol=1e-7,rtol=1e-6)
    # Zero derivative of rank-assigned weights is an almost-everywhere claim;
    # it does not say the backbone attention has zero total gradient.


def test_actual_detach_model_backbone_and_cam():
    baseline=model().eval(); detached=model(True).eval()
    detached.load_state_dict(baseline.state_dict(),strict=True)
    x=torch.randn(2,3,32,32)
    a,b=baseline(x),detached(x)
    for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=0,rtol=0)
    b[2].square().mean().backward()
    assert detached.head.weight.grad.abs().sum()>0
    assert detached.blocks[0].attn.qkv.weight.grad.abs().sum()>0
    cam1,cam2=model(cam=True).eval(),model(True,cam=True).eval()
    cam1.load_state_dict(baseline.state_dict());cam2.load_state_dict(baseline.state_dict())
    torch.testing.assert_close(cam1(x),cam2(x),atol=0,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_detach_amp():
    m=model(True).cuda()
    with torch.autocast('cuda',dtype=torch.float16):
        out=m(torch.randn(2,3,32,32,device='cuda'))
        loss=out[0].square().mean()+out[2].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
