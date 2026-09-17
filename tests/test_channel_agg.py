from pathlib import Path
import pytest
import torch
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from models.channel_aggregation import ChannelAggregator, configure_channel_optimizer, scale_channel_lr
from models.mctformer_plus import (build_mctformerplus, adapt_deit_checkpoint_for_mctformerplus,
                                  model_spec_from_instance, validate_detach_channel_checkpoint)
from train_model_v2 import get_args_parser


def test_initial_mean_exact_and_nonuniform_reference_gradient():
    a=ChannelAggregator(384,enabled=True)
    t=torch.randn(2,20,384,requires_grad=True)
    torch.testing.assert_close(a(t),t.mean(-1),atol=0,rtol=0)
    assert torch.equal(ChannelAggregator(384)(t),t.mean(-1))
    with torch.no_grad():a.theta.normal_()
    w=a.weights()
    assert (w>0).all()
    torch.testing.assert_close(w.sum(),torch.tensor(1.))
    torch.testing.assert_close(a(t),(t*w).sum(-1),atol=1e-7,rtol=1e-6)
    a(t).square().sum().backward()
    assert a.theta.grad.abs().sum()>0 and t.grad.abs().sum()>0


@pytest.mark.parametrize('mult',[1.,10.])
def test_theta_no_decay_and_exact_lr_ratio(mult):
    args=get_args_parser().parse_args([]);args.lr=3.125e-5
    m=build_mctformerplus('small',input_size=32,num_classes=20,channel_agg=True)
    opt=create_optimizer(args,m)
    configure_channel_optimizer(opt,m,mult)
    sched,_=create_scheduler(args,opt)
    scale_channel_lr(opt)
    theta=m.channel_aggregator.theta
    group=next(g for g in opt.param_groups if any(p is theta for p in g['params']))
    assert group['weight_decay']==0 and 'channel_aggregator.theta' in m.no_weight_decay()
    assert sum(p is theta for g in opt.param_groups for p in g['params'])==1
    for epoch in [0,4,10,44]:
        sched.step(epoch);scale_channel_lr(opt)
        assert group['lr']==pytest.approx(opt.param_groups[0]['lr']*mult)


def test_model_baseline_rng_cct_and_cam_unchanged():
    torch.manual_seed(5)
    baseline=build_mctformerplus('small',input_size=32,num_classes=20).eval()
    torch.manual_seed(5)
    m=build_mctformerplus('small',input_size=32,num_classes=20,channel_agg=True).eval()
    assert set(m.state_dict())-set(baseline.state_dict())=={'channel_aggregator.theta'}
    for key,v in baseline.state_dict().items():assert torch.equal(v,m.state_dict()[key])
    x=torch.randn(2,3,32,32)
    with torch.no_grad():
        a,b=baseline(x),m(x)
        for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=0,rtol=0)
        m.channel_aggregator.theta.normal_()
        c=m(x)
        assert not torch.equal(c[0],a[0])
        torch.testing.assert_close(c[1],a[1],atol=0,rtol=0)
        torch.testing.assert_close(c[2],a[2],atol=0,rtol=0)
    cam=build_mctformerplus('small',cam=True,input_size=32,num_classes=20,channel_agg=True).eval()
    old=build_mctformerplus('small',cam=True,input_size=32,num_classes=20).eval()
    cam.load_state_dict(m.state_dict());old.load_state_dict(baseline.state_dict())
    torch.testing.assert_close(cam(x),old(x),atol=0,rtol=0)
    cls,_,_=cam.forward_with_label(x)
    torch.testing.assert_close(cls,(c[0]>0).float(),atol=0,rtol=0)
    checkpoint=dict(model=m.state_dict(),model_spec=model_spec_from_instance(m))
    validate_detach_channel_checkpoint(checkpoint,channel_agg=True)
    with pytest.raises(ValueError):validate_detach_channel_checkpoint(checkpoint)


def test_deit_theta_zero_and_same_initial_weights():
    path=Path('/home/peng/.cache/torch/hub/checkpoints/deit_small_patch16_224-cd65a155.pth')
    if not path.exists():pytest.skip('Local DeiT checkpoint required')
    m=build_mctformerplus('small',input_size=32,num_classes=20,channel_agg=True)
    adapted,report=adapt_deit_checkpoint_for_mctformerplus(torch.load(path,map_location='cpu'),m)
    m.load_state_dict(adapted,strict=True)
    assert torch.count_nonzero(m.channel_aggregator.theta)==0
    assert report['zero_initialized_keys']==['channel_aggregator.theta']
    assert report['randomly_initialized_keys']==['head.bias','head.weight']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_channel_amp():
    m=build_mctformerplus('small',input_size=32,num_classes=20,channel_agg=True).cuda()
    with torch.autocast('cuda',dtype=torch.float16):
        out=m(torch.randn(2,3,32,32,device='cuda'))
        loss=out[0].square().mean()+out[2].square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert m.channel_aggregator.theta.grad.abs().sum()>0
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
