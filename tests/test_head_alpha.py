import numpy as np
import torch

from analysis.alpha_sweep import blend,native_blend,ALPHAS
from analysis.head_heatmap import head_statistics
from analysis.head_alpha_statistics import stratified_hash_folds,paired_ci,crossfit,miou,sufficient
from analysis.head_alpha import seed_for,propagate,normalize,extract,configs_for,alpha_crossfit_summary
from models.mctformer_plus import build_mctformerplus


def test_alpha_endpoints_floor_and_monotonicity():
    m=torch.tensor([-2.,0.,.5,4.])
    a=torch.tensor([0.,1e-10,2.,1.])
    assert torch.equal(blend(m,a,0),m.relu())
    assert torch.equal(blend(m,a,1),a)
    torch.testing.assert_close(blend(m,a,.5)[2:],native_blend(m,a)[2:],atol=1e-6,rtol=1e-6)
    values=torch.stack([blend(m,a,t) for t in ALPHAS])
    assert (values[1:,2]>=values[:-1,2]).all()
    assert (values[1:,3]<=values[:-1,3]).all()
    assert blend(m,a,.5)[1]>native_blend(m,a)[1] # Floor is disclosed, not hidden.


def test_head_stats_uniform_distinct_single_label():
    a=torch.ones(2,2,3,4)
    labels=torch.tensor([[1,0,0],[1,1,0]])
    a[1,1,0]=torch.tensor([1.,0,0,0]);a[1,1,1]=torch.tensor([0.,1,0,0])
    out=head_statistics(a,labels)
    torch.testing.assert_close(out[:,0,0],torch.ones(2))
    torch.testing.assert_close(out[:,0,1],torch.zeros(2))
    assert torch.isnan(out[0,:,2]).all()
    assert out[1,0,2]==1 and out[1,1,2]==0
    assert out[1,1,0]==.25 and out[1,1,1]==.75


def test_native_head_extraction_and_propagation_equivalence():
    torch.manual_seed(2)
    m=build_mctformerplus('small',input_size=32,num_classes=20,cam=True).eval()
    x=torch.randn(2,3,32,32)
    positive=torch.tensor([0,3,9])
    with torch.no_grad():
        maps,a,p,cells,grid,allmaps,means=extract(m,x,positive)
        seeds=seed_for(dict(kind='native'),maps,a,cells)
        actual=propagate(seeds[None],p)[0].reshape(2,3,*grid)
        torch.testing.assert_close(actual,m.get_cam(allmaps,means)[:,positive],atol=1e-6,rtol=1e-6)
        mean_all=cells.mean(2).mean(0)
        top=seed_for(dict(kind='topm',cells=list(range(72))),maps,a,cells)
        torch.testing.assert_close(top,native_blend(maps,mean_all),atol=1e-6,rtol=1e-6)
        for l,h in [(0,0),(11,5)]:
            candidate=seed_for(dict(kind='head',layer=l,head=h),maps,a,cells)
            torch.testing.assert_close(candidate,native_blend(maps,cells[l,:,h]),rtol=0,atol=0)


def test_propagation_orientation_and_multiview_row_scaling():
    p=torch.tensor([[[1.,2.],[3.,4.]]])
    s=torch.tensor([[[[5.,6.]]]])
    torch.testing.assert_close(propagate(s,p),torch.tensor([[[[17.,39.]]]]))
    a=torch.tensor([[[0.,1.],[2.,1.]]])
    b=torch.tensor([[[2.,0.],[0.,1.]]])
    assert not torch.allclose(normalize(a+b),normalize(2*a+b))


def test_folds_reproducible_balanced_and_order_independent():
    ids=[f'img{i}' for i in range(100)]
    y=np.random.default_rng(0).integers(0,2,(100,6))
    f=stratified_hash_folds(ids,y)
    assert (f==0).sum()==50
    np.testing.assert_array_equal(f,stratified_hash_folds(ids,y))
    order=np.random.default_rng(9).permutation(100)
    other=stratified_hash_folds([ids[i] for i in order],y[order])
    np.testing.assert_array_equal(other,f[order])
    assert abs(y[f==0].sum(0)-y[f==1].sum(0)).max()<=4


def toy_confusions():
    rng=np.random.default_rng(5)
    x=rng.integers(0,20,(3,10,21,21))
    x[1]=x[0]
    return x


def test_paired_bootstrap_identical_and_miou_sufficient():
    from tools.evaluate_cam_threshold_grid import confusion_metrics
    x=toy_confusions()
    np.testing.assert_allclose(miou(sufficient(x.sum(1))),confusion_metrics(x.sum(1))['mean_iou'])
    absolute,delta=paired_ci(x,reps=120)
    np.testing.assert_allclose(delta[:,:2],0)
    assert absolute.shape==(2,3)


def test_crossfit_selection_heldout_and_identical_ci():
    x=np.repeat(np.eye(21,dtype=np.int32)[None,None],3,axis=0)
    x=np.repeat(x,10,axis=1)*10
    folds=np.array([0]*5+[1]*5)
    r=crossfit(x,folds,reps=120)
    assert r['selected_candidate_indices']==[1,1]
    assert r['point']==1 and r['delta']==0
    np.testing.assert_allclose(np.array(r['ci95'])[:,2],0)
    x[1,5:,0,1]=100
    x[2,:5,0,1]=100
    r=crossfit(x,folds,reps=120)
    assert r['selected_candidate_indices']==[1,2]
    assert r['delta']<0 # Picks on opposite folds; cannot evaluate on selection half.


def test_configs_no_trainable_or_oracle_topm():
    assert len(configs_for('alpha'))==22 and len(configs_for('head'))==73
    ranking=dict(kappa=list(range(72)),gamma=list(reversed(range(72))))
    top=configs_for('topm',ranking)
    assert len(top)==15 and top[1]['cells']==[0] and top[8]['cells']==[71]


def test_alpha_crossfit_metadata_preserves_numeric_reference():
    r=alpha_crossfit_summary(toy_confusions(),np.arange(10)%2,configs_for('alpha')[:3],reps=120)
    assert isinstance(r['reference'],float) and isinstance(r['reference_description'],str)
    assert len(r['selected_alphas'])==2
