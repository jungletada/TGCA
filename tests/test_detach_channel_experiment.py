import json
import math
import torch
from argparse import Namespace

from analysis.weight_stats import normalized_entropy, all_product_weights, positive_image_mean
from analysis.detach_channel_diagnostics import classification_command, discard_classification_predictions
from experiments.ablations.run_detach_channel import MATRIX, gates, checkpoint_job
from models.mctformer_plus import build_mctformerplus


def test_weight_statistics_are_native_all_product():
    records=[torch.rand(2,6,24,24) for _ in range(12)]
    m=build_mctformerplus('small',input_size=32,num_classes=20,patch_pooling='c2p',
                         c2p_pooling_layers='all',c2p_pooling_reduction='product')
    torch.testing.assert_close(all_product_weights(records,20),m.c2p_spatial_weights(records,4),atol=0,rtol=0)
    torch.testing.assert_close(normalized_entropy(torch.full((2,20,4),.25)),torch.ones(2,20))
    onehot=torch.tensor([[1.,0,0,0]])
    assert normalized_entropy(onehot).item()==0
    labels=torch.tensor([[1,1,0],[1,0,0]])
    torch.testing.assert_close(positive_image_mean(torch.tensor([[1.,3,999],[10.,999,999]]),labels),torch.tensor([2.,10.]))


def test_registered_gates_no_threshold_retuning():
    old=[dict(method='gwrp',seed=s,fixed_mean_iou_percent=m) for s,m in [(0,70.063),(11,68.632)]]
    def new(d0,d11,a0=70.563,a11=69.132):
        return [dict(method=k,seed=s,fixed_mean_iou_percent=v) for k,s,v in
                [('detach',0,d0),('detach',11,d11),('a1',0,a0),('a1',11,a11)]]
    g=gates(new(71.1,68.1),old)
    assert g['E1']=='repair_gate_passed_requires_more_seeds'
    assert gates(new(71.099,68.1),old)['E1']=='s11_recovered_s0_gain_not_retained'
    assert gates(new(71.1,68.099),old)['E1']=='s11_not_recovered_fallback_candidate'
    assert gates(new(60,60),old)['E1']=='both_below_V0_stop'
    assert gates(new(71.1,68.1,72,66),old)['E4']=='mixed_seed_signs'
    assert gates(new(71.1,68.1,69,67),old)['E4']=='both_negative'
    assert MATRIX==[('detach',0),('detach',11),('a1',0),('a1',11)]


def test_checkpoint_job_and_classification_flags(tmp_path):
    folder=tmp_path/'epoch_checkpoints';folder.mkdir()
    entry=dict(epoch=0,filename='mctformerplus_epoch_001.pth',sha256='digest')
    (folder/'index.jsonl').write_text(json.dumps(entry)+'\n')
    for variant in ['gwrp','all_product','detach','a1']:
        job=checkpoint_job(tmp_path,variant,11,1)
        command=classification_command(job['checkpoint'],tmp_path/'out',variant)
        assert ('--detach-weights' in command)==(variant=='detach')
        assert ('--channel-agg' in command)==(variant=='a1')
        assert command[command.index('--bootstrap-resamples')+1]=='0'
        assert job['epoch']==1 and job['seed']==11


def test_warning_cleanup_only_after_metrics(tmp_path):
    prediction=tmp_path/'classification_predictions.npz';prediction.write_bytes(b'generated')
    (tmp_path/'CLASSIFICATION_COMPLETE').touch()
    (tmp_path/'classification_metrics.json').write_text(json.dumps(dict(finite=True)))
    source=tmp_path/'checkpoint.pth';source.write_bytes(b'keep')
    discard_classification_predictions(tmp_path)
    assert source.read_bytes()==b'keep' and not prediction.exists()
    assert (tmp_path/'classification_metrics.json').is_file()


def test_epoch20_warning_is_macro_ap_only_and_does_not_stop(monkeypatch,tmp_path):
    import analysis.detach_channel_diagnostics as diag
    def fake_run(command,root,stage):
        out=root/'classification';out.mkdir()
        (out/'CLASSIFICATION_COMPLETE').touch()
        (out/'classification_metrics.json').write_text(json.dumps(dict(
            finite=True,metrics_percent=dict(class_token=dict(macro_class_ap=77.41,legacy_mean_image_ap=99.)))))
        (out/'classification_predictions.npz').write_bytes(b'disposable')
    monkeypatch.setattr(diag,'run',fake_run)
    state=torch.get_rng_state().clone()
    diag.epoch20_warning(Namespace(detach_weights=True,work_space=str(tmp_path)),tmp_path/'epoch020.pth')
    assert torch.equal(state,torch.get_rng_state())
    record=json.loads((tmp_path/'epoch20_warning/warning.json').read_text())
    assert record['suspected_instability'] and record['epoch']==20
    assert record['class_macro_mAP_percent']==77.41
    assert record['action']=='warn only; continue to epoch45'
