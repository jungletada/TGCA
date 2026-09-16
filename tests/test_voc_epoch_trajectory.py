import json
from pathlib import Path

import pytest

from experiments.ablations.evaluate_voc_epoch_trajectory import (
    MATRIX, PANELS, cam_command, cleanup_predictions, inventory, read_result,
    summarize, training_ready, write_json,
)


def test_training_gate(tmp_path):
    assert not training_ready(tmp_path)
    (tmp_path/'QUEUE_COMPLETE').touch()
    assert training_ready(tmp_path)
    (tmp_path/'QUEUE_FAILED').touch()
    with pytest.raises(RuntimeError):
        training_ready(tmp_path)


def test_inventory_exact_180_and_epoch_order(tmp_path):
    for seed, method in MATRIX:
        root = tmp_path/f'{method}_s{seed}'
        folder = root/'epoch_checkpoints'; folder.mkdir(parents=True)
        (root/'RUN_COMPLETE').touch(); (root/'EPOCH_CHECKPOINTS_VERIFIED').touch()
        rows = []
        for epoch in range(45):
            name = f'mctformerplus_epoch_{epoch+1:03d}.pth'
            (folder/name).write_bytes(b'fixture')
            rows.append(dict(epoch=epoch, filename=name, bytes=7, sha256='fixture'))
        (folder/'index.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    jobs = inventory(tmp_path)
    assert len(jobs)==180 and len({j['checkpoint'] for j in jobs})==180
    assert [j['epoch'] for j in jobs[:8]]==[1]*4+[2]*4
    assert [(j['seed'], j['method']) for j in jobs[:4]]==MATRIX
    Path(jobs[-1]['checkpoint']).unlink()
    with pytest.raises(AssertionError):
        inventory(tmp_path)


def test_native_cam_flags(tmp_path):
    spec = dict(root=tmp_path/'VOC', cam=tmp_path/'train_id.txt', masks=tmp_path/'masks')
    for method in ('gwrp', 'all_product'):
        command = cam_command(spec, tmp_path/'out', dict(method=method, checkpoint='epoch.pth'))
        assert '--online-raw-eval' in command
        assert command[command.index('--scales')+1]=='1.0,0.75,1.25'
        assert command[command.index('--checkpoint')+1]=='epoch.pth'
        if method=='all_product':
            assert command[command.index('--c2p-pooling-layers')+1]=='all'
            assert command[command.index('--c2p-pooling-reduction')+1]=='product'
        assert '--c2p-pooling-affinity' not in command


def test_cleanup_requires_committed_result_and_preserves_checkpoints(tmp_path):
    (tmp_path/'classification').mkdir()
    prediction = tmp_path/'classification/classification_predictions.npz'
    prediction.write_bytes(b'predict')
    checkpoint = tmp_path/'checkpoint.pth'; checkpoint.write_bytes(b'keep')
    metric = tmp_path/'classification/classification_metrics.json'; metric.write_text('{}')
    with pytest.raises(AssertionError):
        cleanup_predictions(tmp_path)
    assert prediction.exists()
    write_json(tmp_path/'result.json', dict(validated=True))
    cleanup_predictions(tmp_path)
    assert not prediction.exists() and checkpoint.read_bytes()==b'keep' and metric.exists()
    cleanup_predictions(tmp_path)
    assert len(json.loads((tmp_path/'cleanup.json').read_text())['removed'])==1


def test_cleanup_refuses_symlink_escape(tmp_path):
    directory = tmp_path/'out'; (directory/'classification').mkdir(parents=True)
    outside = tmp_path/'outside.npz'; outside.write_bytes(b'keep')
    (directory/'classification/classification_predictions.npz').symlink_to(outside)
    write_json(directory/'result.json', {})
    with pytest.raises(ValueError):
        cleanup_predictions(directory)
    assert outside.read_bytes()==b'keep'


def test_result_coverage_and_identity(tmp_path):
    (tmp_path/'classification').mkdir(); (tmp_path/'raw_cam').mkdir()
    for name in ('classification/CLASSIFICATION_COMPLETE', 'raw_cam/EVAL_COMPLETE'):
        (tmp_path/name).touch()
    job = dict(method='gwrp', epoch=1, checkpoint_sha256='sha')
    cls = dict(num_images=1449, finite=True, checkpoint=dict(sha256='sha',epoch=0),
               metrics_percent=dict(class_token=dict(macro_class_ap=91),patch_gwrp=dict(macro_class_ap=90)),
               classification_loss=dict(class_token_multilabel_soft_margin_mean=.1,
                                        patch_gwrp_multilabel_soft_margin_mean=.2))
    point = dict(threshold=.45, mean_iou=.7,semantic_foreground_precision=.8,semantic_foreground_recall=.9)
    cam = dict(num_images=1464,fixed=point,best=point)
    write_json(tmp_path/'classification/classification_metrics.json',cls)
    write_json(tmp_path/'raw_cam/metrics.json',cam)
    row = read_result(tmp_path,job)
    assert row['fixed_mean_iou_percent']==70 and row['patch_macro_mAP_percent']==90
    with pytest.raises(AssertionError):
        read_result(tmp_path,dict(job,epoch=2))
    cam['num_images']=2;write_json(tmp_path/'raw_cam/metrics.json',cam)
    with pytest.raises(AssertionError):
        read_result(tmp_path,job)


def test_partial_curves_and_csv(tmp_path):
    rows = [dict(method=method,seed=seed,epoch=1,**{k:float(seed+1) for k,_ in PANELS})
            for seed,method in MATRIX]
    summarize(tmp_path,rows)
    assert len((tmp_path/'epoch_metrics.csv').read_text().splitlines())==5
    assert (tmp_path/'epoch_curves.svg').stat().st_size>1000
    assert (tmp_path/'epoch_curves.png').stat().st_size>1000
    assert '4/180' in (tmp_path/'EPOCH_TRAJECTORY_REPORT.md').read_text()
