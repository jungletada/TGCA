from argparse import Namespace
import json
import random

import numpy as np
import pytest
import torch

from tools.epoch_checkpoints import EpochCheckpointWriter, sha256


def toy():
    model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.3), torch.nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    return model, optimizer, scheduler, scaler


def run(tmp_path, save):
    torch.manual_seed(11); np.random.seed(11); random.seed(11)
    model, optimizer, scheduler, scaler = toy()
    writer = EpochCheckpointWriter(tmp_path, 'toy') if save else None
    for epoch in range(2):
        optimizer.zero_grad()
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward(); optimizer.step(); scheduler.step()
        if writer:
            writer.save(dict(model=model.state_dict(), epoch=epoch), optimizer, scheduler, scaler,
                        dict(loss=float(loss)), Namespace(seed=11), .7)
    return model, torch.rand(4), np.random.rand(4), random.random()


def test_saving_does_not_change_training_rng_or_weights(tmp_path):
    off = run(tmp_path/'off', False)
    on = run(tmp_path/'on', True)
    for key in off[0].state_dict():
        torch.testing.assert_close(off[0].state_dict()[key], on[0].state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(off[1], on[1], rtol=0, atol=0)
    np.testing.assert_array_equal(off[2], on[2])
    assert off[3] == on[3]
    files = sorted((tmp_path/'on/epoch_checkpoints').glob('*.pth'))
    assert [p.name for p in files] == ['toy_epoch_001.pth', 'toy_epoch_002.pth']
    rows = [json.loads(line) for line in (files[0].parent/'index.jsonl').read_text().splitlines()]
    for epoch, (path, row) in enumerate(zip(files, rows)):
        payload = torch.load(path, map_location='cpu')
        assert payload['epoch'] == epoch and payload['epoch_one_based'] == epoch+1
        assert payload['optimizer']['state'] and 'last_epoch' in payload['lr_scheduler']
        assert set(payload['rng_state']) == {'python', 'numpy', 'torch', 'cuda'}
        assert row['sha256'] == sha256(path)
    payload = torch.load(files[-1], map_location='cpu')
    restored, optimizer, scheduler, scaler = toy()
    restored.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['lr_scheduler'])
    scaler.load_state_dict(payload['scaler'])
    for key in restored.state_dict():
        torch.testing.assert_close(restored.state_dict()[key], on[0].state_dict()[key], rtol=0, atol=0)


def test_duplicate_snapshot_is_not_overwritten(tmp_path):
    model, optimizer, scheduler, scaler = toy()
    writer = EpochCheckpointWriter(tmp_path, 'toy')
    args = (dict(model=model.state_dict(), epoch=0), optimizer, scheduler, scaler, {}, Namespace(), 0.)
    path = writer.save(*args)
    digest = sha256(path)
    with pytest.raises(FileExistsError):
        writer.save(*args)
    assert sha256(path) == digest
    with pytest.raises(FileExistsError):
        EpochCheckpointWriter(tmp_path, 'toy')


def test_cli_opt_in():
    from train_model_v2 import get_args_parser
    parser = get_args_parser()
    assert parser.parse_args([]).save_every_epoch is False
    assert parser.parse_args(['--save-every-epoch']).save_every_epoch is True


def test_queue_matrix_and_recipe(tmp_path):
    from experiments.ablations.run_voc_epoch_checkpoints import MATRIX, training_command
    from experiments.baselines.run_default_voc_coco import dataset_spec
    assert MATRIX == [(0, 'gwrp'), (0, 'all_product'), (11, 'gwrp'), (11, 'all_product')]
    for seed, method in MATRIX:
        command = training_command(dataset_spec('VOC12'), tmp_path, method, seed)
        assert command[command.index('--seed')+1] == str(seed)
        assert command[command.index('--epochs')+1] == '45'
        assert '--save-every-epoch' in command
        assert '--c2p-pooling-affinity' not in command and '--probe' not in command
        if method == 'all_product':
            assert command[command.index('--c2p-pooling-layers')+1] == 'all'
            assert command[command.index('--c2p-pooling-reduction')+1] == 'product'
