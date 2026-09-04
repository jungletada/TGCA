from __future__ import annotations

import copy

import pytest
import torch

from engine import (
    FixedLengthBatchSampler,
    _accumulation_backward_step,
    accumulation_spec,
    linear_scaled_learning_rate,
)


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_count = 0
        self.zero_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)

    def zero_grad(self, *args, **kwargs):
        self.zero_count += 1
        return super().zero_grad(*args, **kwargs)


def one_update(model, inputs, targets, micro_batch, accum_iter):
    optimizer = CountingSGD(model.parameters(), lr=0.05)
    optimizer.zero_grad()
    for index in range(accum_iter):
        begin = index * micro_batch
        end = begin + micro_batch
        prediction = model(inputs[begin:end])
        loss = torch.nn.functional.mse_loss(prediction, targets[begin:end])
        boundary = index + 1 == accum_iter
        _accumulation_backward_step(
            loss / accum_iter,
            optimizer,
            model.parameters(),
            loss_scaler=None,
            boundary=boundary,
            max_norm=None,
        )
        if boundary:
            optimizer.zero_grad()
    return optimizer


def test_microbatch_accumulation_matches_full_batch_update():
    torch.manual_seed(7)
    initial = torch.nn.Sequential(
        torch.nn.Linear(5, 8), torch.nn.GELU(), torch.nn.Linear(8, 3)
    )
    full = copy.deepcopy(initial)
    accumulated = copy.deepcopy(initial)
    inputs = torch.randn(4, 5)
    targets = torch.randn(4, 3)
    full_optimizer = one_update(full, inputs, targets, 4, 1)
    accumulated_optimizer = one_update(accumulated, inputs, targets, 2, 2)
    assert full_optimizer.step_count == accumulated_optimizer.step_count == 1
    assert full_optimizer.zero_count == accumulated_optimizer.zero_count == 2
    for left, right in zip(full.parameters(), accumulated.parameters()):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ('micro_batch', 'accum_iter'),
    ((32, 1), (16, 2), (8, 4), (4, 8), (2, 16)),
)
def test_effective_batch_update_and_sample_contract(micro_batch, accum_iter):
    spec = accumulation_spec(10582, micro_batch, accum_iter, world_size=1)
    assert spec['effective_batch_size'] == 32
    assert spec['optimizer_updates_per_epoch'] == 330
    assert spec['micro_batches_per_epoch'] == 330 * accum_iter
    assert spec['consumed_samples_per_epoch_global'] == 10560
    assert spec['discarded_samples_per_epoch_global'] == 22
    assert linear_scaled_learning_rate(5e-4, spec['effective_batch_size']) == 3.125e-5


def test_fixed_length_batch_sampler_stops_at_registered_boundary():
    sampler = torch.utils.data.SequentialSampler(range(10582))
    batches = torch.utils.data.BatchSampler(sampler, batch_size=8, drop_last=True)
    limited = FixedLengthBatchSampler(batches, num_batches=1320)
    materialized = list(limited)
    assert len(limited) == len(materialized) == 1320
    assert sum(len(batch) for batch in materialized) == 10560
    assert materialized[-1][-1] == 10559


def test_invalid_accumulation_contracts_fail():
    with pytest.raises(ValueError):
        accumulation_spec(10, 0, 1)
    with pytest.raises(ValueError):
        accumulation_spec(10, 8, 2)
    with pytest.raises(ValueError):
        linear_scaled_learning_rate(5e-4, 0)
