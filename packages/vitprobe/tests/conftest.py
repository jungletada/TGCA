import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_cpu():
    previous = torch.get_num_threads()
    state = torch.get_rng_state()
    torch.set_num_threads(2)
    torch.manual_seed(20260916)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(previous)
