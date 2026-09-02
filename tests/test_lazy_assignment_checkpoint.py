from argparse import Namespace
from pathlib import Path

import pytest
import torch

from analysis.lazy_assignment.run_class_specific_patch_score import (
    create_frozen_model,
    extract_state_dict,
    load_state_dict_strict,
)


def test_checkpoint_loader_accepts_raw_wrapped_and_uniform_module_prefix():
    torch.manual_seed(3)
    reference = torch.nn.Linear(4, 2)
    raw = reference.state_dict()

    for payload in (
        raw,
        {"model": raw},
        {"model": {f"module.{key}": value for key, value in raw.items()}},
    ):
        target = torch.nn.Linear(4, 2)
        info = load_state_dict_strict(target, payload)
        torch.testing.assert_close(target.weight, reference.weight)
        torch.testing.assert_close(target.bias, reference.bias)
        assert info["missing_keys"] == []
        assert info["unexpected_keys"] == []


def test_checkpoint_loader_rejects_mixed_module_prefixes():
    state = torch.nn.Linear(4, 2).state_dict()
    mixed = {"module.weight": state["weight"], "bias": state["bias"]}
    with pytest.raises(ValueError, match="mixes"):
        extract_state_dict(mixed)


CHECKPOINT = Path(
    "results/mctformerplus/voc/"
    "20260826-mctformerplus-voc-vanilla-s0-22427d6/"
    "mctformerplus_final.pth"
)


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="LHR result checkpoint unavailable")
def test_lhr_mctformerplus_checkpoint_strict_loads_into_current_host():
    payload = torch.load(CHECKPOINT, map_location="cpu")
    args = Namespace(model="mctformerplus", input_size=448)
    model, configuration, info = create_frozen_model(args, payload)
    assert model.__class__.__name__ == "MCTformerPlusCam"
    assert configuration["attention"]["mode"] == "vanilla"
    assert configuration["bcss"].get("variant", "e0") == "e0"
    assert info["missing_keys"] == []
    assert info["unexpected_keys"] == []
