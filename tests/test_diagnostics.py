"""CPU-first tests; no production dataset, checkpoint, or active GPU required."""

import copy
import csv
import json
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image
import pytest
import torch

from analysis.diagnostics import ProbeConfig, TokenProbe
from analysis.diagnostics.metrics import (class_token_affinity, conditional_attention, effective_support,
                                         gini, gwrp_weights, spearman, js_divergence)
from analysis.diagnostics.probe_set import build_sets, mask_transform, stratified_indices
from analysis.diagnostics.spectral import spectrum_and_autocorrelation, correlation_length, high_freq_ratio
from analysis.diagnostics.state import isolated_eval
from analysis.diagnostics.writer import REPO, result_path, state_sha256
from models.mctformer_plus import MCTformerPlus, MCTformerPlusCam


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def workspace():
    # Test artifacts obey the same results-only boundary, then self-clean.
    with tempfile.TemporaryDirectory(prefix='diagnostics_test_', dir=REPO / 'results') as path:
        yield Path(path)


def small_model(cam=False, **kwargs):
    cls = MCTformerPlusCam if cam else MCTformerPlus
    return cls(input_size=32, img_size=32, patch_size=16, embed_dim=24,
               depth=12, num_heads=3, num_classes=3, mlp_ratio=1,
               drop_path_rate=.1, **kwargs)


def data_config(workspace, mini=True, **kwargs):
    root = workspace / 'data'
    (root / 'JPEGImages').mkdir(parents=True)
    (root / 'ImageLabel').mkdir()
    (root / 'masks').mkdir()
    ids = ['one', 'two', 'three', 'four']
    labels = {name: np.asarray(y, np.float32) for name, y in zip(ids, [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0]])}
    rng = np.random.default_rng(8)
    for name in ids:
        Image.fromarray(rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)).save(root / 'JPEGImages' / f'{name}.jpg')
        mask = np.zeros((64, 80), np.uint8)
        mask[8:48, 10:50] = 1
        mask[:2] = 255
        Image.fromarray(mask).save(root / 'masks' / f'{name}.png')
    np.save(root / 'ImageLabel/cls_labels.npy', labels)
    probe = workspace / 'probe_train.txt'; probe.write_text('one\ntwo\n')
    mini_path = workspace / 'mini_train.txt'; mini_path.write_text('three\nfour\n')
    return ProbeConfig(enabled=True, probe_set_path=str(probe), probe_batch_size=2,
                       res_semantic=32, res_spectral=448, data_root=str(root),
                       out_dir=str(workspace / 'diagnostics'), minimum_per_class=0,
                       mini_cam_set_path=str(mini_path) if mini else '',
                       mini_cam_mask_dir=str(root / 'masks') if mini else '', **kwargs)


def test_uniform_and_onehot():
    n = 8
    uniform = torch.full((2, 3, n), 1 / n)
    onehot = torch.zeros_like(uniform); onehot[..., 0] = 1
    torch.testing.assert_close(effective_support(uniform), torch.ones(2, 3))
    torch.testing.assert_close(gini(uniform), torch.zeros(2, 3))
    torch.testing.assert_close(effective_support(onehot), torch.full((2, 3), 1 / n))
    torch.testing.assert_close(gini(onehot), torch.full((2, 3), (n - 1) / n))
    assert torch.isnan(effective_support(conditional_attention(torch.zeros_like(uniform)))).all()
    torch.testing.assert_close(conditional_attention(uniform * 1e-20), uniform)


def test_gini_direct_and_divergence_ties():
    x = conditional_attention(torch.rand(2, 3, 9))
    direct = (x[..., :, None] - x[..., None, :]).abs().sum((-1, -2)) / (2 * 9)
    torch.testing.assert_close(gini(x), direct)
    a = torch.tensor([[0., 0., 1., 2.]])
    torch.testing.assert_close(spearman(a, a), torch.ones(1))
    torch.testing.assert_close(spearman(a, -a), -torch.ones(1))
    assert torch.isnan(spearman(a, torch.ones_like(a))).all()
    torch.testing.assert_close(js_divergence(x, x), torch.zeros(2, 3))
    from scipy.stats import spearmanr
    b = torch.tensor([[2., 1., 1., 0.]])
    assert float(spearman(a, b)) == pytest.approx(spearmanr(a[0], b[0])[0])


def test_class_pairs_missing_and_input_dependent():
    tokens = torch.eye(3)[None].repeat(2, 1, 1)
    labels = torch.tensor([[1, 0, 0], [1, 1, 0]])
    d = class_token_affinity(tokens, labels)
    assert torch.isnan(d['rho_cc'][0])
    assert d['rho_cc'][1] == 0
    assert torch.isnan(d['rho_cc_per_class'][0]).all()


@pytest.mark.parametrize('ties', [False, True])
def test_gwrp_native_elementwise_and_pooling(ties):
    m = small_model()
    maps = torch.randn(2, 3, 4, 5)
    if ties:
        maps = maps.round()
    w = gwrp_weights(maps.flatten(2), m.decay_parameter)
    sorted_values, order = maps.flatten(2).transpose(1, 2).sort(dim=1, descending=True)
    native = torch.logspace(0, 19, 20, base=m.decay_parameter)
    native = native / native.sum()
    actual = w.transpose(1, 2).gather(1, order)
    torch.testing.assert_close(actual, native[None, :, None].expand_as(actual), atol=0, rtol=0)
    torch.testing.assert_close((w * maps.flatten(2)).sum(-1), m.gwrp(maps), atol=1e-7, rtol=1e-6)


def test_spectral_white_constant_and_fit_quality():
    torch.manual_seed(4)
    p, _, _, ac = spectrum_and_autocorrelation(torch.randn(12, 28, 28, 48))
    assert .42 < float(high_freq_ratio(p).mean()) < .58
    # Demeaning removes constants; a zero spectrum has undefined energy ratio.
    zero, _, _, ac_zero = spectrum_and_autocorrelation(torch.ones(2, 28, 28, 8))
    assert torch.count_nonzero(zero) == 0
    assert torch.isnan(high_freq_ratio(zero)).all()
    assert not correlation_length(ac_zero)['xi_valid'].any()
    r = torch.arange(15).float()
    fit = correlation_length(torch.exp(-r[None] / 2.5))
    torch.testing.assert_close(fit['xi'], torch.tensor([2.5]))
    torch.testing.assert_close(fit['xi_r2'], torch.ones(1))
    invalid = torch.ones(1, 15); invalid[:, 3] = -1
    assert torch.isnan(correlation_length(invalid)['xi']).all()
    with pytest.raises(ValueError, match='grid'):
        spectrum_and_autocorrelation(torch.randn(1, 14, 14, 3))


def test_spatial_frequency_separation():
    yy, xx = torch.meshgrid(torch.arange(28), torch.arange(28), indexing='ij')
    def field(k):
        wave = torch.sin(2 * torch.pi * k * xx / 28)
        return torch.stack((wave, torch.ones_like(wave)), -1)[None]
    low, *_ = spectrum_and_autocorrelation(field(1))
    high, *_ = spectrum_and_autocorrelation(field(12))
    assert high_freq_ratio(low).item() < high_freq_ratio(high).item()


def test_fixed_sampling_disjoint_and_masks(workspace):
    y = np.tile(np.eye(3, dtype=int), (10, 1))
    a = stratified_indices(y, size=12, minimum=3)
    assert (y[a].sum(0) >= 3).all()
    np.testing.assert_array_equal(a, stratified_indices(y, size=12, minimum=3))
    cfg = data_config(workspace)
    train = workspace / 'train.txt'; train.write_text('one\ntwo\nthree\nfour\n')
    probe, mini = build_sets(cfg.data_root, 'VOC12', train, cfg.mini_cam_mask_dir,
                             workspace / 'sets', probe_size=2, mini_size=2, minimum=1)
    assert not set(probe.read_text().splitlines()) & set(mini.read_text().splitlines())
    m = json.loads((workspace / 'sets/manifest.json').read_text())
    assert min(m['probe_class_counts']) >= 1
    mask = np.array([[0, 1, 255], [2, 3, 0]], dtype=np.uint8)
    assert set(np.unique(mask_transform(Image.fromarray(mask), 32))) <= {0, 1, 2, 3, 255}


def test_state_guard_exception_and_nested_amp():
    m = small_model().train(); m.head.eval()
    states = [module.training for module in m.modules()]
    torch_before = torch.get_rng_state().clone()
    py_before, np_before = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError, match='deliberate'):
        with torch.autocast('cpu', dtype=torch.bfloat16), isolated_eval(m, 'cpu'):
            assert not m.training and not torch.is_autocast_cpu_enabled()
            torch.rand(4); np.random.rand(2); random.random()
            raise RuntimeError('deliberate')
    assert [module.training for module in m.modules()] == states
    assert torch.equal(torch_before, torch.get_rng_state())
    assert py_before == random.getstate()
    np.testing.assert_array_equal(np_before[1], np.random.get_state()[1])


@pytest.mark.parametrize('pooling,reduction,affinity', [('gwrp', 'mean', False), ('c2p', 'product', True)])
def test_probe_end_to_end_fp32_and_checkpoint_immutable(workspace, pooling, reduction, affinity):
    cfg = data_config(workspace)
    m = small_model(patch_pooling=pooling, c2p_pooling_reduction=reduction, c2p_pooling_affinity=affinity).train()
    probe = TokenProbe(m, cfg, 'cpu')
    fingerprint = state_sha256(m)
    dtypes = []
    handle = m.head.register_forward_hook(lambda module, inputs, output: dtypes.append(output.dtype))
    state = torch.get_rng_state().clone()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = probe.run(0, 1)
    handle.remove()
    assert dtypes and set(dtypes) == {torch.float32}
    assert m.training and state_sha256(m) == fingerprint
    assert torch.equal(state, torch.get_rng_state())
    assert np.isfinite(result['mini_cam_miou_fixed'])
    assert (Path(cfg.out_dir) / 'epoch_000.COMPLETE').exists()
    rows = list(csv.DictReader((Path(cfg.out_dir) / 'metrics.csv').open()))
    assert {r['layer'] for r in rows if r['metric'] == 'kappa'} == {str(i) for i in range(1, 13)} | {'aggregate'}
    meta = json.loads((Path(cfg.out_dir) / 'epoch_000.json').read_text())
    assert meta['model_state_sha256'] == fingerprint and meta['checkpoint_sha256'] is None
    with pytest.raises(FileExistsError):
        probe.run(0, 1)
    assert m.training


def test_native_cam_exact_official_readout():
    from tools.evaluate_mini_cam import native_cam
    model = small_model().eval(); official = small_model(cam=True).eval()
    official.load_state_dict(model.state_dict())
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        _, patches, records, _ = model.forward_features(x)
        maps = model.head(patches.reshape(2, 2, 2, 24).permute(0, 3, 1, 2).contiguous())
        torch.testing.assert_close(native_cam(model, maps, records), official(x), atol=0, rtol=0)


def test_training_next_step_bitwise_unchanged(workspace, monkeypatch):
    cfg = data_config(workspace, mini=False)
    torch.manual_seed(28)
    baseline = small_model().train(); probed = copy.deepcopy(baseline)
    probe = TokenProbe(probed, cfg, 'cpu')
    image_open = Image.open
    def no_masks(path, *args, **kwargs):
        if str(path).endswith('.png'):
            raise AssertionError('Label-only probe must not open semantic masks')
        return image_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', no_masks)
    def step(m, opt):
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        loss = out[0].square().mean() + out[2].square().mean()
        loss.backward(); opt.step(); opt.zero_grad()
        return loss.detach()
    outcomes = []
    for model, use_probe in [(baseline, False), (probed, True)]:
        torch.manual_seed(41)
        optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9)
        first = step(model, optimizer)
        if use_probe:
            probe.run(0, 1)
        second = step(model, optimizer)
        outcomes.append((first, second, state_sha256(model)))
    assert outcomes[0][2] == outcomes[1][2]
    torch.testing.assert_close(outcomes[0][1], outcomes[1][1], atol=0, rtol=0)


def test_disabled_default_and_output_guard(workspace):
    from train_model_v2 import get_args_parser
    args = get_args_parser().parse_args([])
    assert not args.probe
    p = TokenProbe(None, ProbeConfig(), 'cpu')
    assert p.run(0, 0) == {}
    with pytest.raises(ValueError, match='results'):
        result_path('/tmp/diagnostics-not-allowed')
    with pytest.raises(ValueError):
        result_path(REPO / 'results')
    link = workspace / 'escape'; link.symlink_to('/tmp', target_is_directory=True)
    with pytest.raises(ValueError):
        result_path(link / 'out')


def test_nonroot_does_not_write_or_forward(monkeypatch):
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: 1)
    calls = []
    monkeypatch.setattr(torch.distributed, 'broadcast_object_list', lambda status, src: calls.append(src))
    p = TokenProbe(None, ProbeConfig(enabled=True), 'cpu')
    assert p.run(0, 0) == {} and calls == [0, 0]


def test_overlap_and_config_mismatch_rejected(workspace):
    cfg = data_config(workspace)
    cfg.mini_cam_set_path = cfg.probe_set_path
    with pytest.raises(ValueError, match='disjoint'):
        TokenProbe(small_model(), cfg, 'cpu')
    cfg.mini_cam_set_path = ''
    cfg.gwrp_decay = .5
    with pytest.raises(ValueError, match='actual model'):
        TokenProbe(small_model(), cfg, 'cpu')


def test_trajectory_constant_and_no_posthoc_direction():
    from analysis.diagnostics.plot_trajectory import association, registered_direction
    out = association([0, 1, 2], [3, 2, 1], [.1, .2, .3], 'min')
    assert out['spearman'] == -1 and out['peak_epoch_gap'] == 0
    assert np.isnan(association([0, 1, 2], [1, 1, 1], [.1, .2, .3], 'max')['spearman'])
    assert registered_direction('kappa__12') is None
    assert registered_direction('delta_js__aggregate') is None
    missing = association([0, 1, 2], [3, 2, np.nan], [.1, .2, .3], 'min')
    assert missing['cam_peak_epoch'] == 2 and missing['peak_epoch_gap'] == 1


def test_full_trajectory_render_and_no_single_seed_claim(workspace):
    from analysis.diagnostics.plot_trajectory import analyze
    cfg = data_config(workspace)
    probe = TokenProbe(small_model(), cfg, 'cpu')
    for epoch in range(3):
        probe.run(epoch, epoch + 1)
    output = workspace / 'plots'
    analyze([cfg.out_dir], output)
    assert (output / 'trajectories.pdf').stat().st_size > 0
    assert (output / 'layer_profile_gwrp_s0.pdf').exists()
    report = json.loads((output / 'analysis.json').read_text())
    assert all(not r['meets_descriptive_preregistered_threshold'] for r in report['criteria'])
    assert (output / 'correlations_per_seed.csv').exists()


def test_queue_matrix_and_matched_arguments():
    from experiments.run_diag_trajectory import matrix, training_command
    from experiments.baselines.run_default_voc_coco import dataset_spec
    assert matrix('VOC12') == [(0, 'gwrp'), (0, 'c2p'), (1, 'gwrp'), (1, 'c2p'), (2, 'gwrp'), (2, 'c2p')]
    assert matrix('COCO') == [(0, 'c2p')]
    args = training_command(dataset_spec('VOC12'), 'new_run', 0, 'gwrp', 'probe', 'mini')
    assert args[args.index('--epochs') + 1] == '45'
    assert args[args.index('--input-size') + 1] == '448'
    assert '--probe' in args and '--c2p-pooling-affinity' not in args


def test_original_engine_adamw_cct_trajectory_unchanged(workspace):
    from engine import train_one_epoch_mctplus
    cfg = data_config(workspace)
    torch.manual_seed(29)
    baseline = small_model().train(); probed = copy.deepcopy(baseline)
    probe = TokenProbe(probed, cfg, 'cpu')
    outputs = []
    for model, use_probe in [(baseline, False), (probed, True)]:
        torch.manual_seed(49)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.125e-5, weight_decay=.05)
        summaries = []
        for epoch in range(2):
            batches = [(torch.randn(2, 3, 32, 32), torch.tensor([[1., 1., 0.], [0., 1., 1.]]))]
            summaries.append(train_one_epoch_mctplus(model, batches, optimizer, torch.device('cpu'), epoch, loss_scaler=None))
            if use_probe:
                probe.run(epoch, epoch + 1)
        outputs.append((summaries, state_sha256(model), copy.deepcopy(optimizer.state_dict())))
    assert outputs[0][0] == outputs[1][0]
    assert outputs[0][1] == outputs[1][1]
    for index, state in outputs[0][2]['state'].items():
        for key, value in state.items():
            torch.testing.assert_close(value, outputs[1][2]['state'][index][key], atol=0, rtol=0)
