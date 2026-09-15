import numpy as np
import pytest
import torch
from PIL import Image

from models.mctformer_plus import build_mctformerplus, model_spec_from_instance, resolve_mctformerplus_checkpoint_variant
from tools.evaluate_raw_cam_streaming import OnlineCamEvaluator, evaluate


@pytest.mark.parametrize('n', [21, 81])
def test_online_equals_saved_cams(n, tmp_path):
    cams, masks = tmp_path / 'cams', tmp_path / 'masks'
    cams.mkdir(); masks.mkdir()
    ids = tmp_path / 'train_ids.txt'
    ids.write_text('a\nb\nc\n')
    rng = np.random.default_rng(8)
    evaluator = OnlineCamEvaluator(masks, ids, n, tmp_path / 'online')
    for name in ['a', 'b', 'c']:
        target = rng.integers(0, n, (17, 11), dtype=np.uint8)
        target[0, 0] = 255
        payload = {} if name == 'b' else {
            torch.tensor(0): rng.choice(np.arange(60, dtype=np.float32) / 100, (17, 11)),
            torch.tensor(n - 2): np.full((17, 11), .45, np.float32)}
        Image.fromarray(target).save(masks / f'{name}.png')
        np.save(cams / f'{name}.npy', payload)
        evaluator.update(name, payload)
    online = evaluator.finish()
    offline = evaluate(cams, masks, ids, n, tmp_path / 'offline')
    assert online['fixed'] == offline['fixed'] and online['best'] == offline['best']
    a = np.load(tmp_path / 'online/aggregate_confusions.npz')['confusion']
    b = np.load(tmp_path / 'offline/aggregate_confusions.npz')['confusion']
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('amp', [False, True])
def test_coco_all_product_forward_gradient_cam(amp):
    if amp and not torch.cuda.is_available():
        pytest.skip('CUDA required')
    device = 'cuda' if amp else 'cpu'
    torch.manual_seed(6)
    model = build_mctformerplus('small', input_size=32, num_classes=80,
                                patch_pooling='c2p', c2p_pooling_layers='all',
                                c2p_pooling_reduction='product').to(device).eval()
    x = torch.randn(2, 3, 32, 32, device=device)
    with torch.autocast(device_type=device, enabled=amp):
        cls, raw, patch = model(x)
        assert cls.shape == patch.shape == (2, 80)
        assert raw.shape[:3] == (12, 2, 80)
        loss = patch.square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.head.weight.grad.abs().sum() > 0
    checkpoint = {'model': model.state_dict(), 'model_spec': model_spec_from_instance(model)}
    resolve_mctformerplus_checkpoint_variant(checkpoint, 'mctformerplus', num_classes=80)
    cam = build_mctformerplus('small', input_size=32, num_classes=80, cam=True,
                              patch_pooling='c2p', c2p_pooling_layers='all',
                              c2p_pooling_reduction='product').to(device).eval()
    cam.load_state_dict(checkpoint['model'], strict=True)
    assert torch.isfinite(cam(x)).all()
