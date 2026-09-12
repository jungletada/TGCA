import numpy as np
import pytest
from PIL import Image
from tools.evaluate_cam_threshold_grid import image_threshold_confusions, load_cam_winner
from tools.evaluate_raw_cam_streaming import evaluate


@pytest.mark.parametrize('n', [21, 81])
def test_confusions_match_direct_with_ties_and_void(n):
    rng = np.random.default_rng(5)
    thresholds = np.arange(60) / 100
    scores = rng.choice(thresholds, size=(17, 13))
    classes = rng.integers(1, n, size=scores.shape)
    target = rng.integers(0, n, size=scores.shape)
    target[0, 0] = 255
    observed = image_threshold_confusions(scores, classes, target, thresholds, n)
    valid = target != 255
    for i, t in enumerate(thresholds):
        pred = np.where(scores > t, classes, 0)
        direct = np.bincount(n * target[valid] + pred[valid], minlength=n*n).reshape(n, n)
        np.testing.assert_array_equal(observed[i], direct)


def test_streaming_includes_empty_cam_and_highest_coco_class(tmp_path):
    cams, masks = tmp_path / 'cams', tmp_path / 'masks'
    cams.mkdir()
    masks.mkdir()
    np.save(cams / 'a.npy', {79: np.ones((2, 2), np.float32)})
    np.save(cams / 'b.npy', {})
    Image.fromarray(np.full((2, 2), 80, np.uint8)).save(masks / 'a.png')
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(masks / 'b.png')
    ids = tmp_path / 'ids.txt'
    ids.write_text('a\nb\n')
    out = tmp_path / 'out'
    result = evaluate(cams, masks, ids, 81, out)
    assert result['fixed']['mean_iou'] == 1
    total = np.load(out / 'aggregate_confusions.npz')['confusion']
    assert np.all(total.sum((1, 2)) == 8)
    with pytest.raises(FileExistsError):
        evaluate(cams, masks, ids, 81, out)
    with pytest.raises(ValueError):
        load_cam_winner(cams / 'a.npy')
