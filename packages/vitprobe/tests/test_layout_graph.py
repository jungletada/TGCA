import pytest
import torch

from vitprobe import TokenLayout
from vitprobe.graph import aggregate_p2p, propagate_weights


@pytest.mark.parametrize('class_first', [True, False])
def test_layout_all_groups_and_rectangular_grid(class_first):
    layout = TokenLayout(n_cls=1, n_class=2, n_register=4, grid_hw=(2, 3),
                         class_first=class_first)
    groups = ([1] + [2]*2 + [3]*4 + [4]*6 if class_first else
              [4]*6 + [1] + [2]*2 + [3]*4)
    tokens = torch.tensor(groups)[None, :, None]
    assert (tokens[:, layout.cls_slice] == 1).all()
    assert (tokens[:, layout.class_slice] == 2).all()
    assert (tokens[:, layout.register_slice] == 3).all()
    assert (layout.patch_tokens(tokens) == 4).all()
    assert layout.patch_grid(tokens).shape == (1, 2, 3, 1)
    a = torch.arange(layout.n_tokens**2).reshape(layout.n_tokens, layout.n_tokens)
    for name, index in [('class', layout.class_slice), ('cls', layout.cls_slice),
                        ('register', layout.register_slice)]:
        assert torch.equal(layout.query_patch_attention(a, name), a[index, layout.patch_slice])


def test_factory_slices_and_empty_groups():
    layout = TokenLayout.mctformer_plus()
    assert layout.n_tokens == 804 and layout.n_patch == 784
    assert layout.class_slice == slice(0, 20)
    assert layout.patch_slice == slice(20, 804)
    layout = TokenLayout.dinov3(grid=(3, 4))
    assert layout.n_class == 0 and layout.n_tokens == 17
    assert layout.cls_slice == slice(0, 1)
    assert layout.class_slice == slice(1, 1)
    assert layout.register_slice == slice(1, 5)
    plain = TokenLayout(n_cls=1, grid_hw=(2, 2))
    assert plain.n_tokens == 5


@pytest.mark.parametrize('kwargs', [dict(n_class=-1), dict(n_cls=.5),
                                   dict(grid_hw=(0, 2)), dict(class_first=1)])
def test_invalid_layout(kwargs):
    with pytest.raises(ValueError):
        TokenLayout(**kwargs)


def test_layout_mismatch():
    layout = TokenLayout(n_class=2, grid_hw=(2, 2))
    with pytest.raises(ValueError):
        layout.patch_tokens(torch.zeros(1, 7, 4))
    with pytest.raises(ValueError):
        layout.patch_attention(torch.zeros(1, 5, 5))


def test_graph_direction_and_fallback():
    w = torch.tensor([[[1., 0., 0.]]])
    # Key0 is read exclusively by query1; w @ P.T must therefore point to1.
    p = torch.tensor([[[0., 1., 0.], [1., 0., 0.], [0., 0., 1.]]])
    assert torch.equal(propagate_weights(w, p), torch.tensor([[[0., 1., 0.]]]))
    result, fallback = propagate_weights(w, torch.ones_like(p), floor='min',
                                        return_fallback=True)
    assert fallback.all() and torch.equal(result, w)


def test_graph_low_precision_gradient_and_autocast():
    layout = TokenLayout(n_cls=1, n_register=1, grid_hw=(2, 2))
    a = torch.rand(2, 2, 6, 6, requires_grad=True)
    w = torch.rand(2, 3, 4).softmax(-1).detach().requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        p = aggregate_p2p([a], layout)
        out = propagate_weights(w, p)
    assert p.dtype == out.dtype == torch.float32
    out.square().sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad.abs().sum() > 0
    assert torch.isfinite(w.grad).all() and w.grad.abs().sum() > 0
