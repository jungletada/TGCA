import torch
from analysis.head_alpha_preflight import product_weights, TV_GATE


def test_half_boundary_and_fp32_product_formula():
    a = torch.tensor([[[[[1e-10, 2e-10]]]]]).expand(12, 1, 6, 1, 2)
    p = product_weights(a)
    ref = a.double().mean(2).prod(0)
    ref /= ref.sum(-1, keepdim=True)
    torch.testing.assert_close(p.double(), ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(product_weights(a.half().float()), torch.full((1,1,2), .5))
    assert TV_GATE == .001
