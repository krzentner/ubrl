import torch
from ubrl.torch_utils import clamp_identity_grad, soft_clamp


def test_clamp_identity_grad():
    a = torch.tensor([-2, -1, 0, 1, 2])
    a_clamp = clamp_identity_grad(a, min=1.0)
    assert torch.allclose(a_clamp, torch.tensor([1, 1, 1, 1, 2]).float())

    a_clamp = clamp_identity_grad(a, max=1.0)
    assert torch.allclose(a_clamp, torch.tensor([-2, -1, 0, 1, 1]).float())

    a_clamp = clamp_identity_grad(a, min=0.0, max=1.0)
    assert torch.allclose(a_clamp, torch.tensor([0, 0, 0, 1, 1]).float())


def test_soft_clamp():
    a = torch.tensor([-2, -1, 0, 1, 2])
    a_clamp = soft_clamp(a, min=1.0, scale=1.0)
    assert (a_clamp > 1.0).all()
    assert a_clamp[-1] == 2.0
    assert a_clamp[-2] >= 1.0

    a_clamp = soft_clamp(a, max=1.0, scale=0.5)
    assert (a_clamp < 1.0).all()
    assert a_clamp[0] == -2.0
    assert a_clamp[1] == -1.0
    assert a_clamp[2] == 0.0
    assert a_clamp[3] < a_clamp[4]

    a_clamp = soft_clamp(a, min=0.0, max=1.0)
    assert (a_clamp > 0.0).all()
    assert (a_clamp < 1.0).all()
    assert a_clamp[0] < a_clamp[1]
    assert a_clamp[1] < a_clamp[2]
    assert a_clamp[2] < a_clamp[3]
    assert a_clamp[3] < a_clamp[4]
