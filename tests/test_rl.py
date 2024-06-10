import torch
from ubrl import _discount_cumsum, _v_trace_estimation


def test_discount_cumsum():
    B = 7
    L = 9
    discount = 0.9
    rewards = torch.randn(B, L)
    expected_result = torch.zeros_like(rewards)
    expected_result[:, -1] = rewards[:, -1]
    for i in range(L - 2, -1, -1):
        expected_result[:, i] = rewards[:, i] + discount * expected_result[:, i + 1]
    actual_result = _discount_cumsum(rewards, discount)
    assert torch.allclose(actual_result, expected_result, rtol=1e-4)


def test_v_trace_estimation():
    B = 3
    T = 5
    ones = torch.ones(B, T, dtype=torch.float32)
    zeros = torch.ones(B, T, dtype=torch.float32)
    terminated = torch.ones(B, dtype=torch.bool)

    advantages, vf_targets = _v_trace_estimation(
        lmbda=1.0,
        rho_max=1.0,
        c_max=1.0,
        gammas=zeros,
        vf_x=torch.zeros(B, T + 1, dtype=torch.float32),
        rewards=ones,
        action_lls=ones,
        original_action_lls=ones,
        terminated=terminated,
        episode_lengths=T * torch.ones(B).long(),
    )

    expected_advantages = torch.Tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )

    expected_vf_targets = torch.Tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(vf_targets, expected_vf_targets)

    advantages, vf_targets = _v_trace_estimation(
        lmbda=1.0,
        rho_max=1.0,
        c_max=1.0,
        gammas=zeros,
        vf_x=torch.ones(B, T + 1, dtype=torch.float32),
        rewards=ones,
        action_lls=ones,
        original_action_lls=ones,
        terminated=terminated,
        episode_lengths=T * torch.ones(B).long(),
    )

    expected_advantages = torch.Tensor(
        [
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0, 0.0],
            [4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )

    # TODO: The last column here should probably be zero
    expected_vf_targets = torch.Tensor(
        [
            [5.0, 4.0, 3.0, 2.0, 1.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 1.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 1.0],
        ]
    )

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(vf_targets, expected_vf_targets)

    advantages, vf_targets = _v_trace_estimation(
        lmbda=0.5,
        rho_max=1.0,
        c_max=1.0,
        gammas=zeros,
        vf_x=3 * torch.ones(B, T + 1, dtype=torch.float32),
        rewards=ones,
        action_lls=ones,
        original_action_lls=ones,
        terminated=terminated,
        episode_lengths=T * torch.ones(B).long(),
    )

    expected_advantages = torch.Tensor(
        [
            [2.5, 2.0, 1.0, -1.0, -2.0],
            [2.5, 2.0, 1.0, -1.0, -2.0],
            [2.5, 2.0, 1.0, -1.0, -2.0],
        ]
    )

    # TODO: The last column here should probably be zero
    expected_vf_targets = torch.Tensor(
        [
            [4.75, 4.5, 4.0, 3.0, 1.0, 3.0],
            [4.75, 4.5, 4.0, 3.0, 1.0, 3.0],
            [4.75, 4.5, 4.0, 3.0, 1.0, 3.0],
        ]
    )

    assert torch.allclose(advantages, expected_advantages)
    assert torch.allclose(vf_targets, expected_vf_targets)
