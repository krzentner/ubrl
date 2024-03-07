import torch
from outrl.rl import discount_cumsum


def test_discount_cumsum():
    B = 7
    L = 9
    discount = 0.9
    rewards = torch.randn(B, L)
    expected_result = torch.zeros_like(rewards)
    expected_result[:, -1] = rewards[:, -1]
    for i in range(L - 2, -1, -1):
        expected_result[:, i] = rewards[:, i] + discount * expected_result[:, i + 1]
    actual_result = discount_cumsum(rewards, discount)
    assert torch.allclose(actual_result, expected_result)
