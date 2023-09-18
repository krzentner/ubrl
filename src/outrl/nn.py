import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union, Callable, Tuple
import math

import torch
import torch.nn.functional as F
import outrl

SupportsNonlinearity = Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module]
Initializer = Callable[[torch.Tensor], None]
Shape = Union[int, Tuple[int, ...]]


def as_2d(x):
    x = x.squeeze()
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x


@torch.jit.script
def compute_advantages(
    *,
    discount: float,
    gae_lambda: float,
    vf_returns: torch.Tensor,
    rewards: torch.Tensor,
):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    Calculate advantages using a baseline according to Generalized Advantage
    Estimation (GAE)

    The discounted cumulative sum can be computed using conv2d with filter.
    filter:
        [1, (discount * gae_lambda), (discount * gae_lambda) ^ 2, ...]
        where the length is same with max_episode_length.

    expected_returns and rewards should have the same shape.
        expected_returns:
        [ [b_11, b_12, b_13, ... b_1n],
          [b_21, b_22, b_23, ... b_2n],
          ...
          [b_m1, b_m2, b_m3, ... b_mn] ]
        rewards:
        [ [r_11, r_12, r_13, ... r_1n],
          [r_21, r_22, r_23, ... r_2n],
          ...
          [r_m1, r_m2, r_m3, ... r_mn] ]

    Args:
        discount (float): RL discount factor (i.e. gamma).
        gae_lambda (float): Lambda, as used for Generalized Advantage
            Estimation (GAE).
        vf_returns (torch.Tensor): A 2D tensor of value function
            estimates with shape (N, T + 1), where N is the batch dimension
            (number of episodes) and T is the maximum episode length
            experienced by the agent. If an episode terminates in fewer than T
            time steps, the remaining elements in that episode should be set to
            0.
        rewards (torch.Tensor): A 2D tensor of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        episode_lengths (torch.Tensor): A 1D vector of episode lengths.
    Returns:
        torch.Tensor: A 2D vector of calculated advantage values with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining values
            in that episode should be set to 0.

    """
    rewards = as_2d(rewards)
    vf_returns = as_2d(vf_returns)
    n_episodes = rewards.shape[0]
    max_episode_length = rewards.shape[1]
    assert vf_returns.shape == (n_episodes, max_episode_length + 1)

    delta = -vf_returns[:, :-1] + rewards + discount * vf_returns[:, 1:]
    adv_gae = torch.zeros((n_episodes, max_episode_length))
    adv_gae[max_episode_length - 1] = delta[max_episode_length - 1]
    for t in range(max_episode_length - 2, 0, -1):
        adv_gae[:, t] = delta[:, t] + discount * gae_lambda * adv_gae[:, t + 1]
    return adv_gae.squeeze()


def discount_cumsum(x: torch.Tensor, discount: float):
    x = as_2d(x)
    B, L = x.shape
    discount_x = discount * torch.ones_like(x[0])
    discount_x[0] = 1.0
    # Compute discount weights.
    weights = torch.cumprod(discount_x, dim=0)
    # Add channel in dimensions and channel out dimensions
    weights = weights.reshape(1, 1, L)
    x = x.reshape(B, 1, L)
    # Add pad end of episodes to zero
    # Only need 2l - 1 timesteps to make index L valid
    x_pad = torch.cat([x, torch.zeros_like(x[:, :, :-1])], axis=-1)
    returns = F.conv1d(x_pad, weights, stride=1)
    assert returns.shape == (B, 1, L)
    return returns.squeeze()


def soft_update_model(
    target_model: torch.nn.Module, source_model: torch.nn.Module, tau: float
):
    """Update model parameter of target and source model.

    Args:
        target_model (torch.nn.Module):
            Target model to update.
        source_model (torch.nn.Module):
            Source network to update.
        tau (float): Interpolation parameter for doing the
            soft target update.

    """
    for target_param, param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class NonLinearity(torch.nn.Module):
    """Wrapper class for non linear function or module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linear: SupportsNonlinearity):
        super().__init__()

        if isinstance(non_linear, type):
            self.module = non_linear()
        elif callable(non_linear):
            self.module = copy.deepcopy(non_linear)
        else:
            raise ValueError(
                "Non linear function {} is not supported".format(non_linear)
            )

    def forward(self, x: torch.Tensor):
        return self.module(x)

    def __repr__(self):
        return repr(self.module)


def flatten_shape(shape: Shape) -> int:
    if isinstance(shape, int):
        return shape
    else:
        return math.prod(shape)


def make_mlp(
    *,
    input_size: int,
    hidden_sizes: List[int],
    output_size: Optional[int] = None,
    hidden_nonlinearity: SupportsNonlinearity = torch.relu,
    hidden_w_init: Initializer = torch.nn.init.xavier_normal_,
    hidden_b_init: Initializer = torch.nn.init.zeros_,
    output_w_init: Initializer = torch.nn.init.xavier_normal_,
    output_b_init: Initializer = torch.nn.init.zeros_,
    layer_normalization: bool = False,
) -> torch.nn.Sequential:
    layers = torch.nn.Sequential()
    prev_size = input_size
    for size in hidden_sizes:
        step_layers = torch.nn.Sequential()
        if layer_normalization:
            step_layers.add_module("layer_normalization", torch.nn.LayerNorm(prev_size))
        linear_layer = torch.nn.Linear(prev_size, size)
        hidden_w_init(linear_layer.weight)
        hidden_b_init(linear_layer.bias)
        step_layers.add_module("linear", linear_layer)

        if hidden_nonlinearity:
            step_layers.add_module("non_linearity", NonLinearity(hidden_nonlinearity))
        layers.append(step_layers)

        prev_size = size

    if output_size is not None:
        linear_layer = torch.nn.Linear(prev_size, output_size)
        output_w_init(linear_layer.weight)
        output_b_init(linear_layer.bias)
        layers.add_module("output_linear", linear_layer)

    return layers


class DistConstructor:
    def get_encoded_size(self, action_shape: Shape) -> int:
        return flatten_shape(action_shape)

    def get_input_size(self, action_shape: Shape) -> int:
        del action_shape
        raise NotImplementedError()

    def __call__(self, x: torch.Tensor):
        del x
        raise NotImplementedError()

    def encode_actions(
        self, actions: torch.Tensor, dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        del actions, dist
        raise NotImplementedError()


@dataclass
class NormalConstructor(DistConstructor):
    min_std: float = 1e-6
    max_std: float = 2.0
    std_parameterization: str = "exp"

    def get_input_size(self, action_shape: Shape) -> int:
        return 2 * flatten_shape(action_shape)

    def __call__(self, mean_std: torch.Tensor):
        assert len(mean_std.shape) == 2
        input_size = mean_std.shape[1]
        assert input_size % 2 == 0
        I = int(input_size // 2)
        mean = mean_std[:, :I]
        std = mean_std[:, I:]

        if self.std_parameterization == "exp":
            std = std.exp()
        elif self.std_parameterization == "softplus":
            std = std.exp().exp().add(1.0).log()

        std = torch.clamp(std, self.min_std, self.max_std)

        dist = torch.distributions.Normal(mean, std)
        dist = torch.distributions.Independent(dist, 1)
        return dist

    def encode_actions(
        self, actions: torch.Tensor, dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        return actions


@dataclass
class CategoricalConstructor(DistConstructor):
    def get_input_size(self, action_shape: Shape) -> int:
        return flatten_shape(action_shape)

    def __call__(self, logits: torch.Tensor):
        # dist = torch.distributions.Independent(torch.distributions.Categorical(logits=logits), 1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist

    def encode_actions(
        self, actions: torch.Tensor, dist: torch.distributions.Distribution
    ) -> torch.Tensor:
        # return F.one_hot(actions, num_classes=dist.base_dist.probs.shape[1])
        return F.one_hot(actions, num_classes=dist.probs.shape[1])


DEFAULT_DIST_TYPES = {
    "continuous": NormalConstructor,
    "discrete": CategoricalConstructor,
}


def explained_variance(ypred: torch.Tensor, y: torch.Tensor):
    """Explained variation for 1D inputs.

    It is the proportion of the variance in one variable that is explained or
    predicted from another variable.

    interpretation:
        1 => all variance explained
        0 => not predicting anything
        <0 => overfit and predicting badly

    Args:
        ypred (np.ndarray): Sample data from the first variable.
            Shape: :math:`(N, max_episode_length)`.
        y (np.ndarray): Sample data from the second variable.
            Shape: :math:`(N, max_episode_length)`.

    Returns:
        float: The explained variance.

    """
    vary = torch.var(y)

    # Handle corner cases
    if vary == 0:
        if torch.var(ypred) > 0:
            return torch.tensor(0)

        return torch.tensor(1)

    epsilon = 1e-8

    res = 1 - (torch.var(y - ypred) + epsilon) / (vary + epsilon)
    return res


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
