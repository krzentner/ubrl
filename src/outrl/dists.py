from dataclasses import dataclass

import torch
import torch.nn.functional as F

from outrl.nn import flatten_shape, Shape


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
