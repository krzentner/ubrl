from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RunningMeanVar:
    """Calculates running mean and variance of sequence of tensors.

    Inspired by the Tianshou class of the same name.
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    mean: torch.Tensor = torch.tensor(0.0)
    """Running mean of Tensors passed to `update`.
    Can have an arbitrary shape."""

    var: torch.Tensor = torch.tensor(1.0)
    """Running variance of Tensors passed to `update`.
    Not corrected using Bessel's Correction.
    Can have an arbitrary shape."""

    count: int = 0
    """Total number of samples seen by `update`."""

    clip_max: Optional[float] = 10.0
    """Maximal value allowed by output of normalize_batch.
    Can have an arbitrary shape.
    Also results in clipping negative values to be above -clip_max."""

    epsilon: float = 1e-5
    """Added to variance when normalizing to avoid division by zero."""

    def __post_init__(self):
        if not isinstance(self.mean, torch.Tensor):
            self.mean = torch.tensor(self.mean)
        if not isinstance(self.var, torch.Tensor):
            self.var = torch.tensor(self.var)

    def normalize_batch(self, x: torch.Tensor, correction=1) -> torch.Tensor:
        y = (x - self.mean) / (
            self.corrected_var(correction) + self.epsilon
        )
        if self.clip_max:
            y = torch.clamp(y, min=-self.clip_max, max=self.clip_max)
        return y

    def update(self, x: torch.Tensor):
        if len(x.shape) < len(self.mean.shape):
            x = x.unsqueeze(0)
        x_mean = x.mean(dim=0)
        x_var = x.var(dim=0, correction=0)
        x_size = len(x)

        delta = x_mean - self.mean
        m2_a = self.var * self.count
        m2_b = x_var * x_size
        new_total = self.count + x_size
        m2 = m2_a + m2_b + delta**2 * self.count * x_size / new_total

        new_mean = self.mean + delta * x_size / new_total
        new_var = m2 / new_total

        # pprint(locals())

        self.var = new_var
        self.mean = new_mean
        self.count = new_total

    def corrected_var(self, correction=1):
        d = self.count - correction
        if d <= 0:
            d = 1
        return self.var * self.count / d


def test_running_mean_var():
    rmv1 = RunningMeanVar()
    rmv2 = RunningMeanVar()
    a = torch.arange(100, dtype=torch.float)
    rmv1.update(a)
    rmv2.update(a[:50])
    rmv2.update(a[50:])
    assert torch.allclose(rmv1.var, a.var(correction=0))
    assert torch.allclose(rmv1.mean, a.mean())
    assert torch.allclose(rmv2.var, a.var(correction=0))
    assert torch.allclose(rmv1.mean, rmv2.mean)
    assert torch.allclose(rmv1.var, rmv2.var)
    assert rmv1.count == rmv2.count


def test_running_mean_var2():
    rmv1 = RunningMeanVar()
    a = torch.arange(10, dtype=torch.float)
    rmv1.update(a)
    rmv1.update(a)
    assert torch.allclose(rmv1.mean, torch.concatenate([a, a]).mean())
    assert torch.allclose(rmv1.var, torch.concatenate([a, a]).var(correction=0))
    assert torch.allclose(rmv1.corrected_var(), torch.concatenate([a, a]).var())


def test_running_mean_var_shape():
    shape = (5, 3)
    rmv1 = RunningMeanVar(mean=torch.zeros(shape), var=torch.ones(shape))
    a = (
        torch.arange(20, dtype=torch.float32)
        .unsqueeze(-1)
        .unsqueeze(-1)
        .repeat((1,) + shape)
    )
    rmv1.update(a[:10])
    rmv1.update(a[10:])
    assert torch.allclose(rmv1.var, a.var(dim=0, correction=0))
