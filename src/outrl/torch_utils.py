import copy
from dataclasses import dataclass
from textwrap import dedent
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    Union,
    Callable,
    Tuple,
    Optional,
)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SupportsNonlinearity = Union[Callable[[torch.Tensor], torch.Tensor], nn.Module]
Initializer = Callable[[torch.Tensor], None]
Shape = Union[int, Tuple[int, ...]]
Sizes = Union[int, Tuple[int, ...], List[int]]


def as_2d(x):
    x = x.squeeze()
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    return x


def soft_update_model(target_model: nn.Module, source_model: nn.Module, tau: float):
    """Update model parameter of target and source model.

    Args:
        target_model (nn.Module):
            Target model to update.
        source_model (nn.Module):
            Source network to update.
        tau (float): Interpolation parameter for doing the
            soft target update.

    """
    for target_param, param in zip(
        target_model.parameters(), source_model.parameters()
    ):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class NonLinearity(nn.Module):
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


class SqueezeModule(nn.Module):
    """Module that squeezes output.

    Args:
        dim (int?): Index to squeeze.

    """

    def __init__(self, dim: Optional[int] = None):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.squeeze(self.dim)

    def __repr__(self):
        return f"SqueezeModule(dim={self.dim})"


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
    hidden_nonlinearity: SupportsNonlinearity = nn.GELU,
    hidden_w_init: Initializer = nn.init.xavier_normal_,
    hidden_b_init: Initializer = nn.init.zeros_,
    output_w_init: Initializer = nn.init.xavier_normal_,
    output_b_init: Initializer = nn.init.zeros_,
    add_dropout: bool = True,
    layer_normalization: bool = False,
    squeeze_out: bool = False,
) -> nn.Sequential:
    layers = nn.Sequential()
    prev_size = input_size
    for size in hidden_sizes:
        step_layers = nn.Sequential()
        if add_dropout:
            step_layers.add_module("dropout", nn.Dropout())
        if layer_normalization:
            step_layers.add_module("layer_normalization", nn.LayerNorm(prev_size))
        linear_layer = nn.Linear(prev_size, size)
        hidden_w_init(linear_layer.weight)
        hidden_b_init(linear_layer.bias)
        step_layers.add_module("linear", linear_layer)

        if hidden_nonlinearity:
            step_layers.add_module("non_linearity", NonLinearity(hidden_nonlinearity))
        layers.append(step_layers)

        prev_size = size

    original_output_size = output_size
    if output_size is not None:
        if output_size == 0:
            output_size = 1
        linear_layer = nn.Linear(prev_size, output_size)
        output_w_init(linear_layer.weight)
        output_b_init(linear_layer.bias)
        layers.add_module("output_linear", linear_layer)
        if original_output_size == 0:
            layers.add_module("squeeze", SqueezeModule(-1))
        elif squeeze_out:
            layers.add_module("squeeze", SqueezeModule())

    return layers


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


def as_torch_dict(info: Dict[str, Any]):
    info_torch = {}
    for k, v in info.items():
        try:
            info_torch[k] = torch.tensor(v)
        except TypeError:
            warnings.warn(f"Could not convert info {k!r} ({v!r}) to torch.Tensor")
            pass
    return info_torch


def stack_dicts(infos: List[Dict[str, torch.Tensor]]):
    common_keys = set(infos[0].keys())
    all_keys = set(infos[0].keys())
    for info in infos:
        new_keys = set(info.keys())
        common_keys = common_keys.intersection(new_keys)
        all_keys = all_keys.union(new_keys)
    for key in all_keys - common_keys:
        warnings.warn(f"Discarding info not present in every step: {key}")
    return {k: torch.stack([info[k] for info in infos]) for k in common_keys}


class IterCallback(torch.utils.data.IterableDataset):
    """Converts a function that returns an iterator into an iterator."""

    def __init__(self, callback):
        self.callback = callback

    def __iter__(self):
        return self.callback()


def maybe_sample(dist: torch.distributions.Distribution, get_most_likely: bool = False):
    if get_most_likely:
        base_dist = getattr(dist, "base_dist", None)
        mean = getattr(dist, "mean", None)
        logits = getattr(dist, "logits", None)
        if base_dist is not None:
            return maybe_sample(base_dist, get_most_likely)
        elif mean is not None and not torch.isnan(mean).any():
            return mean
        elif logits is not None:
            return torch.argmax(logits, dim=1)
        else:
            raise NotImplementedError(f"Could not get most likely element of {dist}")
    else:
        return dist.sample()


class RunningMeanVar(nn.Module):
    """Calculates running mean and variance of sequence of tensors.

    When used as a module, records inputs such that outputs will become unit mean, variance.

    Algorithms from here:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(
        self,
        init_mean: torch.Tensor = torch.tensor(0.0),
        init_var: torch.Tensor = torch.tensor(1.0),
        count: int = 0,
        clip_max: Optional[float] = 10.0,
        min_var: float = 1e-6,
        trainable: bool = False,
    ):
        super().__init__()
        # Handle user passing in floats
        if not isinstance(init_mean, torch.Tensor):
            init_mean = torch.tensor(init_mean)
        if not isinstance(init_var, torch.Tensor):
            init_var = torch.tensor(init_var)

        if trainable:
            self.mean = nn.Parameter(init_mean)
            """Running mean of Tensors passed to `update`.
            Can have an arbitrary shape."""
        else:
            self.register_buffer("mean", init_mean)

        if trainable:
            self.var = nn.Parameter(init_var)
            """Running variance of Tensors passed to `update`.
            Not corrected using Bessel's Correction.
            Can have an arbitrary shape."""
        else:
            self.register_buffer("var", init_var)

        self.count = count
        """Total number of samples seen by `update`."""

        self.clip_max: Optional[float] = clip_max
        """Maximal value allowed by output of normalize_batch.
        Can have an arbitrary shape.
        Also results in clipping negative values to be above -clip_max."""

        self.min_var: float = min_var
        """Minimal permitted variance to avoid division by zero."""

    def normalize_batch(self, x: torch.Tensor, correction=1) -> torch.Tensor:
        y = (x - self.mean) / (torch.sqrt(self.corrected_var(correction)))
        if self.clip_max:
            y = torch.clamp(y, min=-self.clip_max, max=self.clip_max)
        return y

    def denormalize_batch(self, y: torch.Tensor, correction=1) -> torch.Tensor:
        return (y * torch.sqrt(self.corrected_var(correction))) + self.mean

    def forward(self, x: torch.Tensor):
        return self.update(x)

    def update(self, x: torch.Tensor):
        if len(x.shape) < len(self.mean.shape):
            x = x.unsqueeze(0)
        # Flatten all batch, time dimensions
        while len(x.shape) > len(self.mean.shape) + 1:
            x = x.flatten(end_dim=1)
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

        # from pprint import pprint; pprint(locals())

        assert self.var.shape == new_var.shape
        assert self.mean.shape == new_mean.shape
        self.var = new_var
        self.mean = new_mean
        self.count = new_total

        return self.normalize_batch(x)

    def corrected_var(self, correction=1):
        d = self.count - correction
        if d <= 0:
            d = 1
        cvar = self.var * self.count / d
        return torch.clamp(cvar, self.min_var)

    def stick_preprocess(self, prefix: str, dst: dict):
        dst[f"{prefix}.var"] = self.var.mean().item()
        dst[f"{prefix}.mean"] = self.mean.mean().item()
        dst[f"{prefix}.count"] = self.count


def pack_tensors(tensor_list: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    return (torch.cat(tensor_list)), [len(t) for t in tensor_list]


def pack_tensors_check(
    tensor_list: list[torch.Tensor], expected_lengths: list[int]
) -> torch.Tensor:
    assert expected_lengths == [len(t) for t in tensor_list]
    return torch.cat(tensor_list)


def unpack_tensors(tensor: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    start_i = 0
    out = []
    for length in lengths:
        out.append(tensor[start_i : start_i + length])
        start_i += length
    return out


def pad_tensors(
    tensor_list: list[torch.Tensor],
    expected_lengths: Optional[list[int]] = None,
    target_len: Optional[int] = None,
) -> torch.Tensor:
    lengths = []
    max_len = 0
    longest_tensor = None
    for t in tensor_list:
        if len(t) > max_len:
            max_len = len(t)
            longest_tensor = t
        lengths.append(len(t))
    assert longest_tensor is not None
    if expected_lengths is not None:
        assert lengths == expected_lengths
    if target_len is not None:
        if target_len >= max_len:
            max_len = target_len
        else:
            raise ValueError(
                f"Could not pad to length {target_len} "
                f"because input tensor has length {max_len}"
            )
    zero_pad = torch.zeros(
        max_len,
        *tensor_list[0].shape[1:],
        dtype=tensor_list[0].dtype,
        device=tensor_list[0].device,
    )
    return torch.stack([torch.cat([t, zero_pad[len(t) :]]) for t in tensor_list])


def unpad_tensors(padded: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    assert all(padded.shape[1] >= length for length in lengths)
    return [padded[i, :length] for i, length in enumerate(lengths)]


@dataclass
class DictDataset:
    def __init__(self, elements: Optional[dict[str, torch.Tensor]] = None, **kwargs):
        if elements is None:
            self.elements = kwargs
        else:
            assert len(kwargs) == 0
            self.elements = elements

    def __len__(self):
        length = len(next(iter(self.elements.values())))
        for val in self.elements.values():
            assert len(val) == length
        return length

    def __getitem__(self, index):
        # Single index fast path
        if isinstance(index, int):
            return {k: v[index] for (k, v) in self.elements.items()}
        else:
            result = {}
            for k, v in self.elements.items():
                if isinstance(v, list):
                    result[k] = [v[i] for i in index]
                else:
                    # Hopefully it's some kind of tensor
                    result[k] = v[index]
            return result

    def split(self, p_right=0.5):
        ind_left, ind_right = split_shuffled_indices(len(self), p_right)
        left = {k: v[ind_left] for (k, v) in self.elements.items()}
        right = {k: v[ind_right] for (k, v) in self.elements.items()}
        return left, right

    def minibatches(self, minibatch_size: int, drop_last: bool = False):
        indices = torch.randperm(len(self))
        start_i = 0
        while start_i + minibatch_size < len(indices):
            yield self[indices[start_i : start_i + minibatch_size]]
            start_i += minibatch_size
        if start_i != len(indices) and not drop_last:
            yield self[indices[start_i:]]

    def episode_minibatches(
        self, min_timesteps_per_minibatch: int, drop_last: bool = False
    ):
        assert "length" in self.elements
        indices = torch.randperm(len(self))
        start_i = 0
        while start_i < len(indices):
            n_episodes = 1
            while (
                start_i + n_episodes < len(indices)
                and self.elements["length"][
                    indices[start_i : start_i + n_episodes]
                ].sum()
                < min_timesteps_per_minibatch
            ):
                n_episodes += 1
            if start_i + n_episodes < len(indices):
                yield self[indices[start_i : start_i + n_episodes]]
            elif not drop_last:
                yield self[indices[start_i:]]
            start_i += n_episodes


def split_shuffled_indices(total: int, p_right: float = 0.5):
    indices = torch.randperm(total)
    split_i = int(math.floor(total * (1 - p_right)))
    return indices[:split_i], indices[split_i:]


def approx_kl_div(P_lls, Q_lls):
    Px = P_lls.exp()
    return (Px * (P_lls - Q_lls)).sum()


def average_modules(m1, m2):
    m1_sd = m1.state_dict()
    m2_sd = m2.state_dict()
    return average_state_dicts(m1_sd, m2_sd)


def average_state_dicts(m1_sd, m2_sd):
    if not isinstance(m1_sd, dict) and m1_sd == m2_sd:
        return m1_sd
    out_dict = {}
    for k in m1_sd.keys():
        if isinstance(m1_sd[k], torch.Tensor):
            out_dict[k] = (m1_sd[k] + m2_sd[k]) / 2
        elif isinstance(m1_sd, dict):
            out_dict[k] = average_state_dicts(m1_sd[k], m2_sd[k])
        elif m1_sd[k] == m2_sd[k]:
            out_dict[k] = m1_sd[k]
        else:
            raise ValueError(
                dedent(
                    f"""\
                Could not average state_dict values for key {k}:
                Could not average {m1_sd[k]} and {m2_sd[k]}
                """
                )
            )
        # elif isinstance(m1_sd, list) and len(m1_sd) == len(m2_sd):
        #     out_dict[k] = [
        #     ]
    return out_dict