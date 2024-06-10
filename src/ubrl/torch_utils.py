"""A variety of small utilities for working with pytorch.

These utilities are not specific to reinforcement learning, and generally fall
into a few categories:

- A family of functions for packing / unpacking / padding / unpadding tensors.
  Mostly used for combining tensors from different tensors to batch operations
  across episodes.
- Tools for allowing the user to express probabilistic actions without needing
  to construct complex torch.distributions.Distribution product types.
- Filling in minor "holes" in torch's API, like an automatic normalization
  module, modules for putting commonly used functions into modules, convenient
  functions for constructing networks from high-level descriptions.
"""

import copy
import dataclasses
from dataclasses import dataclass
from typing import (
    TypeVar,
    Callable,
    Generator,
    List,
    Optional,
    Sequence,
    Union,
    Callable,
    Tuple,
    Optional,
)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

T = TypeVar("T")

Sizes = Union[Tuple[int, ...], List[int]]
"""Typically used for "hidden_sizes" of an MLP."""


class CustomTorchDist:
    """Simple API for allowing custom distributions.

    Only includes the bare minimum used by OutRL.

    Methods returning NotImplementedError() will lead to approximation from
    log-likelihoods being used.
    """

    def entropy(self) -> torch.Tensor:
        raise NotImplementedError()

    def kl_div(self, other: "CustomTorchDist") -> torch.Tensor:
        del other
        raise NotImplementedError()


ActionDist = Union[
    torch.distributions.Distribution,
    CustomTorchDist,
    list["ActionDist"],
]


class NonLinearity(nn.Module):
    """Wrapper class to convert a non linear function into a Module.

    Makes a copy when wrapped around a module.

    Args:
        non_linear (callable or type): Non-linear function or type to be
            wrapped.

    """

    def __init__(self, non_linearity: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.function = copy.deepcopy(non_linearity)

    def forward(self, x: torch.Tensor):
        """Calls the wrapped non-linearity."""
        return self.function(x)

    def __repr__(self):
        return repr(self.function)


def wrap_nonlinearity(
    non_linearity: Union[Callable[[torch.Tensor], torch.Tensor], nn.Module]
) -> nn.Module:
    """Convert a non-linearity (possibly a function) into a Module."""
    if isinstance(non_linearity, nn.Module):
        return copy.deepcopy(non_linearity)
    elif isinstance(non_linearity, type):
        return non_linearity()
    else:
        return NonLinearity(non_linearity)


class SqueezeModule(nn.Module):
    """Module that squeezes output.

    Used to produce neural networks that produce "scalar" outputs.

    Args:
        dim (Optional[int]): Index to squeeze.

    """

    def __init__(self, dim: Optional[int] = None):
        super().__init__()

        self.dim = dim

    def forward(self, x: torch.Tensor):
        """Removes specified unit dimensions."""
        return x.squeeze(self.dim)

    def __repr__(self):
        return f"SqueezeModule(dim={self.dim})"


def flatten_shape(shape: Union[int, Tuple[int, ...], list[int]]) -> int:
    """Flatten a shape down to a integer size."""
    if isinstance(shape, int):
        return shape
    else:
        return math.prod(shape)


def make_mlp(
    *,
    input_size: int,
    hidden_sizes: Sizes,
    output_size: Optional[int] = None,
    hidden_nonlinearity: Union[
        Callable[[torch.Tensor], torch.Tensor], nn.Module
    ] = nn.SiLU,
    hidden_w_init: Callable[[torch.Tensor], None] = nn.init.xavier_normal_,
    hidden_b_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
    output_w_init: Callable[[torch.Tensor], None] = nn.init.xavier_normal_,
    output_b_init: Callable[[torch.Tensor], None] = nn.init.zeros_,
    use_dropout: bool = True,
    layer_normalization: bool = False,
) -> nn.Sequential:
    """Helper utility to set up a simple feed forward neural network from an
    input size and series of hidden_sizes.

    If output_size == 0, will squeeze the output down to a "scalar" (removes
    non-batch dimensions on the output).

    Optionally adds dropout and layer_normalization.

    Returns a nn.Sequential of nn.Sequential.
    """

    layers = nn.Sequential()
    prev_size = input_size
    for size in hidden_sizes:
        step_layers = nn.Sequential()
        if use_dropout:
            step_layers.add_module("dropout", nn.Dropout())
        if layer_normalization:
            step_layers.add_module("layer_normalization", nn.LayerNorm(prev_size))
        linear_layer = nn.Linear(prev_size, size)
        hidden_w_init(linear_layer.weight)
        hidden_b_init(linear_layer.bias)
        step_layers.add_module("linear", linear_layer)

        if hidden_nonlinearity:
            step_layers.add_module(
                "non_linearity", wrap_nonlinearity(hidden_nonlinearity)
            )
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

    return layers


def explained_variance(ypred: torch.Tensor, y: torch.Tensor):
    """Explained variation for 1D inputs.

    Similar to R^2 value but also requires the prediction values to have the
    correct constant bias.

    The proportion of the variance in one variable that is explained or
    predicted from another variable.

    Interpretation:

        1 -> all variance explained
        0 -> not predicting anything
        negative values -> overfit and predicting badly

    Args:

        ypred (torch.Tensor): Predicted values.

        y (torch.Tensor): Target values.

    Returns:

        float: The explained variance.

    """
    vary = torch.var(y)

    # Handle corner cases
    if vary == 0:
        if torch.var(ypred) > 0:
            return torch.tensor(0, device=y.device, dtype=y.dtype)

        return torch.tensor(1, device=y.device, dtype=y.dtype)

    epsilon = 1e-8

    res = 1 - (torch.var(y - ypred) + epsilon) / (vary + epsilon)
    return res


class RunningMeanVar(nn.Module):
    """Calculates running mean and variance of sequence of tensors.

    When used as a module, records inputs such that outputs will become zero
    mean and unit variance.

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
        use_mean: bool = True,
        use_var: bool = True,
    ):
        """
        Args:

            trainable (bool): If true, causes mean and var to be nn.Parameters
                tunable via SGD (and also optimized by inputs).
        """

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

        self.use_mean: bool = use_mean
        """Whether to normalize the data to 0 mean."""

        self.use_var: bool = use_var
        """Whether to normalize the data to unit variance."""

    def normalize_batch(self, x: torch.Tensor, correction=1) -> torch.Tensor:
        """Normalizes a batch of data, but does not update the recorded statistics."""
        # Don't use in-place operations in this method!
        y = x
        if self.use_mean:
            y = y - self.mean
        if self.use_var:
            y = y / torch.sqrt(self.corrected_var(correction))
        if self.clip_max:
            y = torch.clamp(y, min=-self.clip_max, max=self.clip_max)
        return y

    def denormalize_batch(self, y: torch.Tensor, correction=1) -> torch.Tensor:
        """Reverse of the normalization operation."""
        # Don't use in-place operations in this method!
        x = y
        if self.use_var:
            y = y * torch.sqrt(self.corrected_var(correction))
        if self.use_mean:
            x = x + self.mean
        return x

    def forward(self, x: torch.Tensor):
        """Updates the statistics and normalizes the batch."""
        self.update(x)
        return self.normalize_batch(x)

    def update(self, x: torch.Tensor):
        """Update the statistics using a batch."""
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

    def corrected_var(self, correction=1):
        """Computes the corrected variance."""
        d = self.count - correction
        if d <= 0:
            d = 1
        cvar = self.var * self.count / d
        return torch.clamp(cvar, self.min_var)

    def noko_preprocess(self, prefix: str, dst: dict):
        """Provide statistics for the noko logging library."""
        dst[f"{prefix}.var"] = self.var.mean().item()
        dst[f"{prefix}.mean"] = self.mean.mean().item()
        dst[f"{prefix}.count"] = self.count


def truncate_packed(
    tensor: torch.Tensor, new_lengths: list[int], to_cut: int
) -> torch.Tensor:
    """Truncate a packed tensor by a fixed number of elements along the first dimension."""
    assert tensor.shape[0] == sum(new_lengths) + to_cut * len(new_lengths)
    unpacked = unpack_tensors(tensor, [length + to_cut for length in new_lengths])
    repacked = pack_tensors([t[:-to_cut] for t in unpacked])[0]
    assert repacked.shape[0] == sum(new_lengths)
    return repacked


def pack_tensors(tensor_list: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    """Concatenates input tensors along first dimension and returns lengths of
    tensors before concatenation.

    Inverse of unpack_tensors.
    """
    return (torch.cat(tensor_list)), [len(t) for t in tensor_list]


def pack_tensors_check(
    tensor_list: list[torch.Tensor], expected_lengths: list[int]
) -> torch.Tensor:
    """Concatenates input tensors along first dimension.

    Asserts that input tensor lengths match expected_lengths.

    Partial inverse of unpack_tensors, when lengths have already been computed.
    """
    assert expected_lengths == [len(t) for t in tensor_list]
    return torch.cat(tensor_list)


def unpack_tensors(tensor: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
    """Converts input tensor into a sequence of tensor given lengths.

    Inverse of pack_tensors.
    """
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
    """Converts a list of Tensors with possibly different lengths into a single
    tensor with a new first dimension and second dimension equal to the largest
    first dimension among input tensors.
    """

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
    """Convert a padded Tensor back into a list given the lengths before padding."""
    assert all(padded.shape[1] >= length for length in lengths)
    return [padded[i, :length] for i, length in enumerate(lengths)]


def pack_padded(padded: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Equivelant to pack_tensors_check(unpad_tensors(...))."""
    return pack_tensors_check(unpad_tensors(padded, lengths), lengths)


def pad_packed(padded: torch.Tensor, lengths: list[int]) -> torch.Tensor:
    """Equivelant to pad_tensors(unpack_tensors(...))."""
    return pad_tensors(unpack_tensors(padded, lengths))


@dataclass(eq=False)
class DictDataset(torch.utils.data.Dataset):
    """A simple in-memory map-style torch.utils.data.Dataset.

    Constructed from a dictionary of equal-length tensors, and produces
    dictionaries of tensors as data-points.

    Has a minibatches() method as a faster alternative to using a DataLoader.
    """

    def __init__(self, elements: Optional[dict[str, torch.Tensor]] = None, **kwargs):
        if elements is None:
            self.elements = kwargs
        else:
            assert len(kwargs) == 0
            self.elements = elements
        # Check length consistencies
        assert len(self) > 0

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
        """Split dataset into two smaller, non-overlapping datasets."""
        ind_left, ind_right = split_shuffled_indices(len(self), p_right)
        left = {k: v[ind_left] for (k, v) in self.elements.items()}
        right = {k: v[ind_right] for (k, v) in self.elements.items()}
        return left, right

    def minibatches(
        self, minibatch_size: int, drop_last: bool = False
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Iterator over minibatches in the dataset. Minimal alternative to
        using a DataLoader."""
        indices = torch.randperm(len(self))
        start_i = 0
        while start_i + minibatch_size < len(indices):
            yield self[indices[start_i : start_i + minibatch_size]]
            start_i += minibatch_size
        if start_i != len(indices) and not drop_last:
            yield self[indices[start_i:]]


def split_shuffled_indices(
    total: int, p_right: float = 0.5
) -> tuple[torch.IntTensor, torch.IntTensor]:
    """Randomly partition indices from 0 to total - 1 into two tensors."""
    indices = torch.randperm(total)
    split_i = int(math.floor(total * (1 - p_right)))
    return indices[:split_i], indices[split_i:]


def softmax_clip(x: torch.Tensor, max_exp: torch.Tensor) -> torch.Tensor:
    """Compute softmax of input, clipping exponential values to a
    maximal value.
    """
    x_exp = x.exp()
    clip_mask = ~torch.isfinite(x_exp) | (x_exp > max_exp)
    clipped_x_exp = x_exp.clone()
    clipped_x_exp[clip_mask] = max_exp
    norm_clipped_x_exp = clipped_x_exp / clipped_x_exp.sum()
    return norm_clipped_x_exp


def approx_entropy_of(P_lls: torch.Tensor) -> torch.Tensor:
    """Approximate the entropy from log likelihoods."""
    return -P_lls


def approx_kl_div_of(P_lls: torch.Tensor, Q_lls: torch.Tensor) -> torch.Tensor:
    """A simpler alternative to calling torch.nn.functional.kl_div with the
    arguments in opposite order and the log_target flag set."""
    Px = P_lls.exp()
    return Px * (P_lls - Q_lls)


def kl_div_of(
    p_dist: Union[ActionDist, list[ActionDist]],
    q_dist: Union[ActionDist, list[ActionDist]],
) -> torch.Tensor:
    """Compute the KL divergence for each timestep in p_dist and q_dist.

    p_dist and q_dist can be lists of per-timestep action distributions, or a
    single distribution with a timestep batch dimension.

    The distribution can either be a torch.distributions.Distribution, or any
    other object that implements a self.kl_div(other) method, as shown in the
    CustomTorchDist class.

    The return value of this function should be differentiable back to
    parameters.
    """

    if isinstance(p_dist, list):
        assert isinstance(q_dist, list)
        assert len(p_dist) == len(q_dist)
        return torch.cat([kl_div_of(p, q) for (p, q) in zip(p_dist, q_dist)])
    elif isinstance(p_dist, torch.distributions.Distribution):
        assert isinstance(q_dist, torch.distributions.Distribution)
        return torch.distributions.kl.kl_divergence(p_dist, q_dist)
    else:
        # Presumably implements the CustomTorchDist API
        return p_dist.kl_div(q_dist)


def entropy_of(
    p_dist: Union[ActionDist, list[ActionDist]],
) -> torch.Tensor:
    """Compute the KL divergence for each timestep in p_dist and q_dist.

    p_dist and q_dist can be lists of per-timestep action distributions, or a
    single distribution with a timestep batch dimension.

    The distribution can either be a torch.distributions.Distribution, or any
    other object that implements a self.kl_div(other) method, as shown in the
    CustomTorchDist class.

    The return value of this function should be differentiable back to
    parameters.
    """

    if isinstance(p_dist, list):
        return torch.cat([entropy_of(p) for p in p_dist])
    else:
        return p_dist.entropy()


def force_concat(elements: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenates data recursively, handling sequences of tensors, scalar
    Tensors, and multi-dimensional tensors.

    Probably only useful for logging.
    """
    if not isinstance(elements, list):
        elements = list(elements)
    if isinstance(elements[0], torch.Tensor):
        if not elements[0].shape:
            return torch.stack(elements)
        else:
            return torch.cat(elements)
    else:
        return torch.tensor(elements)


def clamp_identity_grad(
    value: torch.Tensor,
    min: Optional[torch.Tensor | float] = None,
    max: Optional[torch.Tensor | float] = None,
):
    """A clamp operation that has the same gradients as the
    identity function (i.e. dy/dx = 1).

    See also `soft_clamp`, which likely produces more useful
    gradients at the cost of non-linear behavior.

    """
    if min is not None:
        min_error = torch.clamp(min - value, min=0.0)
        value = value + min_error
    if max is not None:
        max_error = torch.clamp(value - max, min=0.0)
        value = value - max_error
    return value


def soft_clamp(
    value: torch.Tensor,
    min: Optional[torch.Tensor | float] = None,
    max: Optional[torch.Tensor | float] = None,
    scale: Optional[torch.Tensor | float] = None,
) -> torch.Tensor:
    """Implements smooth clamping using tanh."""
    if min is None and max is None:
        return value
    elif min is not None and max is not None:
        midpoint = (max + min) / 2
        if scale is None:
            scale = (max - min) / 2
        x = (value - midpoint) / scale
        y = (torch.tanh(x) * scale) + midpoint
        assert (min <= y).all()
        assert (y <= max).all()
        return y
    elif min is not None and max is None:
        assert scale is not None
        zero_point = min + scale
        x = (value - zero_point) / scale
        y = (torch.tanh(x) * scale) + zero_point
        res = torch.where(value > zero_point, value, y)
        assert (min <= res).all()
        return res
    else:
        assert min is None and max is not None
        assert scale is not None
        zero_point = max - scale
        x = (value - zero_point) / scale
        y = (torch.tanh(x) * scale) + zero_point
        res = torch.where(value < zero_point, value, y)
        assert (res <= max).all()
        return res


class NOPLRScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Learning rate scheduler that just uses a constant learning rate."""

    def get_lr(self):
        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return list(self.base_lrs)


def make_scheduler(
    opt, name: Optional[str], start: float, end: float, expected_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a learning rate scheduler from a string name, start and end
    learning rates, and expected number of training steps."""
    if name is None:
        return NOPLRScheduler(opt)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1.0, end_factor=end / start, total_iters=expected_steps
        )
    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, eta_min=end, T_max=expected_steps
        )
    else:
        raise NotImplementedError(f"LR scheduling of type {name} is not implemented")


def used_for_logging(*args):
    """No-op function used to document that a local variable is used for
    logging and should not be deleted.

    This function is preferred over just leaving a comment since it will also
    "inform" the linter.
    """
    del args


def discount_cumsum(x: torch.Tensor, discount: float) -> torch.Tensor:
    """Discounted cumulative sum."""
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
    x_pad = torch.cat([x, torch.zeros_like(x[:, :, :-1])], dim=-1)
    returns = F.conv1d(x_pad, weights, stride=1)
    assert returns.shape == (B, 1, L)
    return returns.squeeze()


def concat_lists(lists: list[list[T]]) -> list[T]:
    new_list = []
    for l in lists:
        new_list.extend(l)
    return new_list
