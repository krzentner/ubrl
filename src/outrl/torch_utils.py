import copy
from dataclasses import dataclass, is_dataclass, fields, replace
from textwrap import dedent
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

SupportsNonlinearity = Union[Callable[[torch.Tensor], torch.Tensor], nn.Module]
Initializer = Callable[[torch.Tensor], None]
Shape = Union[int, Tuple[int, ...]]
Sizes = Union[int, Tuple[int, ...], List[int]]


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
    list[Union[torch.distributions.Distribution, CustomTorchDist]],
]


class NonLinearity(nn.Module):
    """Wrapper class to convert a non linear function into a Module.

    Makes a copy when wrapped around a module.

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

    Used to produce neural networks that produce "scalar" outputs.

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
    """Flatten a shape down to a integer size."""
    if isinstance(shape, int):
        return shape
    else:
        return math.prod(shape)


def make_mlp(
    *,
    input_size: int,
    hidden_sizes: List[int],
    output_size: Optional[int] = None,
    hidden_nonlinearity: SupportsNonlinearity = nn.SiLU,
    hidden_w_init: Initializer = nn.init.xavier_normal_,
    hidden_b_init: Initializer = nn.init.zeros_,
    output_w_init: Initializer = nn.init.xavier_normal_,
    output_b_init: Initializer = nn.init.zeros_,
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

    return layers


def explained_variance(ypred: torch.Tensor, y: torch.Tensor):
    """Explained variation for 1D inputs.

    Similar to R^2 value but using corrected variances.

    It is the proportion of the variance in one variable that is explained or
    predicted from another variable.

    interpretation:
        1 => all variance explained
        0 => not predicting anything
        <0 => overfit and predicting badly

    Args:
        ypred (torch.Tensor): Sample data from the first variable.
            Shape: :math:`(N, max_episode_length)`.
        y (torch.Tensor): Sample data from the second variable.
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


def sample_or_mode(dist: torch.distributions.Distribution, get_mode: bool = False):
    """Samples from input distribution unless get_mode is True.

    When get_mode is True, returns the mode of the input distribution.

    Allows deterministic evaluation of the most likely sequence of actions from
    apolicy, which often (but not always) perform better than sampling actions.
    """

    if get_mode:
        return dist.mode
    else:
        return dist.sample()


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
        self.use_var: bool = use_var

    def normalize_batch(self, x: torch.Tensor, correction=1) -> torch.Tensor:
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
        # Don't use in-place operations in this method!
        x = y
        if self.use_var:
            y = y * torch.sqrt(self.corrected_var(correction))
        if self.use_mean:
            x = x + self.mean
        return x

    def forward(self, x: torch.Tensor):
        self.update(x)
        return self.normalize_batch(x)

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


def pack_recursive(data: Union[list[T], T]) -> T:
    """Recursively pack tensors contained in dataclasses, dictionaries, lists."""
    if isinstance(data, (list, tuple)):
        if is_dataclass(data[0]):
            return replace(
                data[0],
                **{
                    field.name: pack_recursive([getattr(d, field.name) for d in data])
                    for field in fields(data[0])
                },
            )
        elif isinstance(data[0], dict):
            return {k: pack_recursive(d[k] for d in data) for k in data[0].keys()}
        elif isinstance(data[0], torch.Tensor):
            return pack_tensors(data)[0]
    return data


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
    """A simple in-memory map-stle torch.utils.data.Dataset.

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


def approx_kl_div(P_lls: torch.Tensor, Q_lls: torch.Tensor) -> torch.Tensor:
    """A simpler alternative to calling torch.nn.functional.kl_div with the
    arguments in opposite order and the log_target flag set."""
    Px = P_lls.exp()
    return Px * (P_lls - Q_lls)


def kl_div(
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
        return torch.cat([kl_div(p, q) for (p, q) in zip(p_dist, q_dist)])
    elif isinstance(p_dist, torch.distributions.Distribution):
        assert isinstance(q_dist, torch.distributions.Distribution)
        return torch.distributions.kl.kl_divergence(p_dist, q_dist)
    else:
        # Presumably implements the CustomTorchDist API
        return p_dist.kl_div(q_dist)


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
    return out_dict


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
    else:
        raise NotImplementedError(f"LR scheduling of type {name} is not implemented")


def used_for_logging(*args):
    """No-op function used to document that a local variable is used for
    logging and should not be deleted.

    This function is preferred over just leaving a comment since it will also
    "inform" the linter.
    """
    del args
