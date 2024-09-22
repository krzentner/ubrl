"""Utilities for working with the "Two-Hot" encoding of continuous values.

## Why does this code exist?

Sometimes, when probabilistically modelling a continuous random variable (e.g.
reward in an MDP), the form of the probability distribution of that variable is
not known. It may be multi-modal, or very flat across a wide range.
If the precision requirements and domain of the random variable are known, one simpe solution to modelling it is to quantize the variable and model it using a categorical distribution.

However, when computing the expected value of such a distribution, quantization
introduces significant additional variance, especially if the number of
"quantization bins" is low.
This reduces the training signal when used in a deep learning context.


## How does this code solve the above problem?

A simple solution to the variance introduced in quantization is to weight two
of the bins based on how close the pre-quantized value is to the center of that
bin.
This is equivalent to treating the quantization process itself as a stochastic
process, where first the continuous value is quantized to a bin based on a
"quantization distribution", which is generally a fixed distribution relative
to the location of the continuous value.

The probability of quantization to a particular value is determined by a
function, in this code called quant_prob().
The typical choice for this function (e.g. in the TD-MPC2 code) is to use

    quant_prob(x, b) = max(0, (1 - abs(b - x)) / bin_separation(b, x))

This looks like a simple isosceles triangle centered at the contiuous value.
However, any function where quant_prob(x, b) + quant_prob(x, b + 1) == 1 can be
used.
This code is designed to also work with the cubic polynomial

    quant_prob(x, b) = 3y**2 - 2y**3
    where y = max(0, 1 - abs(b - x) / bin_separation(b, x))


## Training

Training a function approximator to model a conditional probability using this
distribution is extremely simple.
Instead of computing a one-hot and applying a cross-entropy loss, instead
compute the two-hot encoding (and still apply the cross-entropy loss).


## Sampling from the continuous distribution.

Sampling from the _quantized_ distribution is extremely straightforward since it
is just a categorical distribution.
Sampling a continuous value from the "original" distribution is much more
complicated.
In particular, any high-frequency information in the original distribution's
PDF has been lost in the quantization process.
The straightforward process to perform is quantization in reverse:

 1. Choose a bin using the categorical distribution.
 2. De-quantize by sampling from a continuous distribution centered on the bin

The form of that distribution is partially implied by Bayes Theorem applied to
the _quantization_ distribution.

    P(X=x|B=b) = P(B=b|X=x) * P(B=b) / P(X=x)

Unfortunately, P(X=x) is unknown. If we assume it is approximately constant
over the bucket and its neighbors, then we can simplify to

    P(X=x|B=b) o< P(B=b|X=x) * P(B=b)
    P(X=x|B=b) o< quant_prob(x, b) * P(B=b)

In other words, once the bin has been sampled, the PDF or the de-quantization
function should just be the normalized quant_prob() function.

With the (rectified) linear quant_prob, this is equivalent to approximating the
PDF of the continuous distribution using a linear interpolation of points from
the continuous distribution.

With the cubic quant_prob, this is equivalent to approximating the PDF using a
Bezier curve instead.


## Expected value

Computing the exact expected value of a two-hot distribution is complicated by
the edges of the distribution and non-uniform spacing of the bin centers.
If all edge bins are zero and all spacing is uniform, then each bin corresponds
to a symmetric component of the PDF, centers at the bin center.
Consequently, the expected value can be computed as just the weighted average
over the bin centers:

    E(x) = sum(b: bin_weight[b] * bin_center[b])

However, if the spacing is non-uniform, the contribution of each bin is no longer centered on the bin_center.
To correct for this, compute the mean contribution of each side of the bin, and combine those into a bin_mean to use in place of bin_center.

    bin_mean[b] = (bin_left_mean(b) + bin_right_mean(b)) / 2

To find the bin_left_mean(b), integrate the x * quant_prob(x, b) function over the left side of the bin, and similarly for the right.
See notes/two_hot.qmd for derivations using sympy.
In the below descriptions, bin_sep[b] = bin_center[b + 1] - bin_center[b].
See discussion below for handling the edge bins where this would exceed the bounds of the bin_center array.

For the linear case, this results in:

    bin_mean[b] = bin_center[b] * (bin_sep[b - 1] + bin_sep[b]) / 4 - (bin_sep[b - 1]**2 + bin_sep[b]**2)/12

For the cubic case, this results in:

    bin_mean[b] = bin_center[b] * (bin_sep[b - 1] + bin_sep[b])/4 - 3 * (bin_sep[b - 1]**2 + bin_sep[b]**2)/40

If the weight of the edge bins are non-zero, this result will almost never be completely correct, since the values in the edge bin may have been clipped from arbitrarily far away from the bin center.
There are various solutions that may be employed for solve this solution.
In this code, we introduce two new parameters to the distribution, which correspond to bin_sep[0] and bin_sep(len(bin_centers) - 1).
"""

from multiprocessing import Value
from typing import Literal

import torch


def linear_quant_prob(
    x: torch.Tensor,
    bin_centers: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
) -> torch.Tensor:
    bin_sep = torch.cat(
        [
            torch.tensor([left_sep]),
            bin_centers[1:] - bin_centers[:-1],
            torch.tensor([right_sep]),
        ]
    )
    left_bins = torch.clamp(
        torch.bucketize(x, bin_centers, right=True), max=len(bin_sep) - 1
    )
    bin_sep_x = bin_sep[left_bins]
    abs_distance = (x[:, None] - bin_centers[None, :]).abs()
    v = torch.clamp(1 - abs_distance / bin_sep_x[:, None], min=0)
    # If x is before the first bin all prob is to first bin
    # Same with after last bin
    v[x < bin_centers[0], 0] = 1.0
    v[x > bin_centers[-1], -1] = 1.0
    assert torch.allclose(v.sum(dim=1), torch.ones(len(x)))
    return v


def test_linear_quant_prob():
    x = torch.tensor([-6, -5, -4, -2, 0, 2, 3, 6], dtype=torch.float)
    bin_centers = torch.tensor([-5, -3, 0, 1, 3], dtype=torch.float)
    probs = linear_quant_prob(x, bin_centers)
    assert probs.shape == (8, 5)
    prob_sum_per_x = probs.sum(dim=1)

    assert torch.allclose(prob_sum_per_x, torch.ones(8))

    assert (
        probs
        >= torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.6, 0.3, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.1],
            ]
        )
    ).all()


def cubic_quant_prob(
    x: torch.Tensor,
    bin_centers: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
) -> torch.Tensor:
    y = linear_quant_prob(
        x=x, bin_centers=bin_centers, left_sep=left_sep, right_sep=right_sep
    )
    return 3 * y**2 - 2 * y**3


def test_cubic_quant_prob():
    x = torch.tensor([-6, -5, -4, -2, 0, 2, 3, 6], dtype=torch.float)
    bin_centers = torch.tensor([-5, -3, 0, 1, 3], dtype=torch.float)
    probs = cubic_quant_prob(x, bin_centers)
    assert probs.shape == (8, 5)
    prob_sum_per_x = probs.sum(dim=1)

    assert torch.allclose(prob_sum_per_x, torch.ones(8))

    assert (
        probs
        >= torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.7, 0.2, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.1],
                [0.0, 0.0, 0.0, 0.0, 0.1],
            ]
        )
    ).all()


QUANT_PROB_TYPE = Literal["linear"] | Literal["cubic"]


def quant_prob(
    x: torch.Tensor,
    bin_centers: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
    quant_prob_type: QUANT_PROB_TYPE = "cubic",
) -> torch.Tensor:
    if quant_prob_type == "linear":
        return linear_quant_prob(
            x=x, bin_centers=bin_centers, left_sep=left_sep, right_sep=right_sep
        )
    elif quant_prob_type == "cubic":
        return cubic_quant_prob(
            x=x, bin_centers=bin_centers, left_sep=left_sep, right_sep=right_sep
        )
    else:
        raise ValueError(f"Unknown quant_prob_type {quant_prob_type!r}")


def linear_bin_means(
    bin_centers: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
) -> torch.Tensor:
    """See notes/two_hot.qmd for the derivation of this function."""
    bin_sep = torch.cat(
        [
            torch.tensor([left_sep]),
            bin_centers[1:] - bin_centers[:-1],
            torch.tensor([right_sep]),
        ]
    )

    bin_means = (
        bin_centers * (bin_sep[:1] + bin_sep[:-1]) / 4
        - 3 * (bin_sep[:1] ** 2 + bin_sep[:-1] ** 2) / 40
    )
    return bin_means


def cubic_bin_means(
    bin_centers: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
) -> torch.Tensor:
    """See notes/two_hot.qmd for the derivation of this function."""
    bin_sep = torch.cat(
        [
            torch.tensor([left_sep]),
            bin_centers[1:] - bin_centers[:-1],
            torch.tensor([right_sep]),
        ]
    )

    bin_means = (
        bin_centers * (bin_sep[:1] + bin_sep[:-1]) / 4
        - (bin_sep[:1] ** 2 + bin_sep[:-1] ** 2) / 12
    )
    return bin_means


def bin_means(
    bin_centers: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
    quant_prob_type: QUANT_PROB_TYPE = "cubic",
) -> torch.Tensor:
    if quant_prob_type == "linear":
        return linear_bin_means(
            bin_centers=bin_centers, left_sep=left_sep, right_sep=right_sep
        )
    elif quant_prob_type == "cubic":
        return cubic_bin_means(
            bin_centers=bin_centers, left_sep=left_sep, right_sep=right_sep
        )
    else:
        raise ValueError(f"Unknown quant_prob_type {quant_prob_type!r}")


def expected_value(
    bin_centers: torch.Tensor,
    bin_weights: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
    quant_prob_type: QUANT_PROB_TYPE = "cubic",
) -> torch.Tensor:
    assert bin_centers.shape == bin_weights.shape
    the_bin_means = bin_means(
        bin_centers=bin_centers,
        left_sep=left_sep,
        right_sep=right_sep,
        quant_prob_type=quant_prob_type,
    )
    return (the_bin_means * bin_weights).sum()


def sample(
    n_samples: int,
    bin_centers: torch.Tensor,
    bin_weights: torch.Tensor,
    left_sep: float = 1.0,
    right_sep: float = 1.0,
    quant_prob_type: QUANT_PROB_TYPE = "cubic",
) -> torch.Tensor:
    assert bin_centers.shape == bin_weights.shape
    bin_sep = torch.cat(
        [
            torch.tensor([left_sep]),
            bin_centers[1:] - bin_centers[:-1],
            torch.tensor([right_sep]),
        ]
    )
    bin_indices = torch.randint(high=len(bin_centers), size=[n_samples])

    # Choose what side of the bin center to sample from
    bin_left_size = bin_sep[:-1]
    bin_right_size = bin_sep[1:]
    prob_right_side_of_bin = (bin_right_size / (bin_left_size + bin_right_size))[
        bin_indices
    ]
    assert prob_right_side_of_bin.shape == (n_samples,)
    right_side_of_bin = prob_right_side_of_bin < torch.rand(n_samples)

    sep_at_side = bin_sep[bin_indices + right_side_of_bin]
    quant_offset = sample_quant_offset(n_samples, quant_prob_type) * sep_at_side
    offset = -quant_offset
    offset[right_side_of_bin] = quant_offset[right_side_of_bin]
    points = bin_centers[bin_indices] + offset
    assert points.shape == (n_samples,)
    return points


def sample_quant_offset(
    n_samples: int, quant_prob_type: QUANT_PROB_TYPE
) -> torch.Tensor:
    x = torch.randn(n_samples)
    sqrt = torch.sqrt
    if quant_prob_type == "linear":
        return 1 - sqrt(1 - x)
    elif quant_prob_type == "cubic":
        return (
            -sqrt(
                -2
                * (x - 1)
                / (
                    3
                    * (
                        -x / 4
                        + sqrt((x / 2 - 1 / 2) ** 2 / 4 + (x - 1) ** 3 / 27)
                        + 1 / 4
                    )
                    ** (1 / 3)
                )
                + 2
                * (-x / 4 + sqrt((x / 2 - 1 / 2) ** 2 / 4 + (x - 1) ** 3 / 27) + 1 / 4)
                ** (1 / 3)
                + 1
            )
            / 2
            + sqrt(
                2
                * (x - 1)
                / (
                    3
                    * (
                        -x / 4
                        + sqrt((x / 2 - 1 / 2) ** 2 / 4 + (x - 1) ** 3 / 27)
                        + 1 / 4
                    )
                    ** (1 / 3)
                )
                - 2
                * (-x / 4 + sqrt((x / 2 - 1 / 2) ** 2 / 4 + (x - 1) ** 3 / 27) + 1 / 4)
                ** (1 / 3)
                + 2
                + 2
                / sqrt(
                    -2
                    * (x - 1)
                    / (
                        3
                        * (
                            -x / 4
                            + sqrt((x / 2 - 1 / 2) ** 2 / 4 + (x - 1) ** 3 / 27)
                            + 1 / 4
                        )
                        ** (1 / 3)
                    )
                    + 2
                    * (
                        -x / 4
                        + sqrt((x / 2 - 1 / 2) ** 2 / 4 + (x - 1) ** 3 / 27)
                        + 1 / 4
                    )
                    ** (1 / 3)
                    + 1
                )
            )
            / 2
            + 1 / 2
        )


import hot_restart

hot_restart.wrap_module()
if __name__ == "__main__" and not hot_restart.is_restarting_module():
    test_linear_quant_prob()
    test_cubic_quant_prob()
