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

    quant_prob(x, b) = max(0, 1 - abs(b - x))

This looks like a simple isosceles triangle centered at the contiuous value.
However, any function where quant_prob(x, b) + quant_prob(x, b + 1) == 1 can be
used.
This code is designed to also work with the cubic polynomial

    quant_prob(x, b) = 3y**2 - 2y**3
    where y = max(0, 1 - abs(b - x))


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
edges of the distribution.
All non-edge bins correspond to a symmetric component of the PDF about the bin
center.
Consequently, the expected value can be computed as just the weighted average
over the bin centers:

    E(x) = sum(b: bin_weight[b] * bin_center[b])

However, if the weight of the edge bins are non-zero, this result may be correct.
Unfortunately there's no obvious way to fix this, since how far ourside of the
edge bins samples are is not known.
"""

import torch


def linear_quant_prob(x: torch.Tensor, bin_centers: torch.Tensor):
    bin_sep = bin_centers[:-1] - bin_centers[1:]
    left_bins = torch.bucketize(x, bin_centers)
    bin_sep_x = bin_sep[None][left_bins]
    v = torch.clamp(
        1 - (x[:, None] - bin_centers[None, :]).abs() / bin_sep_x, min=0
    )
    assert v.shape == bin_centers.shape
    # If x is before the first bin all prob is to first bin
    # Same with after last bin
    v[x < bin_centers[0], 0] = 1.0
    v[x > bin_centers[-1], -1] = 1.0
    return v


def test_linear_quant_prob():
    x = torch.linspace(-6, 6, 7)
    bin_centers = torch.linspace(-2, 2, 5)
    probs = linear_quant_prob(x, bin_centers)
    assert probs.shape == (7, 5)
    prob_sum_per_x = probs.sum(dim=1)

    assert torch.allclose(prob_sum_per_x, torch.ones(7))

    assert (
        probs
        >= torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.3, 0.3, 0.0, 0.0, 0.0],
                [0.0, 0.3, 0.3, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.3, 0.3, 0.0],
                [0.0, 0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.0, 0.0, 0.1],
            ]
        )
    ).all()


def cubic_quant_prob(x: torch.Tensor, bin_centers: torch.Tensor):
    y = linear_quant_prob(x, bin_centers)
    return 3 * y**2 - 2 * y**3


def expected_value(bin_centers: torch.Tensor, bin_weights: torch.Tensor):
    assert bin_centers.shape == bin_weights.shape
    return (bin_centers * bin_weights).sum()


# def sample(bin_centers: torch.Tensor, bin_weights: torch.Tensor)
