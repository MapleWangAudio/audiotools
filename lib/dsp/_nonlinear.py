import torch
from ..data import dB_to_amplitude


def nonlinear_exp(input, gain=1, type=1):
    """
    mono in mono out

    This function applies a nonlinear transformation to the input tensor using the gain parameter.

    type 0: 1 - exp(x^2 * gain), 纯偶次
    type 1: 1 - exp(x * gain), 偶次+奇次
    Args:
    - input: a tensor of any shape
    - gain: a scalar value to control the amount of nonlinearity applied to the input tensor

    Returns:
    - output: a tensor of the same shape as the input tensor, with the nonlinear transformation applied
    """
    if type == 0:
        output = 1 - torch.exp(input * input * gain)
        return output

    if type == 1:
        output = 1 - torch.exp(input * gain)
        return output


def nonlinear_normal(
    input,
    positive_limit_dB=0,
    negative_limit_dB=0,
    positive_ratio=-1,
    negative_ratio=-1,
    positive_knee_dB=0,
    negative_knee_dB=0,
):
    """
    mono in mono out
    Args:
    - input: a tensor of any shape
    """
    positive_limit_up = dB_to_amplitude(positive_limit_dB + positive_knee_dB)
    positive_limit_down = dB_to_amplitude(positive_limit_dB - positive_knee_dB)
    positive_limit = dB_to_amplitude(positive_limit_dB)
    negative_limit_up = dB_to_amplitude(negative_limit_dB + negative_knee_dB)
    negative_limit_down = dB_to_amplitude(negative_limit_dB - negative_knee_dB)
    negative_limit = dB_to_amplitude(negative_limit_dB)

    if input > 0:
        if input > (positive_limit_up):
            output = positive_limit + (input - positive_limit) * positive_ratio
        elif input > (positive_limit_down) and (positive_knee_dB != 0):
            mix = (input - positive_limit_down) / (
                positive_limit_up - positive_limit_down
            )
            positive_ratio = 1 - (1 - positive_ratio) * mix
            output = positive_limit + (input - positive_limit) * positive_ratio
        else:
            output = input
    else:
        input = -input

        output = input

        output = -output

    return output
