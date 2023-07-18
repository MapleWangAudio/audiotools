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
    posi_limit_dB=0,
    nega_limit_dB=0,
    posi_ratio=-1,
    nega_ratio=-1,
    posi_knee_dB=0,
    nega_knee_dB=0,
):
    """
    mono in mono out
    Args:
    - input: a tensor of any shape
    """
    posi_limit_up = dB_to_amplitude(posi_limit_dB - posi_knee_dB)
    posi_limit_down = dB_to_amplitude(posi_limit_dB + posi_knee_dB)
    posi_limit = dB_to_amplitude(posi_limit_dB)
    nega_limit_up = dB_to_amplitude(nega_limit_dB - nega_knee_dB)
    nega_limit_down = dB_to_amplitude(nega_limit_dB + nega_knee_dB)
    nega_limit = dB_to_amplitude(nega_limit_dB)

    if input > 0:
        if input > (posi_limit_up):
            output = posi_limit + (input - posi_limit) * posi_ratio
        elif input > (posi_limit_down) and (posi_knee_dB != 0):
            mix = (input - posi_limit_down) / (posi_limit_up - posi_limit_down)
            posi_ratio = posi_ratio * mix + 1 - mix
            output = posi_limit + (input - posi_limit) * posi_ratio
        else:
            output = input
    else:
        input = -input

        output = input

        output = -output

    return output
