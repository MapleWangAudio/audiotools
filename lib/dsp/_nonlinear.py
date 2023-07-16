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
    input, up_limit=0, down_limit=0, up_ratio=-1, down_ratio=-1, up_knee=0, down_knee=0
):
    """
    mono in mono out
    使用dB作为单位
    Args:
    - input: a tensor of any shape
    - up_limit: 上限
    - down_limit: 下限
    - up_ratio: 超过上限的增益比例
    - down_ratio: 低于下限的增益比例
    - up_knee: 上限的knee
    - down_knee: 下限的knee
    """
    up_limit = dB_to_amplitude(up_limit)
    down_limit = dB_to_amplitude(down_limit)
    up_knee = dB_to_amplitude(up_knee)
    down_knee = dB_to_amplitude(down_knee)

    if input > 0:
        if input > (up_limit + up_knee):
            output = input + (input - up_limit) * up_ratio
        elif input > (up_limit - up_knee):
            output = input
        else:
            output = input
    else:
        input = -input
        if input > (down_limit + down_knee):
            output = input + (input - down_limit) * down_ratio
        elif input > (down_limit - down_knee):
            output = input
        else:
            output = input
        output = -input

    return output
