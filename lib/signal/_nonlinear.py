import torch


def nonlinear(input, gain=1, type=1):
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

