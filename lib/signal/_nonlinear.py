import torch


def nonlinear(input, gain=1, type=1):
    """
    mono in mono out

    This function applies a nonlinear transformation to the input tensor using the gain parameter.
    The output tensor is computed as 1 - exp(input * gain).

    Args:
    - input: a tensor of any shape
    - gain: a scalar value to control the amount of nonlinearity applied to the input tensor

    Returns:
    - output: a tensor of the same shape as the input tensor, with the nonlinear transformation applied
    """
    if type == 1:
        return 1 - torch.exp(input * gain)

    if type == 2:
        return 1 - torch.exp(input * (input + 1) * gain)
