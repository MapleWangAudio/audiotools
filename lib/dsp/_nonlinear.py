import torch
import torchaudio
from ..data import dB_to_amplitude


def exp(input, gain=1, type=1):
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


def clip(
    input,
    posi_limit_dB=0,
    nega_limit_dB=0,
    posi_one_over_ratio=0,
    nega_one_over_ratio=0,
    posi_knee_dB=0,
    nega_knee_dB=0,
):
    """
    mono in mono out
    input: a tensor
    limit_dB: the dB value of the limit
    ratio: the 1/ratio of the limit
    knee_dB: the dB value of the knee,负无穷-0dB
    """
    input_dB = torchaudio.transforms.AmplitudeToDB("amplitude")(abs(input))
    posi_limit_up = dB_to_amplitude(posi_limit_dB - posi_knee_dB)
    posi_limit_down = dB_to_amplitude(posi_limit_dB + posi_knee_dB)
    posi_limit = dB_to_amplitude(posi_limit_dB)
    if posi_knee_dB != 0:
        posi_one_over_ratio_knee = (
            posi_limit
            - posi_limit_down
            + (posi_limit_up - posi_limit) * posi_one_over_ratio
        ) / (posi_limit_up - posi_limit_down)
    nega_limit_up = dB_to_amplitude(nega_limit_dB - nega_knee_dB)
    nega_limit_down = dB_to_amplitude(nega_limit_dB + nega_knee_dB)
    nega_limit = dB_to_amplitude(nega_limit_dB)
    if nega_knee_dB != 0:
        nega_one_over_ratio_knee = (
            nega_limit
            - nega_limit_down
            + (nega_limit_up - nega_limit) * nega_one_over_ratio
        ) / (nega_limit_up - nega_limit_down)

    if input > 0:
        if input > (posi_limit_up):
            output = posi_limit + (input - posi_limit) * posi_one_over_ratio
        elif input > (posi_limit_down) and (posi_knee_dB != 0):
            mix = (input_dB - posi_limit_dB + posi_knee_dB) / (2 * posi_knee_dB)
            posi_one_over_ratio_knee = posi_one_over_ratio_knee * (1 - mix) + mix
            output = (
                posi_limit_down + (input - posi_limit_down) * posi_one_over_ratio_knee
            )
        else:
            output = input
    else:
        input = -input

        if input > (nega_limit_up):
            output = nega_limit + (input - nega_limit) * nega_one_over_ratio
        elif input > (nega_limit_down) and (nega_knee_dB != 0):
            mix = (input_dB - nega_limit_dB + nega_knee_dB) / (2 * nega_knee_dB)
            nega_one_over_ratio_knee = nega_one_over_ratio_knee * (1 - mix) + mix
            output = (
                nega_limit_down + (input - nega_limit_down) * nega_one_over_ratio_knee
            )
        else:
            output = input

        output = -output

    return output
