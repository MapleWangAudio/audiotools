import torch
import torchaudio
from ..data import dB_to_amplitude


def exp(input, gain, upLimit, downLimit, upCurve, downCurve, mode=1):
    """
    mono in mono out
    """
    input *= gain

    if mode == 0:
        output = 1 - torch.exp(input * input * gain) + input
        return output

    if mode == 1:
        if input > 0:
            output = upLimit * (1 - torch.exp(-upCurve * input))
        else:
            output = -downLimit * (1 - torch.exp(downCurve * input))

        return output

    if mode == 2:
        if input > 0:
            output = upLimit * (input + 1 / upCurve * torch.exp(-upCurve * input)) - (
                upLimit / upCurve
            )
        else:
            output = -downLimit * (
                input - 1 / downCurve * torch.exp(downCurve * input)
            ) - (downLimit / downCurve)

        return output

    if mode == 3:
        if input > 0:
            output = upLimit * (
                (input * input) / 2
                + torch.exp(-upCurve * input)
                / (upCurve * upCurve)
                * (upCurve * input + 1)
            ) - (upLimit / (upCurve * upCurve))
        else:
            output = -downLimit * (
                (input * input) / 2
                - torch.exp(downCurve * input)
                / (downCurve * downCurve)
                * (downCurve * input - 1)
            ) + (downLimit / (downCurve * downCurve))

        return output


def clip(
    input,
    posi_limit_dB=0,
    nega_limit_dB=0,
    posi_one_over_ratio=0,
    nega_one_over_ratio=0,
    posi_knee_dB=0,
    nega_knee_dB=0,
    mode=0,
):
    """
    mono in mono out
    input: a tensor
    limit_dB: the dB value of the limit
    ratio: the 1/ratio of the limit
    knee_dB: the dB value of the knee,0-正无穷dB
    mode: 0更常规,1作为一种补充
    """
    input_dB = torchaudio.transforms.AmplitudeToDB("amplitude")(abs(input))

    if mode == 0:
        if input > 0:
            if input_dB > (posi_limit_dB + posi_knee_dB / 2):
                output = (
                    posi_limit_dB + (input_dB - posi_limit_dB) * posi_one_over_ratio
                )
                output = dB_to_amplitude(output)
            elif input_dB > (posi_limit_dB - posi_knee_dB / 2) and (posi_knee_dB != 0):
                output = input_dB + (
                    (posi_one_over_ratio - 1)
                    * (input_dB - posi_limit_dB + posi_knee_dB / 2)
                    * (input_dB - posi_limit_dB + posi_knee_dB / 2)
                ) / (2 * posi_knee_dB)
                output = dB_to_amplitude(output)
            else:
                output = input
        else:
            input = -input

            if input_dB > (nega_limit_dB + nega_knee_dB / 2):
                output = (
                    nega_limit_dB + (input_dB - nega_limit_dB) * nega_one_over_ratio
                )
                output = dB_to_amplitude(output)
            elif input_dB > (nega_limit_dB - nega_knee_dB / 2) and (nega_knee_dB != 0):
                output = input_dB + (
                    (nega_one_over_ratio - 1)
                    * (input_dB - nega_limit_dB + nega_knee_dB / 2)
                    * (input_dB - nega_limit_dB + nega_knee_dB / 2)
                ) / (2 * nega_knee_dB)
                output = dB_to_amplitude(output)
            else:
                output = input

            output = -output

        return output

    if mode == 1:
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
                    posi_limit_down
                    + (input - posi_limit_down) * posi_one_over_ratio_knee
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
                    nega_limit_down
                    + (input - nega_limit_down) * nega_one_over_ratio_knee
                )
            else:
                output = input

            output = -output

        return output
