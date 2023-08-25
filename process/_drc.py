import torch
from ._general import to_mono


def gain_computer_0(
    input,
    threshold=0,
    one_over_ratio=0.5,
    multichannel=False,
):
    """
    Computes the gain of an audio input for compressor.
    input: audio (dB or amplitude)
    threshold: the value of the limit (dB or amplitude)
    one_over_ratio: the 1/ratio of the limit
    multichannel: True calculates gain for each channel, False calculates gain for all channels
    return: gain value (dB or amplitude)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    output = input

    # 注意接下来torch.where的顺序，不能乱，否则会出错
    output = torch.where(
        input <= threshold,
        0,
        output,
    )

    output = torch.where(
        input > threshold,
        threshold + (input - threshold) * one_over_ratio - input,
        output,
    )

    return output


def gain_computer_1(
    input,
    threshold=0,
    one_over_ratio=0.5,
    knee=0,
    multichannel=False,
):
    """
    Computes the gain of an audio input for compressor.
    input: audio (dB or amplitude)
    threshold: the value of the limit (dB or amplitude)
    one_over_ratio: the 1/ratio of the limit
    knee: the dB value of the knee, [0,+∞) (dB or amplitude)
    multichannel: True calculates gain for each channel, False calculates gain for all channels
    return: gain value (dB or amplitude)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    output = input

    # 注意接下来torch.where的顺序，不能乱，否则会出错
    output = torch.where(
        input <= (threshold - knee / 2),
        0,
        output,
    )

    output = torch.where(
        (input > (threshold - knee / 2)) & (knee != 0),
        (
            (one_over_ratio - 1)
            * (input - threshold + knee / 2)
            * (input - threshold + knee / 2)
        )
        / (2 * knee),
        output,
    )

    output = torch.where(
        input > (threshold + knee / 2),
        threshold + (input - threshold) * one_over_ratio - input,
        output,
    )

    return output


def gain_computer_2(
    input,
    threshold1=0,
    one_over_ratio1=0.5,
    knee1=0,
    threshold2=0.5,
    one_over_ratio2=0.5,
    knee2=0,
    multichannel=False,
):
    """
    Computes the gain of an audio input for compressor. t1 + k1 / 2 < t2 - k2 / 2
    input: audio (dB or amplitude)
    threshold1: the value of the limit (dB or amplitude)
    one_over_ratio1: the 1/ratio of the limit
    knee1: the dB value of the knee, [0,+∞) (dB or amplitude)
    threshold2: the value of the limit (dB or amplitude)
    one_over_ratio2: the 1/ratio of the limit
    knee2: the dB value of the knee, [0,+∞) (dB or amplitude)
    multichannel: True calculates gain for each channel, False calculates gain for all channels
    return: gain value (dB or amplitude)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    output = input

    # 注意接下来3个torch.where的顺序，不能乱，否则会出错

    output = torch.where(
        input <= (threshold1 - knee1 / 2),
        0,
        output,
    )

    output = torch.where(
        (input > (threshold1 - knee1 / 2)) & (knee1 != 0),
        (
            (one_over_ratio1 - 1)
            * (input - threshold1 + knee1 / 2)
            * (input - threshold1 + knee1 / 2)
        )
        / (2 * knee1),
        output,
    )

    output = torch.where(
        input > (threshold1 + knee1 / 2),
        threshold1 + (input - threshold1) * one_over_ratio1 - input,
        output,
    )

    output = torch.where(
        (input > (threshold2 - knee2 / 2)) & (knee2 != 0),
        (
            (one_over_ratio2 - 1)
            * (input - threshold2 + knee2 / 2)
            * (input - threshold2 + knee2 / 2)
        )
        / (2 * knee2),
        output,
    )

    output = torch.where(
        input > (threshold2 + knee2 / 2),
        threshold2 + (input - threshold2) * one_over_ratio2 - input,
        output,
    )

    return output
