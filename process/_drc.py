import torch
from ._general import to_mono


def gain_computer(
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

    # 注意接下来3个torch.where的顺序，不能乱，否则会出错
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
