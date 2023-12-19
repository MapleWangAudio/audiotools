import numpy as np


def gain_computer(
    input,
    threshold=0.0,
    one_over_ratio=0.5,
    knee=0.0,
):
    """
    Computes the gain of an audio input for compressor.
    input: audio (dB or amplitude)
    threshold: the value of the limit (dB or amplitude)
    one_over_ratio: the 1/ratio of the limit
    knee: the value of the knee, [0,+∞) (dB or amplitude)
    return: gain value (dB or amplitude)
    """
    if input <= (threshold - knee / 2):
        output = 0
    elif (input <= (threshold + knee / 2)) & (knee != 0):
        output = (
            (one_over_ratio - 1)
            * (input - threshold + knee / 2)
            * (input - threshold + knee / 2)
        ) / (2 * knee)
    else:
        output = threshold + (input - threshold) * one_over_ratio - input

    return output


def gain_computer_array(
    input,
    threshold=0.0,
    one_over_ratio=0.5,
    knee=0.0,
):
    """
    Computes the gain of an audio input for compressor. 多拐点和多阈值压缩可以通过堆叠多个压缩/扩展来实现.
    input: audio (dB or amplitude)
    threshold: the value of the limit (dB or amplitude)
    one_over_ratio: the 1/ratio of the limit
    knee: the dB value of the knee, [0,+∞) (dB or amplitude)
    multichannel: True calculates gain for each channel, False calculates gain for all channels
    return: gain value (dB or amplitude)
    """

    output = input

    # 注意接下来3个torch.where的顺序，不能乱，否则会出错
    output = np.where(
        input <= (threshold - knee / 2),
        0,
        output,
    )

    output = np.where(
        (input > (threshold - knee / 2)) & (knee != 0),
        (
            (one_over_ratio - 1)
            * (input - threshold + knee / 2)
            * (input - threshold + knee / 2)
        )
        / (2 * knee),
        output,
    )

    output = np.where(
        input > (threshold + knee / 2),
        threshold + (input - threshold) * one_over_ratio - input,
        output,
    )

    return output
