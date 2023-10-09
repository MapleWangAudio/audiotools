def gain_computer(
    input,
    threshold=0,
    one_over_ratio=0.5,
    knee=0,
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
