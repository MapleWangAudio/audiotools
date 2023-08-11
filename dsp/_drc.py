import torch


def gain_computer(
    input,
    threshold=0,
    one_over_ratio=0,
    knee=0,
):
    """
    mono in mono out
    input: dB
    threshold: the dB value of the limit
    ratio: the 1/ratio of the limit
    knee: the dB value of the knee,0-正无穷dB
    """
    if input > (threshold + knee / 2):
        output = threshold + (input - threshold) * one_over_ratio
        output = output - input
    elif input > (threshold - knee / 2) and (knee != 0):
        output = input + (
            (one_over_ratio - 1)
            * (input - threshold + knee / 2)
            * (input - threshold + knee / 2)
        ) / (2 * knee)
        output = output - input
    else:
        output = 0

    return output


def time_coefficient(
    time, sample_rate, lower_limit=torch.tensor(0.1), upper_limit=torch.tensor(0.9)
):
    """
    time: ms
    sample_rate: Hz
    """
    coeff = torch.log(upper_limit / lower_limit)
    coeff *= -1
    return torch.exp(coeff / (time * sample_rate * 0.001))
