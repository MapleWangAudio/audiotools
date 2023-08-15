import torch


def time_coefficient_computer(
    time, sample_rate, lower_limit=torch.tensor(0.1), upper_limit=torch.tensor(0.9)
):
    """
    time: ms
    sample_rate: Hz
    """
    coeff = torch.log(upper_limit / lower_limit)
    coeff *= -1
    return torch.exp(coeff / (time * sample_rate * 0.001))


def smooth_filter(input, pre_input, time_coeff, order=1):
    if order == 1:
        output = time_coeff * pre_input + (1 - time_coeff) * input
    # todo: "order == 2"
    return output
