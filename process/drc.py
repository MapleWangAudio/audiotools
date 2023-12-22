import numpy as np


def gain_computer(
    input,
    th=0.0,
    one_over_ratio=0.5,
    knee=0.0,
):
    """
    Computes the gain reduction based on the input signal and the specified parameters.

    Args:
        input (float): The input signal value.
        th (float, optional): The threshold value. Defaults to 0.0.
        one_over_ratio (float, optional): The inverse of the compression ratio. Defaults to 0.5.
        knee (float, optional): The width of the knee region. Defaults to 0.0.

    Returns:
        float: The computed gain reduction value.
    """
    if input <= (th - knee / 2):
        output = 0
    elif (input <= (th + knee / 2)) & (knee != 0):
        output = (
            (one_over_ratio - 1) * (input - th + knee / 2) * (input - th + knee / 2)
        ) / (2 * knee)
    else:
        output = th + (input - th) * one_over_ratio - input

    return output


def gain_computer_array(
    input,
    th=0.0,
    one_over_ratio=0.5,
    knee=0.0,
):
    """
    Computes the gain computer array for dynamic range compression.

    Args:
        input (numpy.ndarray): The input array.
        th (float, optional): The threshold value. Defaults to 0.0.
        one_over_ratio (float, optional): The inverse of the compression ratio. Defaults to 0.5.
        knee (float, optional): The knee width. Defaults to 0.0.

    Returns:
        numpy.ndarray: The output array after applying dynamic range compression.
    """
    output = input

    output = np.where(
        input <= (th - knee / 2),
        0,
        output,
    )

    output = np.where(
        (input > (th - knee / 2)) & (knee != 0),
        ((one_over_ratio - 1) * (input - th + knee / 2) * (input - th + knee / 2))
        / (2 * knee),
        output,
    )

    output = np.where(
        input > (th + knee / 2),
        th + (input - th) * one_over_ratio - input,
        output,
    )

    return output
