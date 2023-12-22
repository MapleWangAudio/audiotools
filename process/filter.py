import numpy as np


def smooth_filter(
    coeff,
):
    """
    Apply a smooth filter to the input signal.

    Args:
        coeff (float): The time coefficient used for filtering.

    Returns:
        tuple: A tuple containing the numerator and denominator coefficients of the filter.
    """
    b = np.zeros(3)
    a = np.zeros(3)
    b[0] = (1 - coeff) * (1 - coeff)
    b[1] = 0
    b[2] = 0
    a[0] = 1
    a[1] = -2 * coeff
    a[2] = coeff**2

    return a, b
