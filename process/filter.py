import numpy as np


def smooth_filter(
    coeff,
):
    """
    calculate the coefficient of a smooth filter
    coeff: time coefficient
    return: a, b of the filter(iir)
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
