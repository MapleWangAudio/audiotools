import math


def dB_to_amplitude(db):
    """
    Convert decibels to amplitude.

    Args:
        db (float): Decibels.

    Returns:
        float: Amplitude.
    """
    return math.pow(10, db / 20)
