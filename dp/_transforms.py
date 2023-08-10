import torch
import torchaudio


def dB_to_amplitude(dB):
    """
    Convert decibels to amplitude.

    Args:
        dB (float): Decibels.

    Returns:
        float: Amplitude.
    """

    dB = torch.pow(10, dB / 20)

    return dB


def amplitude_to_dB(amplitude):
    """
    Convert amplitude to decibels.

    Args:
        amplitude (float): Amplitude.

    Returns:
        float: Decibels.
    """
    return torchaudio.transforms.AmplitudeToDB("magnitude")(amplitude)
