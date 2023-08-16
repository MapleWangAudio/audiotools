import torchaudio
import torch
from ..dsp import time_coefficient_computer, smooth_filter
from ..dp import dB_to_amplitude, amplitude_to_dB


def peak(input):
    """
    input : audio waveform
    return : peak value (dB)
    """
    input_max = torch.max(torch.abs(input))

    return amplitude_to_dB(input_max)


def RMS(input, mode=1, time=0, sr=44100, pre_RMS=0):
    """
    Computes the root mean square (RMS) of an audio input.
    input : audio waveform
    mode : 1 is digital RMS , 2 is analog RMS
    pre_RMS : previous RMS value (dB)
    return : RMS value (dB)
    """
    if mode == 1:
        RMS = torch.sqrt(torch.mean(torch.square(input)))
    if mode == 2:
        time_coeff = time_coefficient_computer(time, sr)
        pre_RMS = dB_to_amplitude(pre_RMS)
        RMS = smooth_filter(input**2, pre_RMS**2, time_coeff)
        RMS = torch.sqrt(RMS)

    return amplitude_to_dB(RMS)
