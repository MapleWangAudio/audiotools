import torchaudio
import torch
from ..dsp import time_coefficient_computer, smooth_filter


def peak(waveform):
    waveform_max = torch.max(torch.abs(waveform))
    return torchaudio.transforms.AmplitudeToDB("magnitude")(waveform_max)


def RMS(waveform, mode=1, time=0, sr=44100, pre_input=0):
    """
    Computes the root mean square (RMS) of an audio waveform.
    """
    if mode == 1:
        RMS = torch.sqrt(torch.mean(torch.square(waveform)))
    if mode == 2:
        time_coeff = time_coefficient_computer(time, sr)
        RMS = smooth_filter(waveform, pre_input, time_coeff)
        RMS = torch.sqrt(RMS)
    return torchaudio.transforms.AmplitudeToDB("magnitude")(RMS)
