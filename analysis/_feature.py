import torchaudio
import torch
from ..process import time_coefficient_computer, smooth_filter, to_mono
import math


def peak(
    input,
    sr=48000,
    time=1,
    multichannel=False,
):
    """
    Computes the peak value of an audio input.
    input : audio amplitude
    time : peak window (ms)
    sr : sample rate (Hz)
    multichannel : True calculates peak value for each channel, False calculates peak value for all channels
    return : peak value (dB)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    time = time * 0.001
    channel, input_length = input.shape
    buffer_length = int(time * sr)
    buffer_num = int(math.ceil(input_length / buffer_length))
    peak = torch.zeros_like(input)

    # Pad input with zeros to the nearest multiple of time*sr
    if (input_length % buffer_length) != 0:
        input = torch.nn.functional.pad(
            input,
            (
                0,
                int(buffer_num * buffer_length - input_length),
            ),
            "constant",
            0,
        )

    for i in range(buffer_num):
        start = i * buffer_length
        end = (i + 1) * buffer_length
        for j in range(channel):
            peak[j, start:end] = torch.max(torch.abs(input[j, start:end]))

    peak = peak[:, 0:input_length]
    peak = torchaudio.functional.amplitude_to_DB(peak, 20, 0, 0)

    return peak


def RMS(
    input,
    sr=48000,
    time=0,
    mode=1,
    multichannel=False,
):
    """
    Computes the root mean square (RMS) of an audio input.
    input : audio amplitude
    sr : sample rate (Hz)
    time : RMS window (ms)
    multichannel : True calculates RMS value for each channel, False calculates RMS value for all channels
    mode : 0 is digital RMS , 1 is analog RMS
    return : RMS value (dB)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    time = time * 0.001
    channel, input_length = input.shape
    buffer_length = int(time * sr)
    padding_length = int(math.ceil(buffer_length / 2))
    RMS = torch.zeros_like(input)

    # Pad input with zeros to the nearest multiple of time*sr
    if (input_length % buffer_length) != 0:
        input = torch.nn.functional.pad(
            input,
            (
                padding_length,
                padding_length,
            ),
            "constant",
            0,
        )

    if mode == 0:
        for i in range(channel):
            for j in range(input_length):
                RMS[i, j] = torch.sqrt(
                    torch.mean(torch.square(input[i, j : j + buffer_length]))
                )

    if mode == 1:
        time_coeff = time_coefficient_computer(time, sr)
        RMS = smooth_filter(input**2, time_coeff, 1)
        RMS = torch.sqrt(RMS)

    RMS = torchaudio.functional.amplitude_to_DB(RMS, 20, 0, 0)

    return RMS
