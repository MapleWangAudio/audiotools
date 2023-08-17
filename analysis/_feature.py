import torchaudio
import torch
from ..dsp import time_coefficient_computer, smooth_filter, to_mono
import math


def peak(input, sr=48000, time=1, multichannel=False):
    """
    input : audio waveform
    time : find max time (ms)
    sr : sample rate (Hz)
    multichannel : True calculates peak value for each channel, False calculates peak value for all channels
    return : peak value (dB)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    time = time * 0.001
    input_length = int(len(input[0, :]))
    buffer_length = int(time * sr)
    buffer_num = int(math.ceil(input_length / buffer_length))
    channel = int(len(input))
    peak = torch.zeros(channel, input_length)

    if (input_length % buffer_length) != 0:
        # Pad input with zeros to the nearest multiple of time*sr
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


def RMS(input, sr=48000, time=0, multichannel=False, mode=1):
    """
    Computes the root mean square (RMS) of an audio input.
    input : audio waveform
    mode : 1 is digital RMS , 2 is analog RMS
    pre_RMS : previous RMS value (dB)
    return : RMS value (dB)
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = to_mono(input)

    time = time * 0.001
    input_length = int(len(input[0, :]))
    buffer_length = int(time * sr)
    buffer_num = int(math.ceil(input_length / buffer_length))
    channel = int(len(input))
    RMS = torch.zeros(channel, input_length)

    if (input_length % buffer_length) != 0:
        # Pad input with zeros to the nearest multiple of time*sr
        input = torch.nn.functional.pad(
            input,
            (
                0,
                int(buffer_num * buffer_length - input_length),
            ),
            "constant",
            0,
        )

    if mode == 1:
        for i in range(buffer_num):
            start = i * buffer_length
            end = (i + 1) * buffer_length
            for j in range(channel):
                RMS[j, start:end] = torch.sqrt(
                    torch.mean(torch.square(input[j, start:end]))
                )

    if mode == 2:
        time_coeff = time_coefficient_computer(time, sr)
        pre_RMS = torchaudio.functional.DB_to_amplitude(pre_RMS, ref=1.0, power=0.5)
        RMS = smooth_filter(input**2, pre_RMS**2, time_coeff)
        RMS = torch.sqrt(RMS)

    RMS = torchaudio.functional.amplitude_to_DB(RMS, 20, 0, 0)

    return RMS
