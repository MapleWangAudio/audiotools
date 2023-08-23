import torch
import torchaudio
import numpy as np
import math


def time_coefficient_computer(
    time,
    sample_rate=48000,
    coeff=math.log(9),
):
    """
    Compute time coefficient for smooth filter
    time: smooth time (ms)
    sample_rate: sample rate (Hz)
    coeff: control coefficient
    return: time coefficient
    """
    coeff = torch.tensor(coeff)
    coeff *= -1
    return torch.exp(coeff / (time * 0.001 * sample_rate))


def smooth_filter(
    input,
    coeff=time_coefficient_computer(1),
    order=1,
):
    """
    Filter to smooth something
    input: audio amplitude
    time_coeff: time coefficient
    order: 1 or 2
    return: smoothed input
    """
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if order == 1:
        channel, length = input.shape
        output = torch.zeros_like(input)
        for i in range(channel):
            for j in range(1, length):
                output[i, j] = coeff * output[i, j - 1] + (1 - coeff) * input[i, j]

    if order == 2:
        b = torch.zeros(3)
        a = torch.zeros(3)
        b[0] = (1 - coeff) * (1 - coeff)
        b[1] = 0
        b[2] = 0
        a[0] = 1
        a[1] = -2 * coeff
        a[2] = coeff**2

        output = torchaudio.functional.biquad(input, b[0], b[1], b[2], a[0], a[1], a[2])

    return output


def to_mono(input):
    """
    Convert multichannel audio to mono audio
    input: audio amplitude
    return: mono audio amplitude
    """
    if input.ndim != 1:
        input_all = torch.zeros(len(input[0, :]))
        for i in range(0, len(input)):
            input_all = input_all + input[i, :]
        input_mono = input_all / len(input)
    else:
        input_mono = input

    output = input_mono.unsqueeze(0)

    return output


def delete(
    input,
    amp_threshold=0,
    sustain_threshold=2,
    all=False,
):
    """
    When there are consecutive sustian_threshold elements in the tensor whose absolute value is less than amp_threshold,
    delete these elements. Note that it is deleted, not assigned a value of 0
    input: audio data
    amp_threshold: amplitude threshold,if the absolute value of the element is less than this value, it may be deleted
    sustain_threshold: sustain threshold, if there are more than this number of consecutive elements whose absolute value is less than amp_threshold, they may be deleted
    all: True: process the whole input, False: only process the beginning and the end of the input
    return: processed input
    """
    input_copy = input.numpy()
    input_mono = to_mono(input)
    input_mono = input_mono[0, :]
    input_mono = input_mono.numpy()

    if all == False:
        i = 0
        while i < len(input_mono):
            if abs(input_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(input_mono)):
                    if abs(input_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    input_mono = np.concatenate(
                        (input_mono[:i], input_mono[i + count :])
                    )
                    if input_copy.ndim == 1:
                        input_copy = np.concatenate(
                            (input_copy[:i], input_copy[i + count :])
                        )
                    else:
                        input_copy = np.concatenate(
                            (input_copy[:, :i], input_copy[:, i + count :]), axis=1
                        )
            input_copy = np.flip(input_copy)
            input_copy = input_copy.copy()
            input_mono = np.flip(input_mono)
            input_mono = input_mono.copy()
            break

        i = 0
        while i < len(input_mono):
            if abs(input_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(input_mono)):
                    if abs(input_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    input_mono = np.concatenate(
                        (input_mono[:i], input_mono[i + count :])
                    )
                    if input_copy.ndim == 1:
                        input_copy = np.concatenate(
                            (input_copy[:i], input_copy[i + count :])
                        )
                    else:
                        input_copy = np.concatenate(
                            (input_copy[:, :i], input_copy[:, i + count :]), axis=1
                        )
            input_copy = np.flip(input_copy)
            input_copy = input_copy.copy()
            input_mono = np.flip(input_mono)
            input_mono = input_mono.copy()
            break
    else:
        i = 0
        while i < len(input_mono):
            if abs(input_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(input_mono)):
                    if abs(input_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    input_mono = np.concatenate(
                        (input_mono[:i], input_mono[i + count :])
                    )
                    if input_copy.ndim == 1:
                        input_copy = np.concatenate(
                            (input_copy[:i], input_copy[i + count :])
                        )
                    else:
                        input_copy = np.concatenate(
                            (input_copy[:, :i], input_copy[:, i + count :]), axis=1
                        )
                    i -= 1
            i += 1

    return torch.tensor(input_copy)
