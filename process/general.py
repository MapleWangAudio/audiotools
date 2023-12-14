import numpy as np
import math
import soundfile as sf


# todo：生成噪音信号
def generate_signal(
    freq,
    sr,
    dB,
    length,
    mode=0,
):
    """
    Generate a signal
    freq: frequency (Hz)
    sr: sample rate (Hz)
    amplitude: amplitude
    length: length (s)
    mode: 0: sine wave, 1: square wave, 2: triangle wave
    return: signal
    """
    amplitude = dB2amp(dB)
    t = np.linspace(0, length, int(sr * length))

    if mode == 0:
        output = amplitude * np.sin(2 * np.pi * freq * t)
    if mode == 1:
        output = amplitude * np.sign(np.sin(2 * np.pi * freq * t))
    if mode == 2:
        output = amplitude * (2 / np.pi) * np.arctan(np.tan(np.pi * freq * t))

    output = output.reshape(1, -1)

    return output


def time_coeff_computer(
    time,
    sr=48000,
    range_low=0.1,
    range_high=0.9,
):
    """
    Compute time coefficient for smooth filter
    time: smooth time (ms)
    sr: sample rate (Hz)
    range_low: lower limit of time coefficient control range (0,1)
    range_high: upper limit of time coefficient control range (0,1)
    return: time coefficient
    """
    shape_control = math.log((1 - range_low) / (1 - range_high))
    shape_control *= -1
    output = math.exp(shape_control / (time * 0.001 * sr))

    return output


def smoother(
    input,
    old,
    coeff=time_coeff_computer(1),
):
    """
    Filter to smooth something
    input: audio amplitude
    coeff: time coefficient
    return: smoothed input
    """
    return coeff * old + (1 - coeff) * input


def mono(input):
    """
    Convert multichannel audio to mono audio
    input: audio amplitude
    return: mono audio amplitude
    """
    channel = len(input)
    if channel > 1:
        input_all = 0
        for i in range(0, channel):
            input_all = input_all + input[i, :]
        input_mono = input_all / channel
        output = input_mono.reshape(1, -1)
    else:
        output = input

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
    input_copy = input
    input_mono = mono(input)
    input_mono = input_mono[0, :]

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

    output = input_copy

    return output


def amp2dB(input, low_cut=-90):
    """
    Convert amplitude to dB
    input: audio amplitude
    return: audio dB [-90,+∞)
    """
    low_cut = dB2amp(low_cut)
    input = np.where(input >= low_cut, input, low_cut)
    output = 20 * np.log10(input)
    return output


def dB2amp(input):
    """
    Convert dB to amplitude
    input: audio dB
    return: audio amplitude
    """
    return 10 ** (input / 20)


def read(path):
    """
    Read audio file
    path: audio file path
    return: audio amplitude, sample rate
    """
    data, sr = sf.read(path)
    data = data.T
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = data.astype(np.float64)
    return data, sr


def write(data, sr, path):
    """
    Write audio file, bit depth: int 24
    path: audio file path
    data: audio data
    sr: sample rate
    """
    sf.write(path, data.T, sr, "PCM_24")


def mix(dry, wet, mix):
    """
    Mix two audio signals
    dry: dry signal
    wet: wet signal
    mix: mix ratio
    return: mixed signal
    """
    if mix < 0.5:
        output = dry + wet * (4 * mix**2)
    else:
        output = dry * 4 * ((1 - mix) ** 2) + wet

    return output
