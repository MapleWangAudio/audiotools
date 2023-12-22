import numpy as np
import math
import soundfile as sf


def generate_signal(
    freq,
    sr,
    dB,
    length,
    mode=0,
):
    """
    Generate a signal with the specified frequency, sample rate, dB level, length, and mode.

    Args:
        freq (float): The frequency of the signal in Hz.
        sr (int): The sample rate of the signal in samples per second.
        dB (float): The dB level of the signal.
        length (float): The length of the signal in seconds.
        mode (int, optional): The mode of the signal generation. 0: sine wave, 1: square wave, 2: triangle wave

    Returns:
        numpy.ndarray: The generated signal as a 1D numpy array.
    """
    amplitude = dB2amp(dB)
    t = np.linspace(0, length, int(sr * length))

    if mode == 0:
        result = amplitude * np.sin(2 * np.pi * freq * t)
    if mode == 1:
        result = amplitude * np.sign(np.sin(2 * np.pi * freq * t))
    if mode == 2:
        result = amplitude * (2 / np.pi) * np.arctan(np.tan(np.pi * freq * t))

    result = result.reshape(1, -1)

    return result


def time_coeff_computer(
    time,
    sr=48000,
    range_low=0.1,
    range_high=0.9,
):
    """
    Compute the time coefficient for audio processing.

    Args:
        time (float): The time value in milliseconds.
        sr (int, optional): The sample rate of the audio. Defaults to 48000.
        range_low (float, optional): lower limit of time coefficient control range (0,1)
        range_high (float, optional): upper limit of time coefficient control range (0,1)

    Returns:
        float: The computed time coefficient.
    """
    shape_control = math.log((1 - range_low) / (1 - range_high))
    shape_control *= -1
    result = math.exp(shape_control / (time * 0.001 * sr))

    return result


def smoother(
    source,
    old,
    coeff=time_coeff_computer(1),
):
    """
    Smooths the input source value using a coefficient.

    Args:
        source: The new value to be smoothed.
        old: The previous smoothed value.
        coeff: The coefficient used for smoothing. Defaults to the result of `time_coeff_computer(1)`.

    Returns:
        The smoothed value.

    """
    return coeff * old + (1 - coeff) * source


def mono(source):
    """
    Convert multichannel audio to mono audio.

    Args:
        souce: audio amplitude.

    Returns:
        Mono audio amplitude.
    """
    channel = len(source)
    if channel > 1:
        input_all = 0
        for i in range(0, channel):
            input_all = input_all + source[i, :]
        input_mono = input_all / channel
        result = input_mono.reshape(1, -1)
    else:
        result = source

    return result


def delete(
    source,
    amp_threshold=0,
    sustain_threshold=2,
    all=False,
):
    """
    When there are consecutive sustian_threshold elements in the tensor whose absolute value is less than amp_threshold,
    delete these elements. Note that it is deleted, not assigned a value of 0

    Args:
        source: audio data
        amp_threshold: amplitude threshold,if the absolute value of the element is less than this value, it may be deleted
        sustain_threshold: sustain threshold, if there are more than this number of consecutive elements whose absolute value is less than amp_threshold, they may be deleted
        all: True: process the whole input, False: only process the beginning and the end of the input

    Returns:
        Processed source
    """
    source_copy = source
    source_mono = mono(source)
    source_mono = source_mono[0, :]

    if all == False:
        i = 0
        while i < len(source_mono):
            if abs(source_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(source_mono)):
                    if abs(source_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    source_mono = np.concatenate(
                        (source_mono[:i], source_mono[i + count :])
                    )
                    if source_copy.ndim == 1:
                        source_copy = np.concatenate(
                            (source_copy[:i], source_copy[i + count :])
                        )
                    else:
                        source_copy = np.concatenate(
                            (source_copy[:, :i], source_copy[:, i + count :]), axis=1
                        )
            source_copy = np.flip(source_copy)
            source_copy = source_copy.copy()
            source_mono = np.flip(source_mono)
            source_mono = source_mono.copy()
            break

        i = 0
        while i < len(source_mono):
            if abs(source_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(source_mono)):
                    if abs(source_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    source_mono = np.concatenate(
                        (source_mono[:i], source_mono[i + count :])
                    )
                    if source_copy.ndim == 1:
                        source_copy = np.concatenate(
                            (source_copy[:i], source_copy[i + count :])
                        )
                    else:
                        source_copy = np.concatenate(
                            (source_copy[:, :i], source_copy[:, i + count :]), axis=1
                        )
            source_copy = np.flip(source_copy)
            source_copy = source_copy.copy()
            source_mono = np.flip(source_mono)
            source_mono = source_mono.copy()
            break
    else:
        i = 0
        while i < len(source_mono):
            if abs(source_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(source_mono)):
                    if abs(source_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    source_mono = np.concatenate(
                        (source_mono[:i], source_mono[i + count :])
                    )
                    if source_copy.ndim == 1:
                        source_copy = np.concatenate(
                            (source_copy[:i], source_copy[i + count :])
                        )
                    else:
                        source_copy = np.concatenate(
                            (source_copy[:, :i], source_copy[:, i + count :]), axis=1
                        )
                    i -= 1
            i += 1

    result = source_copy

    return result


def amp2dB(source, low_cut=-90):
    """
    Convert amplitude to decibels (dB).

    Args:
        source: The source array of amplitudes.
        low_cut: The lower cutoff value in dB. Default is -90 dB.

    Returns:
        result: The result array of amplitudes in dB.
    """
    low_cut = dB2amp(low_cut)
    source = np.where(source >= low_cut, source, low_cut)
    result = 20 * np.log10(source)
    return result


def dB2amp(source):
    """
    Convert a value from decibels (dB) to amplitude.

    Args:
        source (float): The value in decibels.

    Returns:
        float: The corresponding amplitude value.
    """
    return 10 ** (source / 20)


def read(path):
    """
    Read audio file

    Args:
        path: audio file path

    Returns:
        audio amplitude, sample rate
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

    Args:
        path: audio file path
        data: audio data
        sr: sample rate
    """
    sf.write(path, data.T, sr, "PCM_24")


def mix(dry, wet, mix):
    """
    Mix two audio signals

    Args:
        dry: dry signal
        wet: wet signal
        mix: mix ratio

    Returns:
        mixed signal
    """
    if mix < 0.5:
        result = dry + wet * (4 * mix**2)
    else:
        result = dry * 4 * ((1 - mix) ** 2) + wet

    return result
