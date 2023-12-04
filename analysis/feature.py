from .. import process
import numpy as np
import math
from . import plot


def envelope_peak(
    source,
    sr,
    smooth_length=2,
    buffer_size=16,
    low_cut=None,
    save_path=None,
):
    """
    Computes the peak envelope of an audio input.
    source: audio amplitude
    sr: sample rate of the signal (hz)
    smooth_length: smooth length (ms)
    buffer_size: buffer size
    low_cut: low cut dB
    save_path: save path
    """
    smooth_length = int(smooth_length * 0.001 * sr)
    length = len(source[0, :])

    step = math.ceil(length / buffer_size)
    post_pad_length = step * buffer_size - length

    stage = np.zeros([buffer_size, smooth_length])
    result = np.zeros(step * buffer_size)

    source = np.pad(
        source,
        ((0, 0), (smooth_length - 1, post_pad_length)),
        "constant",
        constant_values=0,
    )

    for i in range(0, step * buffer_size, buffer_size):
        for j in range(buffer_size):
            stage[j] = source[0, i + j : i + j + smooth_length]
        result[i : i + buffer_size] = np.max(stage, 1)

    result = result[:length]
    result = result.reshape(1, -1)

    if isinstance(low_cut, int):
        result = process.amp2dB(result, low_cut)

    if isinstance(save_path, str):
        plot.waveform(result, sr, name=save_path)
        return result
    else:
        return result


def envelope_RMS(
    source, sr, smooth_length=2, buffer_size=16, low_cut=None, save_path=None
):
    """
    Computes the RMS envelope of an audio input.
    source: audio amplitude
    sr: sample rate of the signal (hz)
    smooth_length: smooth length (ms)
    buffer_size: buffer size
    low_cut: low cut dB
    save_path: save path
    """
    smooth_length = int(smooth_length * 0.001 * sr)
    length = len(source[0, :])

    step = math.ceil(length / buffer_size)
    post_pad_length = step * buffer_size - length

    stage = np.zeros([buffer_size, smooth_length])
    result = np.zeros(step * buffer_size)

    source = np.pad(
        source,
        ((0, 0), (smooth_length - 1, post_pad_length)),
        "constant",
        constant_values=0,
    )

    for i in range(0, step * buffer_size, buffer_size):
        for j in range(buffer_size):
            stage[j] = source[0, i + j : i + j + smooth_length]
        result[i : i + buffer_size] = np.linalg.norm(stage, 2, 1)

    result = result[:length]
    result = result / (smooth_length**0.5)
    result = result.reshape(1, -1)

    if isinstance(low_cut, int):
        result = process.amp2dB(result, low_cut)

    if isinstance(save_path, str):
        plot.waveform(result, sr, name=save_path)
        return result
    else:
        return result


def peak_digital(input):
    """
    Computes the peak value of an audio input. Digital Type.
    input: audio amplitude
    return: peak values
    """
    return np.max(input)


def peak_analog_prue(
    input,
    output_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the peak value of an audio input. Analog Prue Type.
    input: audio amplitude
    output_old: peak value of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: peak value
    """
    return release_coeff * output_old + (1 - attack_coeff) * max(
        (input - output_old), 0
    )


def peak_analog_level_corrected0(
    input,
    state_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the peak value of an audio input in decoupled style. Analog Level Correted Type.
    input: audio amplitude
    state_old: peak state of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: peak, peak state
    """
    state = max(input, release_coeff * state_old)
    output = process.smoother(state, state_old, attack_coeff)

    return output, state


def peak_analog_level_corrected1(
    input,
    output_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the peak value of an audio input in branch style. Analog Level Correted Type.
    input: audio amplitude
    output_old: peak value of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: peak
    """
    if input > output_old:
        output = process.smoother(input, output_old, attack_coeff)
    else:
        output = release_coeff * output_old

    return output


def peak_analog_smooth0(
    input,
    state_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the peak value of an audio input in decoupled style. Analog Smooth Type.
    input: audio amplitude
    state_old: peak state of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: peak, peak state
    """
    state = max(
        input,
        process.smoother(input, state_old, release_coeff),
    )
    output = process.smoother(state, state_old, attack_coeff)

    return output, state


def peak_analog_smooth1(
    input,
    output_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the peak value of an audio input in branch style. Analog Smooth Type.
    input: audio amplitude
    output_old: peak value of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: peak
    """
    if input > output_old:
        output = process.smoother(input, output_old, attack_coeff)
    else:
        output = process.smoother(input, output_old, release_coeff)

    return output


def RMS_digital(input):
    """
    Computes the root mean square (RMS) of an audio input. Digital Type
    input: audio amplitude
    return: RMS
    """
    return np.sqrt(np.mean(np.square(input)))


def RMS_analog_prue(
    input,
    output_old,
    coeff,
):
    """
    Computes the root mean square (RMS) of an audio input. Analog Prue Type
    input: audio amplitude
    output_old: RMS value of the last sample
    return: RMS
    """
    output = process.smoother(input**2, output_old**2, coeff)
    output = output**0.5

    return output


def RMS_analog_level_corrected0(
    input,
    state_old,
    output_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the root mean square (RMS) of an audio input in decoupled style. Analog Level Corrected Type
    input: audio amplitude
    state_old: RMS state of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: RMS
    """
    state = (input**2 + (release_coeff * state_old) ** 2) / 2
    state = state**0.5

    output = process.smoother(state, output_old, attack_coeff)

    return output, state


def RMS_analog_level_corrected1(
    input,
    output_old,
    attack_coeff=1,
    release_coeff=1,
):
    """
    Computes the root mean square (RMS) of an audio input in branch style. Analog Level Corrected Type
    input: audio amplitude
    output_old: RMS value of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: RMS
    """
    if input > output_old:
        output = process.smoother(input**2, output_old**2, attack_coeff)
    else:
        output = release_coeff * output_old**2

    output = output**0.5

    return output


def RMS_analog_smooth0(
    input,
    state_old,
    output_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the RMS value of an audio input in decoupled style. Analog Smooth Type.
    input: audio amplitude
    state_old: RMS state of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: RMS, RMS state
    """
    state = (input**2 + (process.smoother(input, state_old, release_coeff)) ** 2) / 2
    state = state**0.5

    output = process.smoother(state, output_old, attack_coeff)

    return output, state


def RMS_analog_smooth1(
    input,
    output_old,
    attack_coeff,
    release_coeff,
):
    """
    Computes the RMS value of an audio input in branch style. Analog Smooth Type.
    input: audio amplitude
    output_old: RMS value of the last sample
    attack_coeff: attack coefficient
    release_coeff: release coefficient
    return: RMS
    """
    if input > output_old:
        output = process.smoother(input**2, output_old**2, attack_coeff)
    else:
        output = process.smoother(input**2, output_old**2, release_coeff)

    output = output**0.5

    return output


def drc_time_test_signal(
    freq=1000,
    sr=48000,
    dB1=-45.0,
    length1=1.0,
    dB2=0.0,
    length2=4.0,
    dB3=-45.0,
    length3=5.0,
):
    """
    Generate a signal for time test
    freq: frequency of the signal (hz)
    sr: sample rate of the signal (hz)
    amplitude1: amplitude of the first stage
    length1: length of the first stage (s)
    amplitude2: amplitude of the second stage
    length2: length of the second stage (s)
    amplitude3: amplitude of the third stage
    length3: length of the third stage (s)
    return: signal
    """
    stage1 = process.generate_signal(freq, sr, dB1, length1)
    stage2 = process.generate_signal(freq, sr, dB2, length2)
    stage3 = process.generate_signal(freq, sr, dB3, length3)
    signal = np.hstack((stage1, stage2, stage3))
    return signal


def drc_time_extract(test, result, sr=96000):
    """
    Extract the time from the time test signal
    test: time test signal raw
    result: time test signal processed
    sr: sample rate of the signal (hz)
    return: time featrue
    """
    result = np.abs(result[0, :])
    test = np.abs(test[0, :])

    result_peak = np.zeros_like(result)
    test_peak = np.zeros_like(test)
    for i in range(int(0.02 * sr)):
        result_peak[0, i] = peak_digital(result[0, 0 : i + 1])
        test_peak[0, i] = peak_digital(test[0, 0 : i + 1])
    for i in range(int(0.02 * sr), len(result)):
        result_peak[0, i] = peak_digital(result[0, int(i - 0.02 * sr) : i])
        test_peak[0, i] = peak_digital(test[0, int(i - 0.02 * sr) : i])

    gain = result / test

    gain[0, 0:100] = 1
    gain = np.where(np.isnan(gain), 3.1623e-05, gain)

    gain = process.amp2dB(gain)
    return gain


def drc_ratio_test_signal(freq=1000, sr=96000, start=-60):
    """
    Generate a signal for ratio test [start,0]
    freq: test signal frequency
    sr: sample rate of the signal (hz)
    return: signal
    """
    num = -start + 1
    stage = np.zeros([num, sr * 5])
    signal = np.zeros(1)
    for i in range(num):
        dB = i + start
        stage_stage = process.generate_signal(freq, sr, dB, 5)
        stage[i] = stage_stage[0]
        signal = np.concatenate((signal, stage[i]), 0)
    signal = signal[1:]
    signal = signal.reshape(1, -1)

    return signal


def drc_ratio_extract(ratiotest, sr, num=61):
    """
    Extract the ratio from the ratio test signal
    ratiotest: ratio test signal processed
    sr: sample rate of the signal (hz)
    return: ratio feature
    """
    output = np.zeros(num)
    ratiotest = ratiotest[0]

    # 先选取每个阶段的最后1s，因为前面没意义，同时为peak计算节省运算量
    ratiotest_temp = np.zeros(1)
    for i in range(num):
        temp = ratiotest[int(i * sr * 5 + sr * 4) : int((i + 1) * sr * 5)]
        ratiotest_temp = np.concatenate((ratiotest_temp, temp), 0)
    ratiotest = ratiotest_temp[1:]
    ratiotest = ratiotest.reshape(1, -1)

    ratiotest_peak = envelope_peak(ratiotest, sr, 20, 512)

    for i in range(num):
        # 由于往前看了20ms，前几个值会有问题，所以删去每个阶段的前100ms
        temp = ratiotest_peak[0, int(i * sr + sr * 0.1) : int((i + 1) * sr)]
        temp = min(temp)
        output[i] = temp

    output = output.reshape(1, -1)

    return output
