import torch
import torchaudio.functional as F
from . import general
from .. import analysis


def gain_computer(
    input,
    threshold=0,
    one_over_ratio=0.5,
    knee=0,
    multichannel=False,
):
    """
    Computes the gain of an audio input for compressor. 多拐点和多阈值压缩可以通过堆叠多个压缩/扩展来实现.
    input: audio (dB or amplitude)
    threshold: the value of the limit (dB or amplitude)
    one_over_ratio: the 1/ratio of the limit
    knee: the dB value of the knee, [0,+∞) (dB or amplitude)
    multichannel: True calculates gain for each channel, False calculates gain for all channels
    return: gain value (dB or amplitude)
    """
    if input.dim() == 0:
        input = input.unsqueeze(0)
    if input.dim() == 1:
        input = input.unsqueeze(0)

    if multichannel == False:
        input = general.to_mono(input)

    output = input

    # 注意接下来3个torch.where的顺序，不能乱，否则会出错
    output = torch.where(
        input <= (threshold - knee / 2),
        0,
        output,
    )

    output = torch.where(
        (input > (threshold - knee / 2)) & (knee != 0),
        (
            (one_over_ratio - 1)
            * (input - threshold + knee / 2)
            * (input - threshold + knee / 2)
        )
        / (2 * knee),
        output,
    )

    output = torch.where(
        input > (threshold + knee / 2),
        threshold + (input - threshold) * one_over_ratio - input,
        output,
    )

    return output


def timetest_signal(
    freq=1000,
    sr=96000,
    amplitude1=0.0316,
    length1=1,
    amplitude2=1,
    length2=4,
    amplitude3=0.0316,
    length3=5,
):
    """
    Generate a signal for time test
    freq: frequency of the signal
    sr: sample rate of the signal
    amplitude1: amplitude of the first stage
    length1: length of the first stage (s)
    amplitude2: amplitude of the second stage
    length2: length of the second stage (s)
    amplitude3: amplitude of the third stage
    length3: length of the third stage (s)
    return: signal
    """
    stage1 = general.generate_signal(freq, sr, amplitude1, length1)
    stage2 = general.generate_signal(freq, sr, amplitude2, length2)
    stage3 = general.generate_signal(freq, sr, amplitude3, length3)
    signal = torch.cat((stage1, stage2, stage3), 1)
    return signal


def time_extract(test, result, sr):
    result = torch.abs(result[0, :])
    test = torch.abs(test[0, :])

    result = analysis.peak.digital(result, sr, 20, 0)
    test = analysis.peak.digital(test, sr, 20, 0)

    gain = result / test

    gain[0, 0:100] = 1
    gain = torch.where(torch.isnan(gain), torch.tensor(3.1623e-05), gain)

    gain = F.amplitude_to_DB(gain, 20, 0, 0, 90)
    return gain


def ratiotest_signal(freq=1000, sr=96000):
    """
    Generate a signal for ratio test
    freq: test signal frequency
    sr: sample rate of the signal
    return: signal
    """
    stage = torch.zeros(91, sr * 5)
    signal = torch.zeros(1)
    for i in range(91):
        amp = F.DB_to_amplitude(torch.tensor(i - 90), 1, 0.5)
        stage_stage = general.generate_signal(freq, sr, amp, 5)
        stage[i, :] = stage_stage[0, :]
        signal = torch.cat((signal, stage[i, :]), 0)
    signal = signal[1:]
    signal = signal.unsqueeze(0)

    return signal


def ratio_extract(ratiotest, sr):
    """
    Extract the ratio from the ratio test signal
    ratiotest: ratio test signal
    sr: sample rate of the signal
    return: ratio
    """
    output = torch.zeros(91)
    ratiotest = ratiotest[0, :]

    # 先选取每个阶段的最后0.15s，因为前面没意义，同时为peak计算节省运算量
    ratiotest_stage = torch.zeros(1)
    for i in range(91):
        stage = ratiotest[int(i * sr * 5 + sr * 4.85) : (i + 1) * sr * 5]
        ratiotest_stage = torch.cat((ratiotest_stage, stage), 0)
    ratiotest = ratiotest_stage[1:]

    ratiotest = analysis.peak.digital(ratiotest, sr, 20, 0)
    ratiotest = ratiotest[0, :]

    for i in range(91):
        # 由于往前看了20ms，前几个值会有问题，所以删去每个阶段的前50ms
        stage = ratiotest[int(i * sr * 0.15 + sr * 0.05) : int((i + 1) * sr * 0.2)]
        stage = min(stage)
        output[i] = stage
    output = output.unsqueeze(0)
    return output
