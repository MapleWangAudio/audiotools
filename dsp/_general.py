import torch
import torchaudio
import numpy as np


def time_coefficient_computer(
    time, sample_rate, lower_limit=torch.tensor(0.1), upper_limit=torch.tensor(0.9)
):
    """
    time: ms
    sample_rate: Hz
    """
    coeff = torch.log(upper_limit / lower_limit)
    coeff *= -1
    return torch.exp(coeff / (time * sample_rate * 0.001))


def smooth_filter(input, time_coeff, order=1):
    """
    input: waveform
    time_coeff: time coefficient
    order: 1 or 2
    """
    if order == 1:
        output = torch.zeros(len(input))
        for i in range(len(input)):
            output[i] = time_coeff * output[i - 1] + (1 - time_coeff) * input[i]

    if order == 2:
        b = torch.zeros(3)
        a = torch.zeros(3)
        b[0] = (1 - time_coeff) * (1 - time_coeff)
        b[1] = 0
        b[2] = 0
        a[0] = 1
        a[1] = -2 * time_coeff
        a[2] = time_coeff**2

        output = torchaudio.functional.biquad(input, b[0], b[1], b[2], a[0], a[1], a[2])

    return output


def to_mono(data):
    """
    Convert stereo audio to mono audio
    """

    if data.ndim != 1:
        data_all = torch.zeros(len(data[0, :]))
        for i in range(0, len(data)):
            data_all = data_all + data[i, :]
        data_mono = data_all / len(data)
    else:
        data_mono = data

    output = data_mono.unsqueeze(0)

    return output


def delete(data, amp_threshold, sustain_threshold, all=False):
    """
    When there are consecutive sustian_threshold elements in the tensor whose absolute value is less than amp_threshold,
    delete these elements. Note that it is deleted, not assigned a value of 0
    only process the beginning and the end of the data
    all: if True, process the whole data, not only the beginning and the end
    """
    data_copy = data.numpy()
    data_mono = to_mono(data)
    data_mono = data_mono[0, :]
    data_mono = data_mono.numpy()

    if all == False:
        i = 0
        while i < len(data_mono):
            if abs(data_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(data_mono)):
                    if abs(data_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    data_mono = np.concatenate((data_mono[:i], data_mono[i + count :]))
                    if data_copy.ndim == 1:
                        data_copy = np.concatenate(
                            (data_copy[:i], data_copy[i + count :])
                        )
                    else:
                        data_copy = np.concatenate(
                            (data_copy[:, :i], data_copy[:, i + count :]), axis=1
                        )
            data_copy = np.flip(data_copy)
            data_copy = data_copy.copy()
            data_mono = np.flip(data_mono)
            data_mono = data_mono.copy()
            break

        i = 0
        while i < len(data_mono):
            if abs(data_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(data_mono)):
                    if abs(data_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    data_mono = np.concatenate((data_mono[:i], data_mono[i + count :]))
                    if data_copy.ndim == 1:
                        data_copy = np.concatenate(
                            (data_copy[:i], data_copy[i + count :])
                        )
                    else:
                        data_copy = np.concatenate(
                            (data_copy[:, :i], data_copy[:, i + count :]), axis=1
                        )
            data_copy = np.flip(data_copy)
            data_copy = data_copy.copy()
            data_mono = np.flip(data_mono)
            data_mono = data_mono.copy()
            break
    else:
        i = 0
        while i < len(data_mono):
            if abs(data_mono[i]) <= amp_threshold:
                count = 0
                for j in range(i, len(data_mono)):
                    if abs(data_mono[j]) <= amp_threshold:
                        count += 1
                    else:
                        break
                if count >= sustain_threshold:
                    data_mono = np.concatenate((data_mono[:i], data_mono[i + count :]))
                    if data_copy.ndim == 1:
                        data_copy = np.concatenate(
                            (data_copy[:i], data_copy[i + count :])
                        )
                    else:
                        data_copy = np.concatenate(
                            (data_copy[:, :i], data_copy[:, i + count :]), axis=1
                        )
                    i -= 1
            i += 1

    return torch.tensor(data_copy)
