import numpy as np
import torch


def gate(data, amp_threshold, sustain_threshold, all=False):
    """
    When there are consecutive sustian_threshold elements in the tensor whose absolute value is less than amp_threshold,
    delete these elements. Note that it is deleted, not assigned a value of 0
    only process the beginning and the end of the data
    all: if True, process the whole data, not only the beginning and the end
    """
    data_copy = data.numpy()
    data_mono = to_mono(data)
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


def to_mono(data):
    """
    Convert stereo audio to mono audio
    """
    data = data.numpy()

    if data.ndim != 1:
        data_all = np.zeros(len(data[0, :]))
        for i in range(0, len(data)):
            data_all = data_all + data[i, :]
        data_mono = data_all / len(data)
    else:
        data_mono = data

    return torch.tensor(data_mono)
