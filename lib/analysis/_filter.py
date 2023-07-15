import matplotlib.pyplot as plt
import torch
import numpy as np
from scipy import signal


def fvtool(b, a=1, sample_rate=44100, save=True, show=False, name="fvtool", end=False):
    """
    Emulates the functionality of MATLAB's fvtool function.
    """
    b = b.numpy()
    if isinstance(a, torch.Tensor):
        a = a.numpy()
        worN = 8192
    else:
        worN = len(b)

    w, h = signal.freqz(b, a, worN=worN, fs=sample_rate)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(w, 20 * np.log10(abs(h)))
    ax1.set_xscale("log")
    ax1.set_ylabel("Magnitude [dB]")
    ax1.grid()

    ax2.plot(w, np.unwrap(np.angle(h)))
    ax2.set_xscale("log")
    ax2.set_xlabel("Frequency [hz]")
    ax2.set_ylabel("Phase [radians]")
    ax2.grid()

    if save:
        name = name + ".svg"
        plt.savefig(name, format="svg")
    if show:
        plt.show(block=end)
