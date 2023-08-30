import matplotlib.pyplot as plt
import torchaudio
import torch
import numpy as np
from scipy import signal


def waveform(
    input,
    sample_rate=48000,
    show=False,
    end=False,
    dB=False,
    name="waveform",
    save=True,
):
    """
    Plots the waveform of an audio data.
    input: audio amplitude
    sample_rate: sample rate (Hz)
    dB: True plots the waveform in dB, False plots the waveform in amplitude
    save: True saves the plot as a .svg file, False does not save the plot
    show: True shows the plot, False does not show the plot
    name: name of the saved file
    end: the last plot must be set to True, otherwise the plot will not be displayed
    """
    if input.dim() == 0:
        input = input.unsqueeze(0)
    if input.dim() == 1:
        input = input.unsqueeze(0)
    if dB:
        input = torchaudio.functional.amplitude_to_DB(input, 20, 0, 0, 90)
    input = input.numpy()

    num_channels, num_frames = input.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, input[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Ch {c+1}" + " Amplitude")
    axes[num_channels - 1].set_xlabel("Time [s]")
    figure.suptitle(name)

    if save:
        name = name + ".svg"
        plt.savefig(name, format="svg")
    if show:
        plt.show(block=end)


def specgram(
    input,
    sample_rate=48000,
    show=False,
    end=False,
    name="specgram",
    save=True,
):
    """
    Plots the spectrogram of an audio data.
    input: audio amplitude
    sample_rate: sample rate (Hz)
    save: True saves the plot as a .svg file, False does not save the plot
    show: True shows the plot, False does not show the plot
    name: name of the saved file
    end: the last plot must be set to True, otherwise the plot will not be displayed
    """
    if input.dim() == 0:
        input = input.unsqueeze(0)
    if input.dim() == 1:
        input = input.unsqueeze(0)

    specgram = torchaudio.transforms.Spectrogram(n_fft=int(sample_rate / 100))(input)
    specgram_db = torchaudio.functional.amplitude_to_DB(specgram, 20, 0, 0, 90)

    num_channels, num_frames = input.shape
    time_axis = num_frames / sample_rate

    fig, axs = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axs = [axs]
    for c in range(num_channels):
        # 贝斯low b是30.868hz，所以最低限为30hz，图好看一点
        im = axs[c].imshow(
            specgram_db[c, :, :],
            origin="lower",
            aspect="auto",
            extent=[0, time_axis, 30, sample_rate / 2],
        )
        fig.colorbar(im, ax=axs[c])
        axs[c].set_yscale("symlog")
        axs[c].set_ylabel(f"Ch{c+1}" + "freq [hz]")

    axs[num_channels - 1].set_xlabel("Time [s]")
    fig.suptitle(name)

    if save:
        name = name + ".svg"
        plt.savefig(name, format="svg")
    if show:
        plt.show(block=end)


def fvtool(
    b,
    a=1,
    sample_rate=48000,
    show=False,
    end=False,
    name="fvtool",
    save=True,
):
    """
    Emulates the functionality of MATLAB's fvtool function.Calculates and plots the frequency response of a filter.
    b: numerator coefficients of the filter
    a: denominator coefficients of the filter
    sample_rate: sample rate (Hz)
    save: True saves the plot as a .svg file, False does not save the plot
    show: True shows the plot, False does not show the plot
    name: name of the saved file
    end: the last plot must be set to True, otherwise the plot will not be displayed
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
