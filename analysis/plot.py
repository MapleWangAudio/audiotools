import matplotlib.pyplot as plt
import torchaudio
import torchaudio.functional as F
import torchvision
import torch.utils.tensorboard as tb
import torch
import numpy as np
from scipy import signal


def waveform(
    input,
    sr=48000,
    end=False,
    name="waveform.png",
    save=True,
):
    """
    Plots the waveform of an audio data.
    input: audio amplitude
    sr: sample rate (Hz)
    end: the last plot must be set to True, otherwise the plot will not be displayed
    name: name of the saved file
    save: True saves the plot as a file, False does not save the plot
    """
    num_channels, num_frames = input.shape
    time_axis = np.arange(0, num_frames) / sr

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
        save_format = name[-3:]
        name = name
        plt.savefig(name, format=save_format)

    plt.show(block=end)


def specgram(
    input,
    sr=48000,
    end=False,
    name="specgram.png",
    save=True,
):
    """
    Plots the spectrogram of an audio data.
    input: audio amplitude
    sr: sample rate (Hz)
    end: the last plot must be set to True, otherwise the plot will not be displayed
    name: name of the saved file
    save: True saves the plot as a file, False does not save the plot
    """
    # 检查是否有可用的 GPU 设备
    if torch.cuda.is_available():
        # 设置默认的 GPU 设备为 GPU 0
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # 将 tensor 移动到设备 device 上
    input = torch.tensor(input, device=device)
    specgram = torchaudio.transforms.Spectrogram(n_fft=int(sr / 100))(input)
    specgram_db = F.amplitude_to_DB(specgram, 20, 0, 0, 90)

    num_channels, num_frames = input.shape
    time_axis = num_frames / sr

    fig, axs = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axs = [axs]
    for c in range(num_channels):
        # 贝斯low b是30.868hz，所以最低限为30hz，图好看一点
        im = axs[c].imshow(
            specgram_db[c, :, :],
            origin="lower",
            aspect="auto",
            extent=[0, time_axis, 30, sr / 2],
        )
        fig.colorbar(im, ax=axs[c])
        axs[c].set_yscale("symlog")
        axs[c].set_ylabel(f"Ch{c+1}" + "freq [hz]")

    axs[num_channels - 1].set_xlabel("Time [s]")
    fig.suptitle(name)

    if save:
        save_format = name[-3:]
        name = name
        plt.savefig(name, format=save_format)

    plt.show(block=end)


def fvtool(
    b,
    a="None",
    sr=48000,
    end=False,
    name="fvtool.png",
    save=True,
):
    """
    Emulates the functionality of MATLAB's fvtool function.Calculates and plots the frequency response of a filter.
    b: numerator coefficients of the filter
    a: denominator coefficients of the filter
    sr: sample rate (Hz)
    end: the last plot must be set to True, otherwise the plot will not be displayed
    name: name of the saved file
    save: True saves the plot as a file, False does not save the plot
    """
    if isinstance(a, str):
        worN = len(b)
        a = 1
    else:
        worN = 8192

    w, h = signal.freqz(b, a, worN=worN, fs=sr)
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
        save_format = name[-3:]
        name = name
        plt.savefig(name, format=save_format)

    plt.show(block=end)


def show_in_tb(img_path, name="Image"):
    """
    Shows an image in TensorBoard. To lanuch tb in vscode, use ctrl+shift+p, then input >python:launch tesnorboard.
    img_path: a string list path to the image, just read png and jpg.
    name: name of the image in tensorboard
    """
    writer = tb.SummaryWriter()

    # Load the image
    for i in range(len(img_path)):
        img = torchvision.io.read_image(img_path[i], torchvision.io.ImageReadMode.RGB)
        # Add the image to TensorBoard
        writer.add_image(name, img)

    # Close the writer
    writer.close()
