import torchaudio.transforms as transforms
import matplotlib.pyplot as plt
import torchaudio
import torch


def waveform(
    waveform, sample_rate=44100, save=True, show=False, name="waveform", end=False
):
    """
    Plots the waveform of an audio data.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
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
    waveform, sample_rate=44100, save=True, show=False, name="specgram", end=False
):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    specgram = torchaudio.transforms.Spectrogram(n_fft=int(sample_rate / 100))(waveform)
    specgram_db = torchaudio.transforms.AmplitudeToDB()(specgram)

    num_channels, num_frames = waveform.shape
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
