import torchaudio
import torch


def peak(waveform):
    waveform_max = torch.max(torch.abs(waveform))
    return torchaudio.transforms.AmplitudeToDB("magnitude")(waveform_max)


def RMS(waveform):
    """
    Computes the root mean square (RMS) of an audio waveform.
    """
    RMS = torch.sqrt(torch.mean(torch.square(waveform)))
    return torchaudio.transforms.AmplitudeToDB("magnitude")(RMS)
