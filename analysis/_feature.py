import torchaudio
import torch
from ..process import time_coefficient_computer, smooth_filter, to_mono
import math


class peak:
    def digital(
        input,
        sr=48000,
        lookback=1,
        lookahead=1,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Digital Type
        input: audio amplitude
        sr: sample rate (Hz)
        lookback: peak pre window (ms)
        lookahead: peak post window (ms)
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value (dB)
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = to_mono(input)

        pre_pad_length = int(lookback * 0.001 * sr)
        post_pad_length = int(lookahead * 0.001 * sr)

        channel, input_length = input.shape
        peak = torch.zeros_like(input)
        input = torch.abs(input)

        if (pre_pad_length + post_pad_length) == 0:
            peak = input
        else:
            # Pad input with zeros to the nearest multiple of time*sr
            input = torch.nn.functional.pad(
                input,
                (
                    pre_pad_length,
                    post_pad_length,
                ),
                "constant",
                0,
            )

            for i in range(channel):
                for j in range(input_length):
                    peak[i, j] = torch.max(
                        input[i, j : j + pre_pad_length + post_pad_length]
                    )

        peak = torchaudio.functional.amplitude_to_DB(peak, 20, 0, 0)

        return peak

    def analog_prue(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        mode: 0 is Peak Detectors, 1 is Level Corrected Peak Detectors in branch style, 2 is Smooth Peak Detectors in branch style
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value (dB)
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = to_mono(input)

        attack_coeff = time_coefficient_computer(attack_time, sr)
        release_coeff = time_coefficient_computer(release_time, sr)

        channel, input_length = input.shape
        peak = torch.zeros_like(input)
        input = torch.abs(input)

        # Peak Detectors
        for i in range(channel):
            for j in range(1, input_length):
                peak[i, j] = release_coeff * peak[i, j - 1] + (1 - attack_coeff) * max(
                    (input[i, j] - peak[i, j - 1]), 0
                )

        peak = torchaudio.functional.amplitude_to_DB(peak, 20, 0, 0)

        return peak

    def analog_level_corrected(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        mode=1,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        mode: 0 is decoupled style, 1 is branch style
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value (dB)
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = to_mono(input)

        attack_coeff = time_coefficient_computer(attack_time, sr)
        release_coeff = time_coefficient_computer(release_time, sr)

        channel, input_length = input.shape
        peak = torch.zeros_like(input)
        input = torch.abs(input)

        if mode == 0:
            peak_state = torch.zeros_like(input)
            for i in range(channel):
                for j in range(1, input_length):
                    peak_state[i, j] = max(
                        input[i, j], release_coeff * peak_state[i, j - 1]
                    )
            peak = smooth_filter(peak_state, attack_coeff, 1)

        if mode == 1:
            for i in range(channel):
                for j in range(1, input_length):
                    if input[i, j] > peak[i, j - 1]:
                        peak[i, j] = (
                            attack_coeff * peak[i, j - 1]
                            + (1 - attack_coeff) * input[i, j]
                        )

                    else:
                        peak[i, j] = release_coeff * peak[i, j - 1]

        peak = torchaudio.functional.amplitude_to_DB(peak, 20, 0, 0)

        return peak

    def analog_smooth(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        mode=1,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        mode: 0 is decoupled style, 1 is branch style
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value (dB)
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = to_mono(input)

        attack_coeff = time_coefficient_computer(attack_time, sr)
        release_coeff = time_coefficient_computer(release_time, sr)

        channel, input_length = input.shape
        peak = torch.zeros_like(input)
        input = torch.abs(input)

        if mode == 0:
            peak_state = torch.zeros_like(input)
            for i in range(channel):
                for j in range(1, input_length):
                    peak_state[i, j] = max(
                        input[i, j],
                        release_coeff * peak_state[i, j - 1]
                        + (1 - release_coeff) * input[i, j],
                    )
            peak = smooth_filter(peak_state, attack_coeff, 1)

        if mode == 1:
            for i in range(channel):
                for j in range(1, input_length):
                    if input[i, j] > peak[i, j - 1]:
                        peak[i, j] = (
                            attack_coeff * peak[i, j - 1]
                            + (1 - attack_coeff) * input[i, j]
                        )

                    else:
                        peak[i, j] = (
                            release_coeff * peak[i, j - 1]
                            + (1 - release_coeff) * input[i, j]
                        )

        peak = torchaudio.functional.amplitude_to_DB(peak, 20, 0, 0)

        return peak


class RMS:
    def digital(
        input,
        sr=48000,
        lookback=1,
        lookahead=1,
        multichannel=False,
    ):
        """
        Computes the root mean square (RMS) of an audio input. Digital Type
        input: audio amplitude
        sr: sample rate (Hz)
        lookback: RMS pre window (ms)
        lookahead: RMS post window (ms)
        multichannel: True calculates RMS value for each channel, False calculates RMS value for all channels
        return: RMS value (dB)
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = to_mono(input)

        pre_pad_length = int(lookback * 0.001 * sr)
        post_pad_length = int(lookahead * 0.001 * sr)

        channel, input_length = input.shape
        RMS = torch.zeros_like(input)

        # Pad input with zeros to the nearest multiple of time*sr
        input = torch.nn.functional.pad(
            input,
            (
                pre_pad_length,
                post_pad_length,
            ),
            "constant",
            0,
        )

        for i in range(channel):
            for j in range(input_length):
                RMS[i, j] = torch.sqrt(
                    torch.mean(
                        torch.square(input[i, j : j + pre_pad_length + post_pad_length])
                    )
                )

        RMS = torchaudio.functional.amplitude_to_DB(RMS, 20, 0, 0)

        return RMS

    def analog(
        input,
        sr=48000,
        time=1,
        multichannel=False,
    ):
        """
        Computes the root mean square (RMS) of an audio input. Analog Type
        input: audio amplitude
        sr: sample rate (Hz)
        time: RMS window (ms)
        multichannel: True calculates RMS value for each channel, False calculates RMS value for all channels
        return: RMS value (dB)
        """
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = to_mono(input)

        coeff = time_coefficient_computer(time, sr)

        RMS = smooth_filter(input**2, coeff, 1)
        RMS = torch.sqrt(RMS)

        RMS = torchaudio.functional.amplitude_to_DB(RMS, 20, 0, 0)

        return RMS
