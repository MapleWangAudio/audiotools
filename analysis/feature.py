import torch
import torchaudio.functional as F
from .. import process as P


class peak:
    def digital(
        input,
        sr=48000,
        lookback=1,
        lookahead=1,
        is_dB=False,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Digital Type
        input: audio amplitude
        sr: sample rate (Hz)
        lookback: peak pre window (ms)
        lookahead: peak post window (ms)
        is_dB: Effecting pad_value. True calculates peak value in dB, False calculates peak value in amplitude.
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        if is_dB == True:
            pad_value = -90
        else:
            pad_value = 0

        pre_pad_length = int(lookback * 0.001 * sr)
        post_pad_length = int(lookahead * 0.001 * sr)

        channel, input_length = input.shape
        peak = torch.zeros_like(input)

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
                pad_value,
            )

            for i in range(channel):
                for j in range(input_length):
                    peak[i, j] = torch.max(
                        input[i, j : j + pre_pad_length + post_pad_length]
                    )

        return peak

    def analog_prue(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        attack_range_low=0.1,
        attack_range_high=0.9,
        release_range_low=0.1,
        release_range_high=0.9,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        attack_range_low: attack control range low (0,1)
        attack_range_high: attack control range high (0,1)
        release_range_low: release control range low (0,1)
        release_range_high: release control range high (0,1)
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        attack_coeff = P.time_coefficient_computer(
            attack_time, sr, attack_range_low, attack_range_high
        )
        release_coeff = P.time_coefficient_computer(
            release_time, sr, release_range_low, release_range_high
        )

        channel, input_length = input.shape
        peak = torch.zeros_like(input)

        # Peak Detectors
        for i in range(channel):
            for j in range(1, input_length):
                peak[i, j] = release_coeff * peak[i, j - 1] + (1 - attack_coeff) * max(
                    (input[i, j] - peak[i, j - 1]), 0
                )

        return peak

    def analog_level_corrected(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        attack_range_low=0.1,
        attack_range_high=0.9,
        release_range_low=0.1,
        release_range_high=0.9,
        mode=1,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        attack_range_low: attack control range low (0,1)
        attack_range_high: attack control range high (0,1)
        release_range_low: release control range low (0,1)
        release_range_high: release control range high (0,1)
        mode: 0 is decoupled style, 1 is branch style
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        attack_coeff = P.time_coefficient_computer(
            attack_time, sr, attack_range_low, attack_range_high
        )
        release_coeff = P.time_coefficient_computer(
            release_time, sr, release_range_low, release_range_high
        )

        channel, input_length = input.shape
        peak = torch.zeros_like(input)

        if mode == 0:
            peak_state = torch.zeros_like(input)
            for i in range(channel):
                for j in range(1, input_length):
                    peak_state[i, j] = max(
                        input[i, j], release_coeff * peak_state[i, j - 1]
                    )
            peak = P.smooth_filter(peak_state, attack_coeff)

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

        return peak

    def analog_smooth(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        attack_range_low=0.1,
        attack_range_high=0.9,
        release_range_low=0.1,
        release_range_high=0.9,
        mode=1,
        multichannel=False,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        attack_range_low: attack control range low (0,1)
        attack_range_high: attack control range high (0,1)
        release_range_low: release control range low (0,1)
        release_range_high: release control range high (0,1)
        mode: 0 is decoupled style, 1 is branch style
        multichannel: True calculates peak value for each channel, False calculates peak value for all channels
        return: peak value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        attack_coeff = P.time_coefficient_computer(
            attack_time, sr, attack_range_low, attack_range_high
        )
        release_coeff = P.time_coefficient_computer(
            release_time, sr, release_range_low, release_range_high
        )

        channel, input_length = input.shape
        peak = torch.zeros_like(input)

        if mode == 0:
            peak_state = torch.zeros_like(input)
            for i in range(channel):
                for j in range(1, input_length):
                    peak_state[i, j] = max(
                        input[i, j],
                        release_coeff * peak_state[i, j - 1]
                        + (1 - release_coeff) * input[i, j],
                    )
            peak = P.smooth_filter(peak_state, attack_coeff)

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

        return peak


class RMS:
    def digital(
        input,
        sr=48000,
        lookback=1,
        lookahead=1,
        is_dB=False,
        multichannel=False,
    ):
        """
        Computes the root mean square (RMS) of an audio input. Digital Type
        input: audio amplitude
        sr: sample rate (Hz)
        lookback: RMS pre window (ms)
        lookahead: RMS post window (ms)
        is_dB: Effecting pad_value. True calculates RMS value in dB, False calculates RMS value in amplitude.
        multichannel: True calculates RMS value for each channel, False calculates RMS value for all channels
        return: RMS value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        if is_dB == True:
            pad_value = -90
        else:
            pad_value = 0

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
            pad_value,
        )

        for i in range(channel):
            for j in range(input_length):
                RMS[i, j] = torch.sqrt(
                    torch.mean(
                        torch.square(input[i, j : j + pre_pad_length + post_pad_length])
                    )
                )

        return RMS

    def analog_prue(
        input,
        sr=48000,
        time=1,
        range_low=0.1,
        range_high=0.9,
        multichannel=False,
    ):
        """
        Computes the root mean square (RMS) of an audio input. Analog Type
        input: audio amplitude
        sr: sample rate (Hz)
        time: attack time (ms)
        range_low: time control range low (0,1)
        range_high: time control range high (0,1)
        multichannel: True calculates RMS value for each channel, False calculates RMS value for all channels
        return: RMS value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        coeff = P.time_coefficient_computer(time, sr, range_low, range_high)

        input = torch.square(input)

        RMS = P.smooth_filter(input, coeff)
        RMS = torch.sqrt(RMS)

        return RMS

    def analog_level_corrected(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        attack_range_low=0.1,
        attack_range_high=0.9,
        release_range_low=0.1,
        release_range_high=0.9,
        mode=1,
        multichannel=False,
    ):
        """
        Computes the root mean square (RMS) of an audio input. Analog Type
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        attack_range_low: attack control range low (0,1)
        attack_range_high: attack control range high (0,1)
        release_range_low: release control range low (0,1)
        release_range_high: release control range high (0,1)
        mode: 0 is decoupled style, 1 is branch style
        multichannel: True calculates RMS value for each channel, False calculates peak value for all channels
        return: RMS value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        attack_coeff = P.time_coefficient_computer(
            attack_time, sr, attack_range_low, attack_range_high
        )
        release_coeff = P.time_coefficient_computer(
            release_time, sr, release_range_low, release_range_high
        )

        channel, input_length = input.shape
        input = torch.square(input)
        RMS = torch.zeros_like(input)

        if mode == 0:
            RMS_state = torch.zeros_like(input)
            for i in range(channel):
                for j in range(1, input_length):
                    RMS_state[i, j] = (
                        input[i, j] + release_coeff * RMS_state[i, j - 1]
                    ) / 2
            RMS = P.smooth_filter(RMS_state, attack_coeff)

        if mode == 1:
            for i in range(channel):
                for j in range(1, input_length):
                    if input[i, j] > RMS[i, j - 1]:
                        RMS[i, j] = (
                            attack_coeff * RMS[i, j - 1]
                            + (1 - attack_coeff) * input[i, j]
                        )

                    else:
                        RMS[i, j] = release_coeff * RMS[i, j - 1]

        RMS = torch.sqrt(RMS)

        return RMS

    def analog_smooth(
        input,
        sr=48000,
        attack_time=1,
        release_time=1,
        attack_range_low=0.1,
        attack_range_high=0.9,
        release_range_low=0.1,
        release_range_high=0.9,
        mode=1,
        multichannel=False,
    ):
        """
        Computes the RMS value of an audio input. Analog Type.
        input: audio amplitude
        sr: sample rate (Hz)
        attack_time: attack time (ms)
        release_time: release time (ms)
        attack_range_low: attack control range low (0,1)
        attack_range_high: attack control range high (0,1)
        release_range_low: release control range low (0,1)
        release_range_high: release control range high (0,1)
        mode: 0 is decoupled style, 1 is branch style
        multichannel: True calculates RMS value for each channel, False calculates peak value for all channels
        return: RMS value
        """
        if input.dim() == 0:
            input = input.unsqueeze(0)
        if input.dim() == 1:
            input = input.unsqueeze(0)

        if multichannel == False:
            input = P.to_mono(input)

        attack_coeff = P.time_coefficient_computer(
            attack_time, sr, attack_range_low, attack_range_high
        )
        release_coeff = P.time_coefficient_computer(
            release_time, sr, release_range_low, release_range_high
        )

        channel, input_length = input.shape
        input = torch.square(input)
        RMS = torch.zeros_like(input)

        if mode == 0:
            peak_state = torch.zeros_like(input)
            for i in range(channel):
                for j in range(1, input_length):
                    peak_state[i, j] = (
                        input[i, j]
                        + release_coeff * peak_state[i, j - 1]
                        + (1 - release_coeff) * input[i, j]
                    ) / 3
            RMS = P.smooth_filter(peak_state, attack_coeff)

        if mode == 1:
            for i in range(channel):
                for j in range(1, input_length):
                    if input[i, j] > RMS[i, j - 1]:
                        RMS[i, j] = (
                            attack_coeff * RMS[i, j - 1]
                            + (1 - attack_coeff) * input[i, j]
                        )

                    else:
                        RMS[i, j] = (
                            release_coeff * RMS[i, j - 1]
                            + (1 - release_coeff) * input[i, j]
                        )

        RMS = torch.sqrt(RMS)

        return RMS


class drc_time:
    def test_signal(
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
        stage1 = P.generate_signal(freq, sr, amplitude1, length1)
        stage2 = P.generate_signal(freq, sr, amplitude2, length2)
        stage3 = P.generate_signal(freq, sr, amplitude3, length3)
        signal = torch.cat((stage1, stage2, stage3), 1)
        return signal

    def time_extract(test, result, sr=96000):
        """
        Extract the time from the time test signal
        test: time test signal raw
        result: time test signal processed
        sr: sample rate of the signal (hz)
        return: time featrue
        """
        if test.dim() == 0:
            test = test.unsqueeze(0)
        if test.dim() == 1:
            test = test.unsqueeze(0)

        if result.dim() == 0:
            result = result.unsqueeze(0)
        if result.dim() == 1:
            result = result.unsqueeze(0)

        result = torch.abs(result[0, :])
        test = torch.abs(test[0, :])

        result = peak.digital(result, sr, 20, 0)
        test = peak.digital(test, sr, 20, 0)

        gain = result / test

        gain[0, 0:100] = 1
        gain = torch.where(torch.isnan(gain), torch.tensor(3.1623e-05), gain)

        gain = F.amplitude_to_DB(gain, 20, 0, 0, 90)
        return gain


class drc_ratio:
    def test_signal(freq=1000, sr=96000):
        """
        Generate a signal for ratio test
        freq: test signal frequency
        sr: sample rate of the signal (hz)
        return: signal
        """
        stage = torch.zeros(91, sr * 5)
        signal = torch.zeros(1)
        for i in range(91):
            amp = F.DB_to_amplitude(torch.tensor(i - 90), 1, 0.5)
            stage_stage = P.generate_signal(freq, sr, amp, 5)
            stage[i, :] = stage_stage[0, :]
            signal = torch.cat((signal, stage[i, :]), 0)
        signal = signal[1:]
        signal = signal.unsqueeze(0)

        return signal

    def ratio_extract(ratiotest, sr):
        """
        Extract the ratio from the ratio test signal
        ratiotest: ratio test signal processed
        sr: sample rate of the signal (hz)
        return: ratio feature
        """
        if ratiotest.dim() == 0:
            ratiotest = ratiotest.unsqueeze(0)
        if ratiotest.dim() == 1:
            ratiotest = ratiotest.unsqueeze(0)

        output = torch.zeros(91)
        ratiotest = ratiotest[0, :]

        # 先选取每个阶段的最后0.15s，因为前面没意义，同时为peak计算节省运算量
        ratiotest_stage = torch.zeros(1)
        for i in range(91):
            stage = ratiotest[int(i * sr * 5 + sr * 4.85) : (i + 1) * sr * 5]
            ratiotest_stage = torch.cat((ratiotest_stage, stage), 0)
        ratiotest = ratiotest_stage[1:]

        ratiotest = peak.digital(ratiotest, sr, 20, 0)
        ratiotest = ratiotest[0, :]

        for i in range(91):
            # 由于往前看了20ms，前几个值会有问题，所以删去每个阶段的前50ms
            stage = ratiotest[int(i * sr * 0.15 + sr * 0.05) : int((i + 1) * sr * 0.2)]
            stage = min(stage)
            output[i] = stage
        output = output.unsqueeze(0)
        return output
