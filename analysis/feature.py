from .. import process as P
import numpy as np


class peak:
    def digital(input):
        """
        Computes the peak value of an audio input. Digital Type
        input: audio amplitude
        return: peak values
        """

        return np.max(input)

    def analog_prue(
        input,
        output_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the peak value of an audio input. Analog Type.
        input: audio amplitude
        output_old: peak value of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: peak value
        """
        output = release_coeff * output_old + (1 - attack_coeff) * max(
            (input - output_old), 0
        )

        return output

    def analog_level_corrected0(
        input,
        state_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the peak value of an audio input in decoupled style. Analog Type.
        input: audio amplitude
        state_old: peak state of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: peak value, peak state
        """
        state = max(input, release_coeff * state_old)
        peak = P.smoother(state, state_old, attack_coeff)

        return peak, state

    def analog_level_corrected1(
        input,
        output_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the peak value of an audio input in branch style. Analog Type.
        input: audio amplitude
        output_old: peak value of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: peak value
        """
        if input > output_old:
            peak = attack_coeff * output_old + (1 - attack_coeff) * input
        else:
            peak = release_coeff * output_old

        return peak

    def analog_smooth0(
        input,
        state_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the peak value of an audio input in decoupled style. Analog Type.
        input: audio amplitude
        state_old: peak state of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: peak value, state
        """
        state = max(
            input,
            release_coeff * state_old + (1 - release_coeff) * input,
        )
        peak = P.smooth_filter(state, attack_coeff)

        return peak, state

    def analog_smooth0(
        input,
        output_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the peak value of an audio input in branch style. Analog Type.
        input: audio amplitude
        output_old: peak value of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: peak value
        """
        if input > output_old:
            peak = attack_coeff * output_old + (1 - attack_coeff) * input
        else:
            peak = release_coeff * output_old + (1 - release_coeff) * input

        return peak


class RMS:
    def digital(input):
        """
        Computes the root mean square (RMS) of an audio input. Digital Type
        input: audio amplitude
        return: RMS value
        """

        return np.sqrt(np.mean(np.square(input)))

    def analog_prue(
        input,
        output_old,
        coeff,
    ):
        """
        Computes the root mean square (RMS) of an audio input. Analog Type
        input: audio amplitude
        output_old: RMS value of the last sample
        return: RMS value
        """
        output = P.smoother(input**2, output_old**2, coeff)
        output = output**0.5

        return output

    def analog_level_corrected0(
        input,
        state_old,
        output_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the root mean square (RMS) of an audio input in decoupled style. Analog Type
        input: audio amplitude
        state_old: RMS state of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: RMS value
        """
        state = (input**2 + (release_coeff * state_old) ** 2) / 2
        state = state**0.5

        output = attack_coeff * output_old + (1 - attack_coeff) * state

        return output, state

    def analog_level_corrected1(
        input,
        output_old,
        attack_coeff=1,
        release_coeff=1,
    ):
        """
        Computes the root mean square (RMS) of an audio input in branch style. Analog Type
        input: audio amplitude
        output_old: RMS value of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: RMS value
        """
        if input > output_old:
            output = attack_coeff * output_old**2 + (1 - attack_coeff) * input**2
        else:
            output = release_coeff * output_old**2

        output = output**0.5

        return output

    def analog_smooth0(
        input,
        state_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the RMS value of an audio input in decoupled style. Analog Type.
        input: audio amplitude
        state_old: RMS state of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: RMS value, state
        """
        state = (
            input**2 + (release_coeff * state_old + (1 - release_coeff) * input) ** 2
        ) / 2
        state = state**0.5

        output = attack_coeff * output + (1 - attack_coeff) * state

        return output, state

    def analog_smooth1(
        input,
        output_old,
        attack_coeff,
        release_coeff,
    ):
        """
        Computes the RMS value of an audio input in branch style. Analog Type.
        input: audio amplitude
        output_old: RMS value of the last sample
        attack_coeff: attack coefficient
        release_coeff: release coefficient
        return: RMS value
        """
        if input > output_old:
            output = attack_coeff * output_old**2 + (1 - attack_coeff) * input**2

        else:
            output = release_coeff * output_old**2 + (1 - release_coeff) * input**2

        output = output**0.5

        return output


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
        signal = np.hstack((stage1, stage2, stage3))
        return signal

    def time_extract(test, result, sr=96000):
        """
        Extract the time from the time test signal
        test: time test signal raw
        result: time test signal processed
        sr: sample rate of the signal (hz)
        return: time featrue
        """
        result = np.abs(result[0, :])
        test = np.abs(test[0, :])

        for i in range(result.size(0)):
            if i < (0.02 * sr):
                result[i] = peak.digital(result[0, :i])
                test = peak.digital(test[0, :i])
            else:
                result = peak.digital(result[0, i - int(0.02 * sr) : i])
                test = peak.digital(test[0, i - int(0.02 * sr) : i])

        gain = result / test

        gain[0, 0:100] = 1
        gain = np.where(np.isnan(gain), np.tensor(3.1623e-05), gain)

        gain = amp2dB(gain)
        return gain


class drc_ratio:
    def test_signal(freq=1000, sr=96000):
        """
        Generate a signal for ratio test
        freq: test signal frequency
        sr: sample rate of the signal (hz)
        return: signal
        """
        stage = np.zeros(91, sr * 5)
        signal = np.zeros(1)
        for i in range(91):
            amp = dB2amp(i - 90)
            stage_stage = P.generate_signal(freq, sr, amp, 5)
            stage[i, :] = stage_stage[0, :]
            signal = np.vstack(signal, stage[i, :])
        signal = signal[1:]
        signal = signal[np.newaxis, :]

        return signal

    def ratio_extract(ratiotest, sr):
        """
        Extract the ratio from the ratio test signal
        ratiotest: ratio test signal processed
        sr: sample rate of the signal (hz)
        return: ratio feature
        """
        output = np.zeros(91)
        ratiotest = ratiotest[0, :]

        # 先选取每个阶段的最后0.15s，因为前面没意义，同时为peak计算节省运算量
        ratiotest_stage = np.zeros(1)
        for i in range(91):
            stage = ratiotest[int(i * sr * 5 + sr * 4.85) : (i + 1) * sr * 5]
            ratiotest_stage = np.vstack(ratiotest_stage, stage)
        ratiotest = ratiotest_stage[1:]

        for i in range(ratiotest.size(0)):
            if i < (0.02 * sr):
                ratiotest[i] = peak.digital(ratiotest[0, :i])
            else:
                ratiotest[i] = peak.digital(ratiotest[0, i - int(0.02 * sr) : i])

        for i in range(91):
            # 由于往前看了20ms，前几个值会有问题，所以删去每个阶段的前50ms
            stage = ratiotest[int(i * sr * 0.15 + sr * 0.05) : int((i + 1) * sr * 0.2)]
            stage = min(stage)
            output[i] = stage

        output = output[np.newaxis, :]

        return output


def amp2dB(input):
    """
    Convert amplitude to dB
    input: audio amplitude
    return: audio dB [-90,+∞)
    """
    output = 20 * np.log10(input + 3.1623e-05)
    return output


def dB2amp(input):
    """
    Convert dB to amplitude
    input: audio dB
    return: audio amplitude
    """
    output = 10 ** (input / 20)
    return output
