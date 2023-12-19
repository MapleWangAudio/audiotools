import math
import numpy as np


# todo：完善exp
def exp(
    input,
    gain=1,
    mode=1,
    upLimit=1,
    downLimit=1,
    upCurve=1,
    downCurve=1,
):
    """
    Exp type nonlinear
    input: audio amplitude
    gain: the gain of the nonlinear
    mode: the mode of the nonlinear
    upLimit: the limit of the positive nonlinear
    downLimit: the limit of the negative nonlinear
    upCurve: the curve of the positive nonlinear
    downCurve: the curve of the negative nonlinear
    return: distorted audio amplitude
    """
    input *= gain

    if mode == 0:
        output = 1 - math.exp(input * input) + input
        return output

    if mode == 1:
        if input > 0:
            output = upLimit * (1 - math.exp(-upCurve * input))
        else:
            output = -downLimit * (1 - math.exp(downCurve * input))

        return output

    if mode == 2:
        if input > 0:
            output = upLimit * (input + 1 / upCurve * math.exp(-upCurve * input)) - (
                upLimit / upCurve
            )
        else:
            output = -downLimit * (
                input - 1 / downCurve * math.exp(downCurve * input)
            ) - (downLimit / downCurve)

        return output

    if mode == 3:
        if input > 0:
            output = upLimit * (
                (input * input) / 2
                + math.exp(-upCurve * input)
                / (upCurve * upCurve)
                * (upCurve * input + 1)
            ) - (upLimit / (upCurve * upCurve))
        else:
            output = -downLimit * (
                (input * input) / 2
                - math.exp(downCurve * input)
                / (downCurve * downCurve)
                * (downCurve * input - 1)
            ) + (downLimit / (downCurve * downCurve))

        return output


def clip(
    input,
    limit_posi=0.0,
    limit_nega=0.0,
    one_over_ratio_posi=0.0,
    one_over_ratio_nega=0.0,
    knee_posi=0.0,
    knee_nega=0.0,
):
    """
    Apply clipping distortion to the input signal.

    Args:
        input (float): The input signal value.
        limit_posi (float, optional): The positive limit for clipping. Defaults to 0.0.
        limit_nega (float, optional): The negative limit for clipping. Defaults to 0.0.
        one_over_ratio_posi (float, optional): The ratio for positive clipping. Defaults to 0.0.
        one_over_ratio_nega (float, optional): The ratio for negative clipping. Defaults to 0.0.
        knee_posi (float, optional): The knee width for positive clipping. Defaults to 0.0.
        knee_nega (float, optional): The knee width for negative clipping. Defaults to 0.0.

    Returns:
        float: The output signal value after applying clipping distortion.
    """
    if input > 0:
        if input > (limit_posi + knee_posi / 2):
            output = limit_posi + (input - limit_posi) * one_over_ratio_posi
        elif input > (limit_posi - knee_posi / 2) and (knee_posi != 0):
            output = input + (
                (one_over_ratio_posi - 1)
                * (input - limit_posi + knee_posi / 2)
                * (input - limit_posi + knee_posi / 2)
            ) / (2 * knee_posi)
        else:
            output = input
    else:
        input = -input
        if input > (limit_nega + knee_nega / 2):
            output = limit_nega + (input - limit_nega) * one_over_ratio_nega
        elif input > (limit_nega - knee_nega / 2) and (knee_nega != 0):
            output = input + (
                (one_over_ratio_nega - 1)
                * (input - limit_nega + knee_nega / 2)
                * (input - limit_nega + knee_nega / 2)
            ) / (2 * knee_nega)
        else:
            output = input
        output = -output

    return output


def clip_array(
    input,
    limit_posi=0.0,
    limit_nega=0.0,
    one_over_ratio_posi=0.0,
    one_over_ratio_nega=0.0,
    knee_posi=0.0,
    knee_nega=0.0,
):
    """
    Apply clipping distortion to the input signal array.

    Args:
        input (ndarray): Input array to be clipped.
        limit_posi (float, optional): Positive limit for clipping. Defaults to 0.0.
        limit_nega (float, optional): Negative limit for clipping. Defaults to 0.0.
        one_over_ratio_posi (float, optional): Ratio for positive clipping. Defaults to 0.0.
        one_over_ratio_nega (float, optional): Ratio for negative clipping. Defaults to 0.0.
        knee_posi (float, optional): Positive knee width for soft clipping. Defaults to 0.0.
        knee_nega (float, optional): Negative knee width for soft clipping. Defaults to 0.0.

    Returns:
        ndarray: Clipped output array.
    """
    input_posi = np.where(input > 0, input, 0)
    input_nega = np.where(input > 0, 0, -input)
    output_posi = input_posi
    output_nega = input_nega

    output_posi = np.where(
        (input_posi > (limit_posi - knee_posi / 2)) & (knee_posi != 0),
        input_posi
        + (
            (one_over_ratio_posi - 1)
            * (input_posi - limit_posi + knee_posi / 2)
            * (input_posi - limit_posi + knee_posi / 2)
        )
        / (2 * knee_posi),
        output_posi,
    )
    output_posi = np.where(
        input_posi > (limit_posi + knee_posi / 2),
        limit_posi + (input_posi - limit_posi) * one_over_ratio_posi,
        output_posi,
    )

    output_nega = np.where(
        (input_nega > (limit_nega - knee_nega / 2)) & (knee_nega != 0),
        input_nega
        + (
            (one_over_ratio_nega - 1)
            * (input_nega - limit_nega + knee_nega / 2)
            * (input_nega - limit_nega + knee_nega / 2)
        )
        / (2 * knee_nega),
        output_nega,
    )
    output_nega = np.where(
        input_nega > (limit_nega + knee_nega / 2),
        limit_nega + (input_nega - limit_nega) * one_over_ratio_nega,
        output_nega,
    )
    output_nega = -output_nega

    output = np.where(
        input > 0,
        output_posi,
        output_nega,
    )

    return output
