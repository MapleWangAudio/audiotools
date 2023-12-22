import math
import numpy as np


def exp(
    input,
    gain=1,
    mode=1,
    up_limit=1,
    down_limit=1,
    up_curve=1,
    down_curve=1,
):
    """
    Apply exponential distortion to the input signal.

    Args:
        input (float): The input signal.
        gain (float, optional): The gain factor. Defaults to 1.
        mode (int, optional): The distortion mode. Defaults to 1.
        upLimit (float, optional): The upper limit for positive input values. Defaults to 1.
        downLimit (float, optional): The upper limit for negative input values. Defaults to 1.
        upCurve (float, optional): The curve factor for positive input values. Defaults to 1.
        downCurve (float, optional): The curve factor for negative input values. Defaults to 1.

    Returns:
        float: The distorted output signal.
    """
    input *= gain

    if mode == 0:
        output = 1 - math.exp(input * input) + input
        return output

    if mode == 1:
        if input > 0:
            output = up_limit * (1 - math.exp(-up_curve * input))
        else:
            output = -down_limit * (1 - math.exp(down_curve * input))

        return output

    if mode == 2:
        if input > 0:
            output = up_limit * (input + 1 / up_curve * math.exp(-up_curve * input)) - (
                up_limit / up_curve
            )
        else:
            output = -down_limit * (
                input - 1 / down_curve * math.exp(down_curve * input)
            ) - (down_limit / down_curve)

        return output

    if mode == 3:
        if input > 0:
            output = up_limit * (
                (input * input) / 2
                + math.exp(-up_curve * input)
                / (up_curve * up_curve)
                * (up_curve * input + 1)
            ) - (up_limit / (up_curve * up_curve))
        else:
            output = -down_limit * (
                (input * input) / 2
                - math.exp(down_curve * input)
                / (down_curve * down_curve)
                * (down_curve * input - 1)
            ) + (down_limit / (down_curve * down_curve))

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
