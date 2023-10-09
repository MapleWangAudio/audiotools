import math


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
    posi_limit=0,
    nega_limit=0,
    posi_one_over_ratio=0,
    nega_one_over_ratio=0,
    posi_knee=0,
    nega_knee=0,
):
    """
    mono in mono out
    input: audio amplitude
    posi_limit: the value of the limit of the positive input,[0,+∞)
    nega_limit: the value of the limit of the negative input,[0,+∞)
    posi_one_over_ratio: 1/ratio of the positive input,[0,+∞)
    nega_one_over_ratio: 1/ratio of the negative input,[0,+∞)
    posi_knee: the value of the posi knee,[0,+∞)
    nega_knee: the value of the nega knee,[0,+∞)
    """
    if input > 0:
        if input > (posi_limit + posi_knee / 2):
            output = posi_limit + (input - posi_limit) * posi_one_over_ratio
        elif input > (posi_limit - posi_knee / 2) and (posi_knee != 0):
            output = input + (
                (posi_one_over_ratio - 1)
                * (input - posi_limit + posi_knee / 2)
                * (input - posi_limit + posi_knee / 2)
            ) / (2 * posi_knee)
        else:
            output = input
    else:
        input = -input
        if input > (nega_limit + nega_knee / 2):
            output = nega_limit + (input - nega_limit) * nega_one_over_ratio
        elif input > (nega_limit - nega_knee / 2) and (nega_knee != 0):
            output = input + (
                (nega_one_over_ratio - 1)
                * (input - nega_limit + nega_knee / 2)
                * (input - nega_limit + nega_knee / 2)
            ) / (2 * nega_knee)
        else:
            output = input
        output = -output

    return output
