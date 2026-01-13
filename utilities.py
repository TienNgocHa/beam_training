import numpy as np


def dB_to_natural(x_in):
    return 10**(x_in / 10)


def natural_to_dB(x_in):
    return 10 * np.log10(x_in)
