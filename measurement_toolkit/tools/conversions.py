import numpy as np


def amplitude_to_dB(amplitude, zero_amplitude=1):
    return 20 * np.log10(amplitude/zero_amplitude)


def dB_to_amplitude(dB, zero_amplitude=1):
    return 10**(dB/20) * zero_amplitude


def amplitude_to_dBm(amplitude):
    return amplitude_to_dB(amplitude, zero_amplitude=1e-3)


def dBm_to_amplitude(dB):
    return dB_to_amplitude(dB, zero_amplitude=1e-3)
