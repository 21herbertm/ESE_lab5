# -*- coding: utf-8 -*-
"""
ESE 224
LAB_5
MELANIE HERBERT, ALINA HO
"""

import numpy as np

class gaussian_pulse(object):

# MU = MEAN OF GAUSSIAN PULSE
# SIGMA = VARIANCE OF GAUSSIAN PULSE
# FS = SAMPLING FREQUENCY
# T = DURATION OF SIGNAL

    def __init__(self, mu, sigma, T, fs):
        self.N = np.int(np.floor( T * fs))

        # Create the active part
        # self.sig = signal.gaussian(self.N, std=sigma)
        self.t = np.arange(0, T, 1 / fs)
        self.sig = np.exp(-(self.t-mu)**2 / (2 * sigma**2))

        # Create the time array
        self.t = np.arange(0, T, 1 / fs)