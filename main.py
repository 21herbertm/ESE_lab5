# -*- coding: utf-8 -*-
"""
ESE 224

LAB 5
FEB 21 2022

MELANIE HERBERT, ALINA HO
"""

import matplotlib.pyplot as plt

from gaussian_pulse import gaussian_pulse
from subsample import subsample
from dft import dft
from reconstruct import reconstruct

###############################################################################
############################ Q U E S T I O N 2.2 ##############################
###############################################################################

"""
RELATING TO QUESTION 2.1: FIRST DERVIVE EQUIVALENT OF THEOREM RELATING THE SPECTRA OF THE DISCRETE TIME SIGNAL AND THE SUBSAMPLED VERSION

DTFT IS A DIRAC TRAIN WITH SPIKES SPACED BY SUBSAMPLING FREQUENCY IS THE INVERSE OF THE TIME
"""

# PARAMETERS BEING PASSED INTO THE GAUSSIAN PULSE CLASS AND FUNCTION

mu = 1 # MU = MEAN OF GAUSSIAN PULSE
sigma = 0.0001 # SIGMA = VARIANCE OF GAUSSIAN PULSE
f_s = 40000 # FS = SAMPLING FREQUENCY
f_ss = 4000 #FSS = SUBSAMPLING FREQUENCY
T = 2 # T = DURATION OF SIGNAL OR TOTAL OBSERVATION PERIOD
N = T*f_s

# CALL TO GAUSSIAN PULSE CLASS
# CREATES GAUSSIAN PULSE
gaussian_obj = gaussian_pulse(mu, sigma, T, f_s)
x = gaussian_obj.sig
t = gaussian_obj.t

# CALL TO SUBSAMPLE CLASS WITH SOLVE FUNCTION
subsample_obj = subsample(x, 1/f_s, 1/f_ss)
x_s, x_delta = subsample_obj.solve()

# CODE FOR PLOTTING THE ORIGINAL SIGNAL AND SUBSAMPLED SIGNAL GRAPH
# PROBLEM 2.2
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('QUESTION 2.2: Original signal and subsampled signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal')
axs[1].plot(t, x_delta)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal')
plt.savefig('signal_and_subsampled_time_'+ str(sigma) + '.png')
plt.show()

###############################################################################
############################ Q U E S T I O N 2.3 ##############################
###############################################################################

"""
TAKE DFT OF THE FUNCTIONS OF THE RETURNED SUBSAMPLED SIGNAL AND THE ITS DELTA TRAIN REPRESENTATION

ONLY VARY THE STANDARD DEVIATION OF THE GAUSSIAN PULSE 

PURPOSE IS TO OBSERVE CASES WITH AND ALSO WITHOUT ALIASING
"""

DFT_x = dft(x, f_s) # DFT USES X SIGNAL AND F_S SAMPLING FREQUENCY
[_, _, freqs_c, X_c] = DFT_x.solve_using_numpy_fft()

DFT_x_delta = dft(x_delta, f_s)
[_, _, _, X_delta_c] = DFT_x_delta.solve_using_numpy_fft()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('QUESTION 2.3: DFT of original signal and subsampled signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(freqs_c, abs(X_c))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('DFT')
axs[1].plot(freqs_c, abs(X_delta_c))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('DFT')
plt.savefig('signal_and_subsampled_freq_'+ str(sigma) + '.png')
plt.show()


# Problem 2.6 (first instance)

# Reconstruct

reconstruct_obj = reconstruct(x_s, 1/f_s, 1/f_ss)
x_r = reconstruct_obj.solve()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('QUESTION 2.6 PART A : Original signal and reconstructed signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal')
#axs[1].plot(t, x_r)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal')
plt.savefig('signal_and_reconstructed_time_'+ str(sigma) + '.png')
plt.show()

###############################################################################
############################ Q U E S T I O N 2.4 ##############################
###############################################################################

# SUBSAMPLE CLASS
# CALL SOLVE_PREFILTERING FUNCTION IN THE SUBSAMPLE CLASS
# SPECTRUM OF THE SIGNAL HAS A BANDWIDTH W THAT EXCEEDS V
# THEREFORE, WE CAN AVOID ALISING BY IMPLEMENTING A LOW PASS FILTER TO ELIMINATE FREQUENCIES ABOVE V BEFORE SUBSAMPLING
subsample_obj = subsample(x, 1/f_s, 1/f_ss)
x_s, x_delta = subsample_obj.solve_prefiltering()

DFT_x = dft(x, f_s)
[_, _, freqs_c, X_c] = DFT_x.solve_using_numpy_fft()

DFT_x_delta = dft(x_delta, f_ss)
[_, _, _, X_delta_c] = DFT_x_delta.solve_using_numpy_fft()

# DISPLAY GRAPH OF DFT OF ORIGINAL SIGNAL AND PRE FILTERED + SUBSAMPLED SIGNAL
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('QUESTION 2.4 AND 2.5: DFT of original signal and prefiltered + subsampled signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(freqs_c, abs(X_c))
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('DFT')
axs[1].plot(freqs_c, abs(X_delta_c))
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('DFT')
plt.savefig('signal_and_prefiltered_freq_'+ str(sigma) + '.png')
plt.show()

# Problem 2.6 (second instance)
# Reconstruct

reconstruct_obj = reconstruct(x_s, 1/f_s, 1/f_ss)
x_r = reconstruct_obj.solve()

# Plot
fig, axs = plt.subplots(2)
axs[0].grid()
axs[1].grid()
fig.suptitle('QUESTION: 2.6 PART B:  Original signal and reconstructed signal' )
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.6)
axs[0].plot(t, x)
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Signal')
axs[1].plot(t, x_r)
#axs[1].plot(t, x_r)
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Signal')
plt.savefig('signal_and_reconstructed_no_loss_time_'+ str(sigma) + '.png')
plt.show()