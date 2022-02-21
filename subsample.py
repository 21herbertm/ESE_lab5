# -*- coding: utf-8 -*-
"""
ESE 224

MELANIE HERBERT, ALINA HO

SUB-SAMPLING FUNCTION FOR QUESTIONS 2.2 ONWARDS
"""

import numpy as np
from dft import dft
from idft import idft

class subsample(object):

# RETURNS THE SUBSAMPLED SIGNAL AND DELTA TRAIN REPRESENTATION
    def __init__(self, x, T_s, tau):
        
        self.x = x # INPUT SIGNAL
        self.T_s = T_s # SAMPLING TIME
        self.f_s = np.int(1/T_s)
        self.tau = tau
        self.f_ss = np.int(1/tau)
        self.N = len(x)
        
    # NO PREFILTERING FOR 2.3
    def solve(self):

        step = np.int(self.tau/self.T_s)
        x_s = self.x[0::step]
        x_delta = np.zeros(self.N)
        x_delta[0::step] = x_s
        
        return x_s, x_delta
        
    # WITH PREFILTERING MODIFICATION FOR PROBLEM 2.4 AND 2.5
    def solve_prefiltering(self):
        
        # LOW PASS FILTERING TO ELIMINATE FREQUENCIES ABOVE V BEFORE SUBSAMPLING
        fmax = self.f_ss/2
        DFT = dft(self.x, self.f_s)
        [_, _, f_c, X_c] = DFT.solve_using_numpy_fft()
        index_min  = np.min(np.where(f_c >= -fmax)[0])
        index_max = np.max(np.where(f_c <= fmax)[0])
        X_band = np.zeros(self.N)
        X_band[index_min:index_max] = X_c[index_min:index_max] # PREV COMMENTED
        # X_BAND NEEDS TO BE FILLED WITH VALUES NOT 0 USING INDEX_MIN AND INDEX_MAX TO GET A SUBSAMPLE
        X_band = np.roll(X_band, np.int(np.floor(self.N/2+1)))
        iDFT = idft(X_band, self.f_s, self.N)
        x_band, t = iDFT.solve_ifft()


        #SLICE WILL GET A PORTION OF A LIST FROM START END STEP, WHERE STEP IS WHAT TO SKIP IN THE LIST
        #MYLIST[START:END:STEP]
        # Subsample
        #GET THE STEP, TAU/SAMPLE TIME
        step = np.int(self.tau/self.T_s)
        #SLICE OF THE X_BAND LIST
        x_s = x_band[0::step]
        x_delta = np.zeros(self.N)
        '''
         The double colon is a special case in Python's extended slicing feature.
          The extended slicing notation string[start:stop:step] uses three arguments start , stop , and step to carve
           out a subsequence.
          It accesses every step -th element between indices start (included) and stop (excluded).
          '''
        x_delta[0::step] = x_s # PREV COMMENTED
        
        return x_s, x_delta
        
        
        