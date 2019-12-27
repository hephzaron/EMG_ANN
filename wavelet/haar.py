import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fft2, ifft
import scipy.io as sio
import copy

## general simulation parameters

fs = 1024
npnts = fs*5 # 5 seconds# parameters
freq = 4 # peak frequency

# centered time vector
timevec = np.arange(0,npnts)/fs
timevec = timevec - np.mean(timevec)

# for power spectrum
hz = np.linspace(0,fs/2,int(np.floor(npnts/2)+1))
# create Haar wavelet
HaarWavelet = np.zeros(npnts)
HaarWavelet[np.argmin(timevec**2) : np.argmin((timevec-.5)**2) ] = 1
HaarWavelet[np.argmin((timevec-.5)**2) : np.argmin((timevec-1-1/fs)**2)] = -1

# amplitude spectrum
HaarWaveletPow = np.abs(fft(HaarWavelet)/npnts)


# time-domain plotting
plt.subplot(211)
plt.plot(timevec,HaarWavelet,'k')
plt.xlabel('Time (sec.)')
plt.title('Haar wavelet in time domain')

# frequency-domain plotting
plt.subplot(212)
plt.plot(hz,HaarWaveletPow[:len(hz)],'k')
plt.xlim([0,freq*3])
plt.xlabel('Frequency (Hz)')
plt.title('Haar wavelet in frequency domain')
plt.show()