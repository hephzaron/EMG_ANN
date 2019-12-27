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

## Mexican hat wavelet

# the wavelet
s = .4
MexicanWavelet = (2/(np.sqrt(3*s)*np.pi**.25)) * (1- (timevec**2)/(s**2) ) * np.exp( (-timevec**2)/(2*s**2) )

# amplitude spectrum
MexicanPow = np.abs(fft(MexicanWavelet)/npnts)


# time-domain plotting
plt.subplot(211)
plt.plot(timevec,MexicanWavelet,'k')
plt.xlabel('Time (sec.)')
plt.title('Mexican wavelet in time domain')

# frequency-domain plotting
plt.subplot(212)
plt.plot(hz,MexicanPow[:len(hz)],'k')
plt.xlim([0,freq*3])
plt.xlabel('Frequency (Hz)')
plt.title('Mexican wavelet in frequency domain')
plt.show()