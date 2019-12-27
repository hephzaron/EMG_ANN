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


## Difference of Gaussians (DoG)
# (approximation of Laplacian of Gaussian)

# define sigmas
sPos = .1
sNeg = .5

# create the two GAussians
gaus1 = np.exp( (-timevec**2) / (2*sPos**2) ) / (sPos*np.sqrt(2*np.pi))
gaus2 = np.exp( (-timevec**2) / (2*sNeg**2) ) / (sNeg*np.sqrt(2*np.pi))

# their difference is the DoG
DoG = gaus1 - gaus2


# amplitude spectrum
DoGPow = np.abs(fft(DoG)/npnts)


# time-domain plotting
plt.subplot(211)
plt.plot(timevec,DoG,'k')
plt.xlabel('Time (sec.)')
plt.title('DoG wavelet in time domain')

# frequency-domain plotting
plt.subplot(212)
plt.plot(hz,DoGPow[:len(hz)],'k')
plt.xlim([0,freq*3])
plt.xlabel('Frequency (Hz)')
plt.title('DoG wavelet in frequency domain')
plt.show()