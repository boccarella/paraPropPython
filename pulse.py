import numpy as np
from scipy import signal
from matplotlib import pyplot as pl
class pulse:
    def __init__(self, A, f, tstart, tend, tsamp):
        self.amplitude = A
        self.frequency = f
        self.start = tstart
        self.end = tend
        self.sample_interval = tsamp
        self.centre = (tstart + tend) / 2.
        self.nSamples = int((self.end - self.start) / tsamp)
        self.time_space = np.linspace(self.start, self.end, self.nSamples)
        self.freq_space = np.fft.fftfreq(self.nSamples, self.sample_interval)
        self.real = np.zeros(self.nSamples)
        self.imag = np.zeros(self.nSamples)
        self.abs = np.zeros(self.nSamples)
        self.complex = self.abs + 1j * self.imag
        # self.time_space_ifft = np.linspace(-self.end/2, self.end/2, nSamples)

    def do_gaussian(self, tcentral):
        i, q, e = signal.gausspulse(self.time_space - tcentral, fc=self.frequency, retquad=True, retenv=True)
        self.real = i
        self.imag = q
        self.abs = e
        self.centre = tcentral

    def doFFT(self):
        fft_pulse = np.fft.fft(self.real + 1j * self.imag)
        # fft_pulse = np.fft.fft(self.real)
        # fft_pulse = np.fft.fft(self.abs)
        return fft_pulse

    def doIFFT(self, fft_pulse):  # NOTE THIS FUNCTION CREATES A NEW PULSE FROM AN OLD PULSe
        ifft_pulse = np.fft.ifft(fft_pulse)
        self.real = ifft_pulse.real
        self.imag = ifft_pulse.imag
        self.abs = abs(ifft_pulse)

    def get_from_array(self, array):
        self.real = array.real
        self.imag = array.imag
        self.abs = abs(array)

frequency_central = 200e6 #Frequency of Transmitter Pulse
t_end = 800e-9 #Time length of pulse [s]
t_start = 0 #starting time
t_samples = 1e-9 #Sample Interval
amplitude0 = 1 #TX Amplitude

t_pulse = 50e-9 #Time of Pulse centre

tx_pulse = pulse(amplitude0, frequency_central, t_start, t_end, t_samples) #Start TX Pulse object

tx_pulse.do_gaussian(t_pulse) #Set TX pulse to gaussian pulse

pl.plot(tx_pulse.time_space, tx_pulse.real)
pl.show()