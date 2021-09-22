# paraPropPython example use of paraPropPython.py notebook
# s. prohira, c. sbrocco

import paraPropPython as ppp
import numpy as np
import matplotlib.pyplot as pl
from paraPropPython import receiver as rc
from pulse import *

import util
import matplotlib.pyplot as plt
import numpy as np
import scipy as sci
import scipy.constants as constant
import scipy.io.wavfile as wav
import scipy.signal as sig
import scipy.interpolate as interp
from scipy.signal import butter, lfilter
from numpy import linalg as la
import pickle
import csv
from numpy.lib.format import open_memmap
import os
import sys
import time
from datetime import datetime
import scipy.signal as signal
import paraProp_alex as prop_alex
def touch(fname, times=None):
    with open(fname, 'a'):
        os.utime(fname, times)

def create_memmap(fname, path_to_directory, shape0, dtype0 = 'float32', mode0 ='w+'): #Create a Blank Memmap for recording data
    full_file = path_to_directory + fname
    A = open_memmap(full_file, dtype=dtype0, mode=mode0, shape=shape0)
    return A
#pulse_file = path_to_directory + simul_directory + 'pulse_input.npy'
#np.save(pulse_file, pulse_input)
### first, initialize an instance of paraProp by defining its dimensions and frequency of interest ###


'''
iceDepth = 150. # m
iceLength = 220. # m
dx = 1 # m
dz = 0.05 # m

### it is useful to set the reference depth as the source depth when you only have one transmitter ###
sourceDepth = 50. # m
sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)



frequency_central = 200e6 #Frequency of Transmitter Pulse
t_end = 800e-9 #Time length of pulse [s]
t_start = 0 #starting time
t_samples = 1e-9 #Sample Interval
amplitude0 = 1 #TX Amplitude

t_pulse = 50e-9 #Time of Pulse centre

#Define Receiver Positions:
max_depth_tx = 50.
max_depth_rx = 50.
step_interval = 5.
nReceivers = int(max_depth / step_interval) + 1
nTransmitters = nReceivers
sourceDepth_list = np.linspace(0, max_depth_tx, nReceivers)
receiverDepth_list = np.linspace(0, max_depth_rx, nTransmitters)

#Range List #Change Range of Receivers
Range_list = np.array( [10, 20, 30, 50, 80, 100, 120, 150, 200] )
nRanges = len(Range_list)

tx_pulse = pulse(amplitude0, frequency_central, t_start, t_end, t_samples) #Start TX Pulse object
tx_pulse.do_gaussian(t_pulse) #Set TX pulse to gaussian pulse
nSamples = tx_pulse.nSamples

#Decompose Pulse in Frequencies:
fft_pulse = tx_pulse.doFFT()
nFreq = len(tx_pulse.freq_space)

### useful arrays for plotting ###
z = sim.get_z()
x = sim.get_x()

path_to_simul = path_to_directory + simul_directory

if not os.path.exists(path_to_directory + simul_directory):
    os.mkdir(path_to_directory + simul_directory)
##### steady state example #####

#now = datetime.now()
#date_str = now.strftime('%Y.%m.%d.%H:%M:%S')
info_file = path_to_directory + simul_directory + 'simul_info.txt'
fout = open(info_file, 'w+')
fout.write(site+'\t#site\n')
fout.write(method + '\t#method\n')
fout.write(str(polarization) + '\t#polarization\n')
fout.write(nProfile + '\tnProfile\n')
fout.write(str(amplitude_pulse) + '\t#Amplitude\n')
fout.write(str(frequency_central/1e6) + '\t#Central-Frequency-MHz\n')
fout.write(str(xMax) + '\t#xMax-m\n')
fout.write(str(zMax) + '\t#zMax-m\n')
fout.write(str(min_depth) + '\t#min-depth\n')
fout.write(str(max_depth) + '\t#max-depth\n')
fout.write(str(x_rx) + '\t#horizontal-seperation\n')
fout.write(str(depthStep) + '\t#depth-interval\n')
fout.write(str(nSamples) + '\t#Number-Samples\n')
fout.write(str(sample_interval) + '\t#Sampling-interval-s\n')
fout.write(str(recording_time) + '\t#recording-time-s\n')
fout.write(str(nyquist_frequency)+'\t#nyquist-freauency-Hz\n')

now = datetime.now()
date_str = now.strftime('%Y.%m.%d.%H:%M:%S')
fout.write(date_str + '\t#datetime\n')
fout.close()

data_list_file = path_to_directory + simul_directory + 'data_list.txt'
fout2 = open(data_list_file, 'a')

#local_directory = os.getcwd()
#path_to_directory = '/media/alex/Elements/'
path_to_directory = os.getcwd() + '/'
simul_directory = 'output/'

for k in range(nTransmitters): #Loop over the transmitters in the simulation
    sourceDepth = sourceDepth_list[k]
    ascan_data_rx_file = 'ascan_pulse_rx_depth_tx=' + str(sourceDepth) + '.npy'
    ascan_profile_rx = create_memmap(ascan_data_rx_file, path_to_simul, shape0=(nRanges, nReceivers, nFreq))

    ascan_data_tx_file = 'ascan_pulse_tx_depth_tx=' + str(sourceDepth) + '.npy'
    ascan_profile_tx = create_memmap(ascan_data_tx_file, path_to_simul, shape0=nFreq)
    for i in range(1, nFreq):
        amplitude_fft = fft_pulse[i]
        frequency_i = abs(tx_pulse.freq_space[i])
        print(i, round(frequency_i / 1e6, 2), 'MHz', amplitude_fft)

        sim.set_dipole_source_profile(frequency_i / 1e9, sourceDepth) #Set the dipole source

        sim.set_cw_source_signal(frequency_i / 1e9) #Set the frequency

        sim.A = np.array([amplitude_fft], dtype='complex') #set the amplitude
        sim.do_solver() #Run the simulation

        #signal_tx = sim.get_field(TX.x, TX.z)
        #signal_rx = sim.get_field(RX.x, RX.z)
        for j in range(nReceivers): #Scan over each receiver postion -> record amplitude at each receiver location
            for l in range(nRanges):
                receiver_depth = receiverDepth_list[j]
                Range = Range_list[l]
                RX_j = rc(Range, receiver_depth)
                signal_rx = sim.get_field(RX_j.x, RX_j.z) #Amplitude value at each point
                ascan_profile_rx[l, j, i] = signal_rx
'''

def generate_ascan(pulse_tx, tx_array, rx_array, path_to_file, file_name): #Creates an Array of Scans for a set of receivers and transmitters
    nRanges_rx = len(rx_array)
    nDepths_rx = len(rx_array[0])
    nDepths_tx = len(tx_array)

    freq_space = pulse_tx.freq_space
    nFreqs = len(freq_space)

    fft_pulse = pulse_tx.doFFT()

    ascan_array = create_memmap(file_name, path_to_file, shape0 = (nFreqs, nDepths_tx, nRanges_rx, nDepths_tx), dtype0=complex)
    for i in range(freq_space):
        print(i, int(freq_space[i]/1e6), 'MHz')
        freqquency_i = freq_space[i]
        amplitude_fft = fft_pulse[i]

        for j in range(nDepths_tx):
            sourceDepth = tx_array[j]
            sim.set_dipole_source_profile(frequency_i / 1e9, sourceDepth, amplitude_fft[i])  # Set the dipole source
            sim.set_cw_source_signal(frequency_i / 1e9)  # Set the frequency
            sim.do_solver()  # Run the simulation

            for k in range(nRanges_rx):
                for l in range(nDepths_rx):
                    RX = rc(rx_array[l,k][0], rx_array[l,k][1]) #Get Receiver position (range and depth)
                    signal_rx = sim.get_field(RX.x, RX.z)
                    ascan_array[i, j, k, l] = signal_rx

    np.save(rx_array)
    np.save(tx_array)
    np.save(freq_space)

    return ascan_array, tx_array, rx_array