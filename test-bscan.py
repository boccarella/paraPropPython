# paraPropPython
# c. sbrocco, s. prohira
#Additions made by Alex Kyriacou

import util
import numpy as np
import paraPropPython as ppp
from paraPropPython import receiver as rx
import scipy.signal as signal
import util
from backwardsSolver import backwards_solver
from matplotlib import pyplot as pl
import time


#Set up simulation
#-> Add Pure Ice + Water Pocket

#Set up profile function TODO: Move these functions to another file

#Set Water Pocket
X_sphere = 50
Z_sphere = 50
R_sphere = 10

def pure_ice(x, z, x_sph = X_sphere, z_sph = Z_sphere, r_sph = R_sphere):
    dr_sq = (x - x_sph)**2 + (z - z_sph)**2
    dr = np.sqrt(dr_sq)

    n_water = 72 + 2j
    n_ice = 1.82 + 1e-5j
    n_air = 1.00003
    n_material = n_ice
    if dr <= r_sph:
        n_material = n_water
    else:
        if z >= 0:
            n_material = n_ice
        else:
            n_material = n_air
    return n_material

def make_pulse(t, t_central, amplitude, freq_central):
    sig = amplitude * signal.gausspulse(t - t_central, freq_central)
    return sig
from numpy.lib.format import open_memmap

def create_memmap(file, dimensions, data_type ='complex'):
    A = open_memmap(file, shape = dimensions, mode='w+', dtype = data_type)
    return A

def print2hms(time_s):
    hrs = int(time_s / 3600.)
    mins = int((time_s % 60)/60.)
    secs = (time_s % 3600)
    print(str(hrs) + ' h, ' + str(mins) + ' m, ' + str(secs) + ' s')
##### steady state example #####

### first, initialize an instance of paraProp by defining its dimensions and frequency of interest ###
iceDepth = 200. # m
iceLength = 100. # m
dx = 1 # m
dz = 0.05 # m

freq = 0.3 #GHz

sample_frequency = 2. #Sample Frequency [GHz]
dt = 1/sample_frequency #Time Interval [ns]
nyquist_frequency = sample_frequency/2 #Nyquist Frequency [GHz]
sampling_time = 1000. #Total Sampling Time -> time of recording [ns]

tspace = np.arange(0, sampling_time, dt)
tcentral = 50.

freq_min = 0.2
freq_max = 0.4

def sim_settings(file, sim0):
    fout_settings = open(file, 'w+')
    fout_settings.write('#iceLength' + '\t' + str(sim0.iceLength) + '\n')
    fout_settings.write('#iceDepth' + '\t' + str(sim0.iceDepth) + '\n')
    fout_settings.write('#dx' + '\t' + str(sim0.dx) + '\n')
    fout_settings.write('#dz' + '\t' + str(sim0.dz) + '\n')
    fout_settings.write('#refDepth' + '\t' + str(sim.refDepth))

    for line in fout_settings:
        print(line)
    fout_settings.close()


#Set Receivers
#Place Receivers at different ranges and depths
min_range = 1
max_range = 100.
range_step = 2.
range_list = np.arange(min_range, max_range, range_step)
nRanges = len(range_list)

min_depth = -20.
max_depth = 100.
depth_step = 2.
depth_list = np.arange(min_depth, max_depth, depth_step)
nDepths = len(depth_list)
reference_depth = 20.

rxList = []
for i in range(nRanges):
    rxList.append(rx(range_list[i], reference_depth))
for i in range(nDepths):
    rxList.append(rx(range_list[0], depth_list[i]))

def print_receiver(rx):
    return str(rx.x) + '\t' + str(rx.z)

fout_receivers = open('bscans/receiverList.txt', 'w+')
for j in range(len(rxList)):
    rx_j = rxList[j]
    line = str(j) + '\t' + print_receiver(rx_j)
    if j + 1 < len(rxList):
        line += '\n'
    fout_receivers.write(line)
fout_receivers.close()
rxList = np.array(rxList)

#Transmitter List
txList = depth_list

#Create Pulse
#Set TD signal -> Pulse
impulse = make_pulse(tspace, tcentral, 1. + 0j, freq)
sig = util.normToMax(util.butterBandpassFilter(impulse, freq_min, freq_max, sample_frequency, 4))
#Todo -> save pulse and spectrum
#When we test the TD
#Set Pulse


nSamples = len(sig)

nRX = len(rxList)
#Run a list of simulations
bscan_array = create_memmap('bscans/bscan-pure-ice-water-pocket.npy', dimensions=(nDepths, nRX, nSamples), data_type='complex')
np.save('bscans/tspace.npy', tspace)

for i in range(nDepths):
    sourceDepth = txList[i]
    print('Run Simulation, TX.z = ', sourceDepth)
    # Set Simulation
    sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=reference_depth)

    ### useful arrays for plotting ###
    z = sim.get_z()
    x = sim.get_x()

    # Permittivity Profile
    sim.set_n2('func', nFunc=pure_ice)
    # TODO: Save profile

    # Set source

    sim.set_dipole_source_profile(freq, sourceDepth)
    sim.set_td_source_signal(sig, dt)
    freq_space = sim.get_frequency()
    if i == 0:
        np.save('bscans/freq_space.npy', freq_space)

    tx_spectrum = sim.get_spectrum()
    if i == 0:
        np.save('bscans/tx_spectrum.npy', tx_spectrum)
        sim_settings('bscans/simulation-settings.txt', sim)

    tstart = time.time()
    rx_pulses = backwards_solver(sim, rxList, freq_min, freq_max, 0.1)
    tend = time.time()

    simul_time = tend - tstart
    print('Simulation time:', simul_time)
    remaining_time = (nDepths - i) * simul_time
    print2hms(remaining_time)

    for j in range(nRX):
        rx = rx_pulses[j]
        bscan_array[i][j] = rx.spectrum
