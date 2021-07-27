# paraPropPython
# c. sbrocco, s. prohira
#Additions made by Alex Kyriacou

import numpy as np
import paraPropPython as ppp
from paraPropPython import receiver as rx
import scipy.signal as signal
import util
from backwardsSolver import backwards_solver
from matplotlib import pyplot as pl


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

### it is useful to set the reference depth as the source depth when you only have one transmitter ###
sourceDepth = 50. # m

#Set Receivers
#Place Receivers at different ranges and depths
min_range = 1
max_range = 100.
range_step = 2.
range_list = np.arange(min_range, max_range, range_step)
nRanges = len(range_list)

min_depth = 1.
max_depth = 100.
depth_step = 1.
depth_list = np.arange(min_depth, max_depth, 2.)
nDepths = len(depth_list)

rxList = []
for i in range(nRanges):
    rxList.append(rx(range_list[i], -10))
for i in range(nDepths):
    rxList.append(rx(range_list[0], depth_list[i]))

def print_receiver(rx):
    return str(rx.x) + '\t' + str(rx.z)

fout_receivers = open('receiverList.txt', 'w+')
for j in range(len(rxList)):
    rx_j = rxList[j]
    line = str(j) + '\t' + print_receiver(rx_j)
    if j + 1 < len(rxList):
        line += '\n'
    fout_receivers.write(line)
fout_receivers.close()
rxList = np.array(rxList)

#Set Simulation
sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)

### useful arrays for plotting ###
z = sim.get_z()
x = sim.get_x()

#Permittivity Profile
sim.set_n2('func', nFunc=pure_ice)
#TODO: Save profile

#Set source
sim.set_dipole_source_profile(freq, sourceDepth)
#TODO Save source

def sim_settings(file, sim0):
    fout_settings = open(file, 'w+')
    fout_settings.write('#iceLength' + '\t' + str(sim0.iceLength) + '\n')
    fout_settings.write('#iceDepth' + '\t' + str(sim0.iceDepth) + '\n')
    fout_settings.write('#dx' + '\t' + str(sim0.dx) + '\n')
    fout_settings.write('#dz' + '\t' + str(sim0.dz) + '\n')
    fout_settings.write('#refDepth' + '\t' + str(sim0.refDepth))

    for line in fout_settings:
        print(line)
    fout_settings.close()
sim_settings('simulation-settings.txt', sim)

#Set TD signal -> Pulse
impulse = make_pulse(tspace, tcentral, 1. + 0j, freq)
sig = util.normToMax(util.butterBandpassFilter(impulse, freq_min, freq_max, sample_frequency, 4))
#Todo -> save pulse and spectrum

#When we test the TD
#Set Pulse
sim.set_td_source_signal(sig, dt)

freq_space = sim.get_frequency()
tx_spectrum = sim.get_spectrum()

#Set Continous Wave Signal
'''
Test Pulse
'''


fig = pl.figure(figsize=(10,20), dpi = 100)
ax1 = fig.add_subplot(211)
ax1.plot(tspace, sig.real)
ax1.set_xlabel('Time [ns]')
ax1.grid()

ax2 = fig.add_subplot(212)
ax2.plot(freq_space, abs(tx_spectrum))
ax2.set_xlabel('Frequency [GHz]')
ax2.grid()
fig.savefig('transmitter-pulse-and-spectrum.png')
pl.close()

nRx = len(rxList)

rx_pulses = backwards_solver(sim, rxList, freq_min, freq_max, 0.1)

rx_spec= rx_pulses[0].spectrum
nSamples = len(rx_spec)
rx_spectra = create_memmap('rx_spectra2.npy', dimensions=(nRx, nSamples))


for i in range(nRx):
    rx = rx_pulses[i]
    rx_spectra[i] = rx.spectrum
#TODO: save pulses properly
np.save('tspace.npy', tspace)
np.save('freq_space.npy', freq_space)


'''
np.save('tx_spectrum,npy', tx_spectrum)
np.save('rx_spectrum.npy', rx_pulses)
'''

