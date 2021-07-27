# paraPropPython
# c. sbrocco, s. prohira
#Additions made by Alex Kyriacou

import util
import numpy as np
import paraPropPython as ppp
from paraPropPython import receiver as rx
import scipy.signal as signal
import util
from backwardsSolver import backwards_solver, backwards_solver_stationary
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

    n_material = n_air
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
iceDepth = 100. # m
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
    for j in range(nDepths):
        rxList.append(rx(range_list[i], depth_list[j]))


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

#fout_settings = open('settings.txt', 'w+')


#Set Simulation
sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)
sim3 = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)

### useful arrays for plotting ###
z = sim.get_z()
x = sim.get_x()

#Permittivity Profile
sim.set_n2('func', nFunc=pure_ice)
sim3.set_n2('func', nFunc=pure_ice)

#TODO: Save profile

#Set source
sim.set_dipole_source_profile(freq, sourceDepth)
sim3.set_dipole_source_profile(freq, sourceDepth)

#TODO Save source

def sim_settings(file, sim0):
    fout_settings = open(file, 'w+')
    fout_settings.write('#iceLength' + '\t' + str(sim0.iceLength)) + '\n'
    fout_settings.write('#iceDepth' + '\t' + str(sim0.iceDepth)) + '\n'
    fout_settings.write('#dx' + '\t' + str(sim0.dx)) + '\n'
    fout_settings.write('#dz' + '\t' + str(sim0.dz)) + '\n'
    fout_settings.write('#refDepth' + '\t' + str(refDepth))

    for line in fout_settings:
        print(line)
    fout_settings.close()

#Set Continous Wave Signal
sim.set_cw_source_signal(freq)
sim3.set_cw_source_signal(freq)

#sim3 = sim

sim2, sim_refl = backwards_solver_stationary(sim)
sim3.do_solver2()

### plot absolute value of field for whole simulation space ###
fig = pl.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111)

pmesh = pl.imshow(np.transpose(abs(sim3.get_field())), aspect='auto', cmap='hot',  vmin=1e-5, vmax=1e-2,
          extent=(x[0], x[-1], z[-1], z[0]))
cbar = pl.colorbar(pmesh)
pl.title("Forward Propagating Field $u_{+}(x,z)$, f = " + str(int(freq*1000))+" MHz")
pl.xlabel("x (m)")
pl.ylabel("z (m)")
pl.show()


### plot absolute value of field for whole simulation space ###
fig = pl.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111)

pmesh = pl.imshow(np.transpose(abs(sim2.get_field())), aspect='auto', cmap='hot',  vmin=1e-5, vmax=1e-2,
          extent=(x[0], x[-1], z[-1], z[0]))
cbar = pl.colorbar(pmesh)
pl.title("Combined field $u_{+}(x,z) + u_{-}(x,z)$, f = " + str(int(freq*1000))+" MHz")
pl.xlabel("x (m)")
pl.ylabel("z (m)")
pl.show()

### plot absolute value of field for whole simulation space ###

fig = pl.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111)

pmesh = pl.imshow(np.transpose(abs(sim_refl.get_field())), aspect='auto', cmap='hot',
          extent=(x[0], x[-1], z[-1], z[0]))
pl.title("Backwards Field, $u_{-}(x,z)$, f = " + str(int(freq*1000))+" MHz")
cbar = pl.colorbar(pmesh)
pl.xlabel("x (m)")
pl.ylabel("z (m)")
pl.show()

### plot absolute value of field for whole simulation space ###
fig = pl.figure(figsize=(10,10), dpi=100)
ax = fig.add_subplot(111)

pmesh = pl.imshow(10*np.transpose(np.log10(abs(sim_refl.get_field()))), aspect='auto', cmap='hot',
          extent=(x[0], x[-1], z[-1], z[0]))
pl.title("Absolute Field, " + str(int(freq*1000))+" MHz")
cbar = pl.colorbar(pmesh)
pl.xlabel("x (m)")
pl.ylabel("z (m)")
pl.show()