import numpy as np
from permittivity import *
from geometry import triangle, sphere
import paraPropPython as ppp
from paraPropPython import receiver as rx
import util
from backwardsSolver import backwards_solver
from matplotlib import pyplot as pl

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import h5py
import sys
import datetime
import os

#Create Output File TODO: Have an input file

if len(sys.argv) < 2:
    print('error: should have format: python enceladus-environment.py <output_file>')

fout = str(sys.argv[1])
#Check if File Exists
if os.path.isfile(fout) == True:
    print('Warning, you are about to overwrite an existing file: ', fout, '\nProceed? [yes or no]')
    yes = {'yes', 'y'}
    no = {'no', 'n'}
    choice = input().lower()
    if choice is yes:
        print('Proceed')
        os.remove(fout)
    elif choice is no:
        print('Abort')
        sys.exit()
    else:
        print("Error, please start again and enter (yes / y) or (no / n)")
        sys.exit()

output_hdf = h5py.File(fout)


sigma_solid_pure_ice = 8.7e-6 #uS/m
P_surface = 0.9
sigma_pure_snow = sigma_from_p(P_surface, sigma_solid_pure_ice)
print(sigma_pure_snow)

freq_test = 500e6 #100 MHz
eps_i_ice = cond2eps_im(sigma_solid_pure_ice, freq_test)
print(eps_i_ice)

eps_snow = 1.2 + 0.015j #Permittivity of Enceladus Snow (Type III)
m_snow = eps2m(eps_snow) #

eps_ice = 3.2 + 1j*eps_i_ice
m_ice = eps2m(eps_ice)
print('Snow, eps_r =', eps_snow, 'n = ', m_snow, 'alpha = ', alpha(eps_snow, freq_test))
print('Ice, eps_r =', eps_ice, 'n =', m_ice, 'alpha = ', alpha(eps_ice, freq_test))

eps_meteor = 8.2 + 0.1558j #Taken from Herique et al. (2018) Direct Observations of Asteroid Interior and Regolith Structure: Science Measurement Requirements
eps_vacuum = 1.0
eps_water = 82 + 899j #Based on the Conductivity of Salt Water -> 5 S/m

def enceladus_2layer(z, snow_depth=100): #Create a flat layer of snow above ice
    n_snow = eps2m(eps_snow)
    n_ice = eps2m(eps_ice)
    n_vacuum = 1.0
    n_material = n_vacuum

    if z >= 0 and z < snow_depth:
        n_material = n_snow
    elif z >= snow_depth:
        n_material = n_ice
    return n_material

#TODO: modify refractive index profile of code to be frequency dependent -> account for frequency dependent permittivity and conductivity

def enceladus_environ(x, z, snow_depth = 100, meteor_list = [], crevass_list = [], aquifer_list=[]): #Creates a 2 layer geometry with added meteorites (spheres), crevasses and aquifers (triangles)
    n_medium = enceladus_2layer(z, snow_depth)

    numMeteors = len(meteor_list)
    numCrevasses = len(crevass_list)
    numAquifers = len(aquifer_list)
    #Loop over meteorites


    for i in range(numMeteors):
        meteor_i = meteor_list[i] #must be sphere
        if meteor_i.isInside(x,z) == True:
            n_medium = eps2m(meteor_i.eps_r)

    for i in range(numCrevasses):
        crevass_i = crevass_list[i]
        if crevass_i.isInside(x,z) == True:
            n_medium = eps2m(crevass_i.eps_r)

    for i in range(numAquifers):
        aquifer_i = aquifer_list[i]
        if aquifer_i.isInside(x,z) == True:
            n_medium = eps2m(aquifer_i.eps_r)

    return n_medium


#meteor1 = sphere(20, 30, 10, eps_meteor)
#crevass1 = triangle(Point(40,0), Point(60,0), Point(50,40), eps_vacuum)
aquifer1 = triangle(Point(480, 400), Point(500,500), Point(520,500), eps_water)

aquifer_list1 = []
aquifer_list1.append(aquifer1)
snow_depth1 = 20.
'''
def nEnceladus(x,z):
    return enceladus_environ(x,z, snow_depth= 20, meteor_list=[meteor1], crevass_list=[crevass1], aquifer_list=[aquifer1])
'''
def nEnceladus(x,z):
    return enceladus_environ(x,z, snow_depth=snow_depth1, aquifer_list=aquifer_list1)

iceLength = 1000
iceDepth = 500
dx = 1
dz = 0.5
sourceDepth = 100
freq_centre = 0.5

sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth, airHeight = 30)
z = sim.get_z()
x = sim.get_x()

method1 = 'func'
sim.set_n2(method1, nFunc=nEnceladus)
sim.set_dipole_source_profile(freq_centre, sourceDepth)

#Set TD signal -> Pulse
sample_frequency = 2. #Sample Frequency [GHz]
dt = 1/sample_frequency #Time Interval [ns]
nyquist_frequency = sample_frequency/2 #Nyquist Frequency [GHz]
sampling_time = 1000. #Total Sampling Time -> time of recording [ns]
freq_min = 0.4
freq_max = 0.6
band = freq_max - freq_min

tspace = np.arange(0, sampling_time, dt)
tcentral = 50.

Signal_Amplitude = 1000 + 0j
impulse = util.make_pulse(tspace, tcentral, 1. + 0j, freq_centre)
sig = util.normToMax(util.butterBandpassFilter(impulse, freq_min, freq_max, sample_frequency, 4))
#Todo -> save pulse and spectrum

#When we test the TD
#Set Pulse
sim.set_td_source_signal(sig, dt)

freq_space = sim.get_frequency()
tx_spectrum = sim.get_spectrum()

#Add Transmitters:
z_start = -25.
z_end = 500
z_step = 10.

x_start = 0.
x_end = 1000
x_step = 50.
zTX = np.arange(z_start, z_end, z_step)

#Add Receivers
rxList = []
zRX = np.arange(z_start, z_end, z_step)
xRX = np.arange(x_start, x_end, x_step)

for i in range(len(xRX)):
    for j in range(len(zRX)):
        rx_ij = rx(xRX[i], zRX[j])
        rxList.append((rx_ij))
nRX = len(rxList)

def print_var_name(variable):
 for name in globals():
     if eval(name) is variable:
        print(name)


'''
x = np.arange(0, 100, dx)
z = np.arange(-30, 200, dz)
nX = len(x)
nZ = len(z)

n_prof2 = np.zeros(shape=(nX, nZ), dtype='complex')
print(nFunc(50,0))
print(nFunc(50, 150))
print(nFunc(20, 30))

for i in range(nX):
    for j in range(nZ):
        print(i,j)
        n_prof2[i,j] = nFunc(x[i],z[j])

fig = pl.figure(figsize=(10,6),dpi=100)
ax = fig.add_subplot(111)
ax.set_title('n2 profile')

#Plot without colorbar limits
pmesh = pl.imshow(np.transpose(np.log10(n_prof2.real)), extent=(0, 100, -200, 30), aspect='auto')
cbar = pl.colorbar(pmesh)
cbar.set_label("$\log{\epsilon_{r}^{'}} $ (real)")
ax.set_xlabel('Range [m]')
ax.set_ylabel('Depth [m]')
pl.show()

for i in range(len(x)):
    for j in range(len(z)):
        print(i,j)
        n_prof2[i,j] = nFunc(x[i],z[i])

print(n_prof2)
pl.figure(figsize = (10,10),dpi=100)
pmesh = pl.imshow(n_prof2.real, aspect='auto', extent=(x[0], x[-1], z[-1], z[0]))
bar = pl.colorbar(pmesh)
bar.set_label('n')
pl.ylabel('Depth z [m]')
pl.xlabel('Range x [m]')
pl.show()
'''