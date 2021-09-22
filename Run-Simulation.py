#!/usr/bin/env python
# coding: utf-8

# In[1]:


# paraPropPython example use of paraPropPython.py notebook
# s. prohira, c. sbrocco

#get_ipython().run_line_magic('matplotlib', 'inline')
import paraPropPython as ppp
import numpy as np
import matplotlib.pyplot as pl
import scipy
from scipy import signal


# In[2]:


from paralell_scan import *
from pulse import *


# In[3]:


### an example of defining n as a function of z (also can be done using a vector, see implementation) ###
def southpole(z):
    A=1.78
    B=-0.43
    C=-0.0132
    return A+B*np.exp(C*z)

def enceladus_pole(z, snow_depth = 10, firn_depth = 20, ice_depth = 100):
    n_ice = 1.82
    n_firn = np.sqrt(2.2)
    n_snow = np.sqrt(1.2)
    n_water = np.sqrt(82.)
    n0 = 1.0003
    
    nz = n_water
    if z < 0:
        nz = n0
    elif z >= 0 and z < snow_depth:
        nz = n_snow
    elif z >= snow_depth and z < firn_depth:
        nz = n_firn
    elif z >= firn_depth and z < ice_depth:
        nz = n_ice
    return nz

def vacuum(z):
    if z > 0:
        nz = 1.0003
    return 1.0003


def pure_ice(z):
    epsilon_ice_1 = 3.2
    epsilon_ice_2 = 1e-5
    nz = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2)
    return nz 

def pure_ice_air(z):
    epsilon_ice_1 = 3.2
    epsilon_ice_2 = 1e-5
    n_ice = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2)
    n_vac = 1
    
    if z < 0: 
        nz = n_vac
    elif z >= 0:
        nz = n_ice
    return nz


def pure_ice_water_pocket(z, ice_depth_1=50, water_pocket_depth=150, ice_depth_2=200):
    epsilon_ice_1 = 3.2
    epsilon_ice_2 = 1e-5
    epsilon_water_1 = 74
    epsilon_water_2 = 1.8
    n_water = np.sqrt(82.)
    n0 = 1
    nz = n_water
    if z < 0:
        nz = n0
    elif z >= 0 and z < ice_depth_1:
        nz = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2)
    elif z >= ice_depth_1 and z < water_pocket_depth:
        nz = np.sqrt(epsilon_water_1 + 1j * epsilon_water_2)
    elif z >= water_pocket_depth and z < ice_depth_2:
        nz = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2)
    return nz

def enceladus_reference(z, snow_depth = 25):
    epsilon_ice_1 = 3.19
    epsilon_ice_2 = 1e-5
    n_ice = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2) 
    epsilon_snow_1 = 1.2
    epsilon_snow_2 = 0.139
    n_snow = np.sqrt(epsilon_snow_1 + 1j * epsilon_snow_2)
    n0 = 1 
    
    
    if z < 0:
        nz = n0
    elif z >= 0 and z < snow_depth:
        nz = n_snow
    elif z >= snow_depth:
        nz = n_ice
    return nz

def enceladus_water_pocket_new(z, snow_depth = 25, ice_depth_1 = 50, water_pocket_depth = 150, ice_depth_2 = 200):
    epsilon_ice_1 = 3.19
    epsilon_ice_2 = 1e-5
    n_ice = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2) 
    epsilon_snow_1 = 1.2
    epsilon_snow_2 = 0.139
    n_snow = np.sqrt(epsilon_snow_1 + 1j * epsilon_snow_2)
    epsilon_water_1 = 85.355 
    epsilon_water_2 = 90.884
    n_water = np.sqrt(epsilon_water_1 + 1j * epsilon_water_2)
    n0 = 1 
    
    
    if z < 0:
        nz = n0
    elif z >= 0 and z < snow_depth:
        nz = n_snow
    elif z >= snow_depth and z < ice_depth_1:
        nz = n_ice
    elif z >= ice_depth_1 and z < water_pocket_depth:
        nz = n_water
    elif z >= water_pocket_depth and z < ice_depth_2:
        nz = n_ice
        
    return nz




def enceladus_water_pocket(z, snow_depth=-20, firn_depth=-10, surface_depth = 0 , ice_depth_1 = 50, water_pocket_depth = 150, ice_depth_2 = 200):
    epsilon_ice_1 = 3.2
    epsilon_ice_2 = 1e-5
    epsilon_water_1 = 74
    epsilon_water_2 = 1.8
    n_ice = np.sqrt(epsilon_ice_1 + 1j * epsilon_ice_2)
    n_water = np.sqrt(epsilon_water_1 + 1j * epsilon_water_2) 
    n_firn = np.sqrt(2.2)
    n_snow = np.sqrt(1.2) 
    #epsilon_snow_1 = 
    #epsilon_snow_2 =
    n_snow = np.sqrt(epsilon_snow_1 + 1j * epsilon_snow_2)
    n0 = 1 
    nz = n_water
    
    if z < snow_depth:
        nz = n0
    elif z >= snow_depth and z < firn_depth:
        nz = n_snow
    elif z >= firn_depth and z < surface_depth:
        nz = n_firn
    elif z >= surface_depth and z < ice_depth_1:
        nz = n_ice
    elif z >= ice_depth_1 and z < water_pocket_depth:
        nz = n_water  
    elif z >= water_pocket_depth and z < ice_depth_2:
        nz = n_ice
        
    return nz

# In[4]:


#pulse_file = path_to_directory + simul_directory + 'pulse_input.npy'
#np.save(pulse_file, pulse_input)
### first, initialize an instance of paraProp by defining its dimensions and frequency of interest ###
from scipy import signal


#sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)


frequency_central = 500e6 #Frequency of Transmitter Pulse
t_end = 1000e-9 #Time length of pulse [s]
t_start = 0 #starting time
t_samples = 1/2e9 #Sample Interval
amplitude0 = 1 #TX Amplitude

t_pulse = 200e-9 #Time of Pulse centre


tx_pulse = pulse(amplitude0, frequency_central, t_start, t_end, t_samples) #Start TX Pulse object

#tx_pulse.do_gaussian(t_pulse) #Set TX pulse to gaussian pulse
tcentral = t_pulse
i, q, e = signal.gausspulse(tx_pulse.time_space - tcentral, fc=tx_pulse.frequency, retquad=True, retenv=True)
tx_pulse.real = i
tx_pulse.imag = q
tx_pulse.abs = e
tx_pulse.centre = tcentral

nSamples = tx_pulse.nSamples

fig = pl.figure(figsize=(12,14), dpi = 100)
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.set_title('Pulse (time space and freq space)', fontsize = 18)
#ax1.plot(tx_pulse.time_space*1e9, tx_pulse.real, label='real')
#ax1.plot(tx_pulse.time_space*1e9, tx_pulse.imag, label='imag')
ax1.plot(tx_pulse.time_space*1e9, tx_pulse.abs, label='absolute pulse')
ax1.set_xlim(190, 210)
ax1.tick_params(labelsize=18)
ax2.tick_params(labelsize=18)
ax1.legend(fontsize=16, shadow=True)
ax1.grid(linestyle='--')
ax1.set_xlabel('Time [ns]', fontsize=16)
ax1.set_ylabel('Amplitude [arb.]', fontsize=16)
#ax2.set_title('Pulse FFT (freq space)', fontsize=18)
ax2.plot(tx_pulse.freq_space/1e6, abs(tx_pulse.doFFT()), label='absolute pulse fft')
ax2.set_xlim(0, 1000)
#ax2.plot(tx_pulse.freq_space/1e6, tx_pulse.doFFT().real, label='real')
#ax2.plot(tx_pulse.freq_space/1e6, tx_pulse.doFFT().imag, label='imag')
ax2.set_xlabel('Frequency [MHz]', fontsize=16)
ax2.set_ylabel('Amplitude [arb.]', fontsize=16)

ax2.grid()
ax2.legend(fontsize=16, shadow=True)
pl.show()

# In[5]:

iceDepth = 150. # m
iceLength = 120. # m
dx = 1 # m
dz = 0.05 # m

### it is useful to set the reference depth as the source depth when you only have one transmitter ###
sourceDepth = 70. # m
sim = ppp.paraProp(iceLength, iceDepth, dx, dz, refDepth=sourceDepth)
z = sim.get_z()
x = sim.get_x()

#Define Receiver Positions:
max_depth_tx = 50.
max_depth_rx = 50.
step_interval = 2
nDepths_rx = int(max_depth_rx / step_interval) + 1

nTransmitters = nDepths_rx
sourceDepth_list = np.linspace(0, max_depth_tx, nTransmitters)
receiverDepth_list = np.linspace(0, max_depth_rx, nDepths_rx)

#Range List #Change Range of Receivers
Range_list = np.array( [10, 20, 30, 50, 80, 100] )
nRanges = len(Range_list)

rx_array = np.ones((nDepths_rx, nRanges, 2))
rx_list = []
for i in range(nDepths_rx):
    for j in range(nRanges):
        rx_array[i, j][0] = Range_list[j]
        rx_array[i, j][1] = receiverDepth_list[i]
        rx_list.append([Range_list[j], receiverDepth_list[i]])
        
fig = pl.figure(figsize=(15,8), dpi = 100)
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_title('Transmitter and \n Receiver Positions', fontsize=25)
ax1.scatter(np.array(rx_list)[:,0], -np.array(rx_list)[:,1], label='RX')

tx_array = np.ones(nTransmitters)

for i in range(nTransmitters):
    tx_array[i] = sourceDepth_list[i]
ax1.scatter(np.zeros(nTransmitters), -tx_array, label='TX')
ax1.legend(shadow=True, loc='upper right', fontsize=17)
ax1.grid()
ax1.set_ylabel('z [m]', fontsize=25)
ax1.set_xlabel('x [m]', fontsize=25)
ax2.set_xlabel('Refractive Index n', fontsize=20)


#Load a profile from vector
#alpline_profile = np.genfromtxt('share/alpine-schumman-indOfRef_5cm.dat')
#sim.set_n('vector', nVec = alpline_profile[:,1])
#sim.set_n('func', nFunc=enceladus_pole)

method = 'func'
profile = 'enceladus_water_pocket_new'
sim.set_n(method, nFunc=enceladus_water_pocket_new)
### plot ###


ax2.plot(sim.get_n(), -z, color='black')
ax2.grid()
ax2.set_title('Index of Refraction Profile \n of Simulation', fontsize=25)
ax2.text(1.2, 15, 'Vacuum' , fontsize = 20, family= 'serif')
ax2.text(1.5, -12, 'Impure Snow', fontsize = 20, family = 'serif')
ax2.text(2.2, -35, 'Pure Ice', fontsize = 20, family = 'serif')
ax2.text(3.9, -70, 'Saline Water', fontsize = 20, family = 'serif')
ax2.set_ylim(-max_depth_rx - 35, 35)
ax1.set_ylim(-max_depth_rx - 5, 13)
ax1.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
plt.show()


pl.show()


sim.set_dipole_source_profile(frequency_central/1e9, max_depth_rx/2)
sim.set_cw_source_signal(frequency_central/1e9)


### run the solver ###
sim.do_solver()

### plot absolute value of field for whole simulation space ###
fig = pl.figure(figsize=(15,8), dpi = 100)
ax = fig.add_subplot(111)

#pmesh = pl.imshow(10*np.log10(np.transpose(abs(sim.get_field()))), aspect='auto', cmap='hot',  vmin=-50, vmax=20, 
 #         extent=(x[0], x[-1], z[-1], z[0]))
pmesh = pl.imshow(np.transpose(abs(sim.get_field())), aspect='auto', cmap='hot',  vmin=1e-5, vmax=1e-2, 
          extent=(x[0], x[-1], z[-1], z[0]))
cbar = pl.colorbar(pmesh)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Amplitude [arb.]', fontsize = 18)
pl.title("Absolute Field, " + str(int((frequency_central/1e9)*1000))+" MHz", fontsize=25)
pl.xlabel("x (m)", fontsize=25)
pl.ylabel("z (m)", fontsize=25)
pl.xticks(fontsize=20)
pl.yticks(fontsize=20)
pl.show()


# In[6]:


from numpy.lib.format import open_memmap

def create_memmap(fname, path_to_directory, shape0, dtype0 = 'float32', mode0 ='w+'): #Create a Blank Memmap for recording data
    full_file = path_to_directory + fname
    A = open_memmap(full_file, dtype=dtype0, mode=mode0, shape=shape0)
    return A

def generate_ascan(pulse_tx, tx_array, rx_array, path_to_file, file_name): #Creates an Array of Scans for a set of receivers and transmitters
    nRanges_rx = len(rx_array[0])
    nDepths_rx = len(rx_array)
    nDepths_tx = len(tx_array)

    freq_space = pulse_tx.freq_space
    nFreqs = len(freq_space)

    fft_pulse = np.fft.fft(pulse_tx.real)

    ascan_array = create_memmap(file_name, path_to_file, shape0 = (nFreqs, nDepths_tx, nRanges_rx, nDepths_rx), dtype0=complex)
    #full_file = path_to_file + file_name
    print(nFreqs, nDepths_tx, nDepths_rx, nRanges_rx)
    #ascan_array = open_memmap(full_file, dtype=complex, mode='w+', shape=(nFreqs, nDepths_tx, nRanges_rx, nDepths_tx))
    
    for i in range(1, len(freq_space)):
        print(round(float(i)/len(freq_space)*100,2), '%, frequency = ', int(freq_space[i]/1e6), 'MHz')
        frequency_i = freq_space[i]
        amplitude_fft = fft_pulse[i]

        for j in range(nDepths_tx):
        #for j in range(1):  # Here!!!
            sourceDepth = tx_array[j]
            print('source depth: ', sourceDepth)

            sim.set_dipole_source_profile(abs(frequency_i) / 1e9, sourceDepth, fft_pulse[i])  # Set the dipole source
            sim.set_cw_source_signal(abs(frequency_i) / 1e9)  # Set the frequency
            sim.do_solver()  # Run the simulation

            for k in range(nRanges_rx):
            #for k in range(1):
                for l in range(nDepths_rx):
                #for l in range(1):
                    
                    RX = rc(rx_array[l,k][0], rx_array[l,k][1]) #Get Receiver position (range and depth)
                    print(round(float(i)/len(freq_space)*100,2), '%')
                    print('RX: ', RX.x, RX.z, 'source depth: ', sourceDepth, 'frequency = ', int(freq_space[i]/1e6), 'MHz')
                    
                    signal_rx = sim.get_field(RX.x, RX.z)
                    print(ascan_array.shape)
                    ascan_array[i, j, k, l] = signal_rx
                    
                    print('TX amplitude: ', 10*np.log10(abs(fft_pulse[i])), 'RX amplitude: ', 10*np.log10(abs(signal_rx)))
                    print('')
    #np.save(rx_array)
    #np.save(tx_array)
    #np.save(freq_space)

    return ascan_array, tx_array, rx_array

# In[11]:

#CREATE THE FOLDER WHERE YOUR SIMULATION TAKES PLACE
path_to_file = 'test/' #You can Rename the folder when you want to make a new simulation!!
if not os.path.exists(path_to_file):
    os.mkdir(path_to_file)

#Create the file name containing your received pulses
file_name = 'ascan.npy'
full_path = path_to_file + file_name

#Create the meta data files
np.save(path_to_file + 'freq_space.npy', tx_pulse.freq_space) #Frequencies used in Simulation
np.save(path_to_file + 'time_space.npy', tx_pulse.time_space) #Time space of pulse
np.save(path_to_file + 'tx_pulse.npy', tx_pulse.real + 1j*tx_pulse.imag) #Transmitter pulse (complex)
np.save(path_to_file + 'tx_array.npy', tx_array) #Array of Transmitter positions (1D)
np.save(path_to_file + 'rx_array.npy', rx_array) #Array of Receiver positions (2D)

#Save Simulation Data to External File
info_file = path_to_file + 'simul_info.txt'
fout = open(info_file, 'w+')

fout.write(method + '\t#method\n')
fout.write(profile + '\tnProfile\n')
fout.write(str(frequency_central/1e6) + '\t#Central-Frequency-MHz\n')
fout.write(str(tx_pulse.nSamples) + '\t#Number-Samples\n')
fout.write(str(t_samples) + '\t#Sampling-interval-s\n')
nyquist_frequency = 1/(2*t_samples)
fout.write(str(nyquist_frequency)+'\t#nyquist-freauency-Hz\n')

now = datetime.now()
date_str = now.strftime('%Y.%m.%d.%H:%M:%S')
fout.write(date_str + '\t#datetime\n')
fout.close()

# In[ ]:

#Run your simulation -> Results are saved to 'file_name' under 
generate_ascan(tx_pulse, tx_array, rx_array, path_to_file, file_name)


# In[ ]:

from util import *
ascan_rx = np.load(path_to_file + 'ascan.npy')
spectrum_rx = ascan_rx[:,0,4,5]
#print(Range_list[4], sourceDepth_list[5])

pl.plot(tx_pulse.freq_space, spectrum_rx)
pl.show()

pulse_rx = np.fft.ifft(spectrum_rx)
pl.plot(tx_pulse.time_space, tx_pulse.real)
pl.show()

range_scan = np.zeros((nRanges, tx_pulse.nSamples))
for i in range(nRanges):
    spectrum_rx = ascan_rx[:,0,i,0]
    nHalf = int(len(spectrum_rx)/2)
    spectrum_rx[:nHalf] = np.zeros(nHalf)
    
    pulse_rx = np.fft.ifft(spectrum_rx)
    pulse_rx = butterBandpassFilter(pulse_rx, 0.25e9,0.75e9, 1/t_samples,5)
    pl.plot(tx_pulse.time_space, abs(pulse_rx))
    pl.show()
    
    range_scan[i,:] = abs(pulse_rx)

 



#In[ ]:


fig=pl.figure(figsize=(12,10), dpi =100)
ax = fig.add_subplot(111)
pl.imshow(range_scan, extent=(0,max(tx_pulse.time_space)*1e9, max(Range_list), 0),aspect='auto')
pl.show()


# In[ ]:


depth_scan = np.zeros((nDepths_rx, tx_pulse.nSamples))

for i in range(nDepths_rx):
    spectrum_rx = ascan_rx[:,0,1,i]
    nHalf = int(len(spectrum_rx)/2)
    spectrum_rx[:nHalf] = np.zeros(nHalf)
    
    pulse_rx = np.fft.ifft(spectrum_rx)
    pulse_rx = butterBandpassFilter(pulse_rx, 0.25e9,0.75e9, 1/t_samples,5)
    pl.plot(tx_pulse.time_space, abs(pulse_rx))
    pl.show()
    
    depth_scan[i,:] = abs(pulse_rx)


# In[ ]:


for j in range(nRanges):
    depth_scan = np.zeros((nDepths_rx, tx_pulse.nSamples))

    for i in range(nDepths_rx):
        spectrum_rx = ascan_rx[:,0,j,i]
        nHalf = int(len(spectrum_rx)/2)
        spectrum_rx[:nHalf] = np.zeros(nHalf)

        pulse_rx = np.fft.ifft(spectrum_rx)
        pulse_rx = butterBandpassFilter(pulse_rx, 0.25e9,0.75e9, 1/t_samples,5)
       

        depth_scan[i,:] = abs(pulse_rx)
    fig=pl.figure(figsize=(19,10), dpi =100)
    ax = fig.add_subplot(111)
    ax.imshow(depth_scan, extent=(0,max(tx_pulse.time_space)*1e9, max(sourceDepth_list), 0),aspect='auto')
    
    if i == nRanges -1:
        ax.set_xlabel('Time [ns]')
    if i == 0:
        ax.set_ylabel('Depth [m]')
    pl.show()

