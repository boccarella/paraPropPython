# paraPropPython
# c. sbrocco, s. prohira
#Additions made by Alex Kyriacou

import util
import numpy as np
import paraPropPython as ppp
import scipy
import time

#Implement Backwards Solution -> based on do_solver()
def cut_xaxis(x, xmin, xmax):
    idx_min = util.findNearest(xmin, x)
    idx_max = util.findNearest(xmax, x)
    return x[idx_min:idx_max]

#Reflection Coefficient -> inputs can be complex
def reflection_coefficient(n1, n2):
    return abs(n1.real - n2.real) / (n1.real + n2.real)

#Transmission Coefficient
def transmission_coefficient(n1, n2):
    return 1 - reflection_coefficient(n1, n2)

#Simulation Solvers -> including backwards scattering solutions

#Stationary Solver -> for a CW signal at a single frequency
def backwards_solver_stationary(sim,  R_threshold = 0.1): #TODO Create Backwards Reflections for stationary targets
    """
         calculates field at points in the simulation for a continous wave signal (solution for one frequency)

         Precondition: index of refraction and source profiles are set
         Parameters:
            sim : Simulation should be setup externally
            rxList : array of Receiver objects
            R_threshold : Reflection coefficient value used as threshold for initiating backwards scattering -> defaults to 0.1
        ----------
    """

    """
    if (len(rxList) == 0):
        print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
    for rx in rxList:
        rx.setup(sim.freq, sim.dt)
    """
    u_plus = 2 * sim.A * sim.source * sim.filt * sim.freq  # Set Forward Propogating Field u_plus
    sim.field[0] = u_plus[sim.fNum:-sim.fNum]  # TODO -> check this part -> why are you already setting field[freq_min:freq_max] = field at u(f)?? -> This part is confusing!

    sim_all_refl = ppp.paraProp(sim.iceLength, sim.iceDepth, sim.dx, sim.dz, refDepth=sim.refDepth, airHeight=sim.airHeight)

    for jXstep in range(sim.xNum):
        n_j = sim.n2[jXstep,:]
        x_j = sim.x[jXstep]
        ### method II ###
        alpha = np.exp(1.j * sim.dx * sim.k0 * (np.sqrt(1. - (sim.kz ** 2 / sim.k0 ** 2)) - 1.))
        B = n_j**2 - 1
        Y = np.sqrt(1. + (n_j / sim.n0) ** 2)
        beta = np.exp(1.j * sim.dx * sim.k0 * (np.sqrt(B + Y ** 2) - Y))

        u_plus = alpha * (util.doFFT(u_plus))
        u_plus = beta * (util.doIFFT(u_plus))
        u_plus = sim.filt * u_plus

        sim.field[jXstep, :] = u_plus[sim.fNum:-sim.fNum] / (np.sqrt(sim.dx * jXstep) * np.exp(-1.j * sim.k0 * sim.dx * jXstep))

        dNx = n_j - sim.n2[jXstep - 1, :]  # Calculate change of referactive index between n(x=x_i, z) and n(x_i-1,z)
        reflelction_z = reflection_coefficient(n_j, sim.n2[jXstep - 1, :])  # Calculate Reflection coefficient
        # Check for refelections in horizontal direction

        if any(reflelction_z) > R_threshold:  # Checks if there are any reflections than threshold -> set to be 10%
            sim_refl = ppp.paraProp(x_j, sim.iceDepth, sim.dx, sim.dz, refDepth=sim.refDepth, airHeight=sim.airHeight)
            # Except that the length is restricted to the point of reflection

            # Source Amplitude -> calculate from field and reflection coefficient
            sim_refl.source[sim.fNum:-sim.fNum] = sim.field[jXstep, :] * reflelction_z[sim.fNum:-sim.fNum]
            sim.field[jXstep, :] *= transmission_coefficient(n_j, sim.n2[jXstep - 1, :])[sim.fNum:-sim.fNum]

            u_minus = 2 * sim_refl.source * sim.filt * sim.freq
            sim_refl.field[jXstep, :] = u_minus[sim.fNum:-sim.fNum]

            nSteps_backwards = jXstep - 1
            mXstep = jXstep - 1  # This indice goes backwards
            for kBack in range(nSteps_backwards):
                n_k = sim.n2[mXstep, :]

                alpha_minus = np.exp(
                    1.j * sim.dx * sim.k0 * (np.sqrt(1. - (sim.kz ** 2 / sim.k0 ** 2)) - 1.))
                B_minus = n_k ** 2 - 1
                Y_minus = np.sqrt(1. + (n_k / sim.n0) ** 2)
                beta_minus = np.exp(1.j * sim.dx * sim.k0 * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))

                u_minus = alpha_minus * (util.doFFT(u_minus))
                u_minus = beta_minus * (util.doIFFT(u_minus))
                u_minus = sim.filt * u_minus

                sim_refl.field[mXstep, :] = u_minus[sim.fNum:-sim.fNum] / (np.sqrt(sim_refl.dx * (kBack + 1)) * np.exp(-1.j * sim.k0 * sim_refl.dx * (1 + kBack)))
                mXstep -= 1
            sim.field[:jXstep, :] += sim_refl.field[:jXstep,:]
            sim_all_refl.field[:jXstep, :] += sim_refl.field[:jXstep,:]
    return sim, sim_all_refl


#General Solver for a pulse or other time-varying signal
#To save computation time -> it includes frequency cuts
#Note -> simulation should be initiatialized before hand

#TODO: Add option to allow 2nd order reflections
#TODO: -> check for 3D reflections -> If I have a circular target -> Am I reflection off of a pipe or a sphere?? -> Option -> Add 1/R to reflected object??
def backwards_solver(sim, rxList, freq_min, freq_max, R_threshold = 0.1): #Perform a simulation with backwards scattering
    """
            calculates field at points in the simulation for a pulse in frequency space

            Precondition: index of refraction, source profiles and signal pulse spectrum are set
            Parameters
            ----------
            sim : Simulation should be setup externally
            rxList : array of Receiver objects
            freq_min: Minimum Frequency -> should be the 'start' of the pulses' spectrum
            freq_max: Maximum Frequency
            R_threshold: Reflection coefficient value used as threshold for initiating backwards scattering -> defaults to 0.1
    """
    #Cut over your frequency space
    freq_cut = cut_xaxis(sim.freq, freq_min, freq_max)
    idx_min = util.findNearest(freq_min, sim.freq)
    idx_max = util.findNearest(freq_max, sim.freq)
    nFreq = len(freq_cut)

    ### check for Receivers ###
    if (len(rxList) == 0):
        print("Warning: Running time-domain simulation with no receivers. Field will not be saved.")
    for rx in rxList:
        rx.setup(sim.freq, sim.dt)

    for iFreq in range(idx_min, idx_max):
        tstart_i = time.time() #Start a time for every frequency step
        freq_i = sim.freq[iFreq] #Frequency_i
        print('solving for: f = ', freq_i, 'GHz, A = ', sim.A[iFreq], 'step:', iFreq - idx_min, 'steps left:', idx_max - iFreq)

        #Add U_positive field
        u_plus = 2 * sim.A[iFreq] * sim.source * sim.filt * freq_i #Set Forward Propogating Field u_plus
        sim.field[0,:] = u_plus[sim.fNum:-sim.fNum]

        for jXstep in range(1, sim.xNum):
            n_j = sim.n2[jXstep,:] #Verticle referactive index profile at x_i, n_i(z) = n(x = x_i, z)
            x_j = sim.x[jXstep]

            ### method II ###
            alpha = np.exp(1.j * sim.dx * sim.k0[iFreq] * (np.sqrt(1. - (sim.kz**2 / sim.k0[iFreq]**2)) - 1.))
            #print('alpha: ', alpha, 'k0', sim.k0, 'kz', sim.kz)

            '''
            #Error -> was looking at f = 0 -> resulted in nonesense answers k0 -> 0, this mean you had inifinite or nan values
            print('Alpha:', alpha)
            print('1st term',np.exp(1.j * sim.dx * sim.k0[iFreq]))
            print('2nd term', (np.sqrt(1. - (sim.kz**2 / sim.k0[iFreq]**2)) - 1.))
            print('sim.kz', sim.kz)
            print('sim.k0', sim.k0[iFreq])
            print('kz/k0', sim.kz/sim.k0[iFreq])
            #alpha = np.exp(1.j * self.dx * self.k0[j] * (np.sqrt(1. - (self.kz**2 / self.k0[j]**2))- 1.))
            '''
            B = n_j ** 2 - 1
            Y = np.sqrt(1. + (n_j / sim.n0) ** 2)
            beta = np.exp(1.j * sim.dx * sim.k0[iFreq] * (np.sqrt(B + Y ** 2) - Y))

            u_plus = alpha * (util.doFFT(u_plus))
            u_plus = beta * (util.doIFFT(u_plus))
            u_plus = sim.filt * u_plus

            sim.field[jXstep, :] = u_plus[sim.fNum:-sim.fNum] / (np.sqrt(sim.dx * jXstep) * np.exp(-1.j * sim.k0[iFreq] * sim.dx * jXstep))
            #print('uplus', u_plus[idx_min:idx_max])
            #print('check field 1',sim.field[jXstep, idx_min:idx_max])
            dNx = n_j - sim.n2[jXstep-1,:] #Calculate change of referactive index between n(x=x_i, z) and n(x_i-1,z)
            reflelction_z = reflection_coefficient(n_j, sim.n2[jXstep-1,:]) #Calculate Reflection coefficient
            #Check for refelections in horizontal direction


            if any(reflelction_z) > R_threshold: #Checks if there are any reflections than threshold -> set to be 10%
                #print('reflection coeff', max(reflelction_z))
                #print('transmission coeff', (1 - max(reflelction_z)))

                sim_refl = ppp.paraProp(x_j, sim.iceDepth, sim.dx, sim.dz, refDepth=sim.refDepth, airHeight=sim.airHeight) #Create New Simulation space -> same settings and dimensions as last one
                #Except that the length is restricted to the point of reflection

                #Source Amplitude -> calculate from field and reflection coefficient
                '''
                sim_refl.source[:] = sim.field[jXstep,:] * reflelction_z
                '''

                #print(len(sim_refl.source), len(sim.field[jXstep,:]), len(reflelction_z))
                sim_refl.source[sim.fNum:-sim.fNum] = sim.field[jXstep,:] * reflelction_z[sim.fNum:-sim.fNum] #TODO -> Do I need to have this in reduced field form??
                #TODO: Use 'svec source method'
                #TODO: Should I even be using a source??

                u_plus[sim.fNum:-sim.fNum] *= transmission_coefficient(n_j, sim.n2[jXstep - 1, :])[sim.fNum:-sim.fNum] #TODO: Does this multiply all z-space by some transmission factor? Or only the points of reflection??
                #print('uplus', u_plus)
                u_minus = 2 * sim_refl.source * sim.filt * sim.freq[iFreq]
                sim_refl.field[jXstep,:] = u_minus[sim.fNum:-sim.fNum] #TODO: Check if this works as origin properly
                #print('u_minus', u_minus)
                nSteps_backwards = jXstep - 1 #Number of Steps back to origin x=0
                mXstep = jXstep-1 #This indice goes backwards
                for kBack in range(nSteps_backwards):
                    n_k = sim.n2[mXstep, :]

                    alpha_minus = np.exp(1.j * sim.dx * sim.k0[iFreq] * (np.sqrt(1. - (sim.kz ** 2 / sim.k0[iFreq] ** 2)) - 1.))
                    B_minus = n_k ** 2 - 1
                    Y_minus = np.sqrt(1. + (n_k / sim.n0) ** 2)
                    beta_minus = np.exp(1.j * sim.dx * sim.k0[iFreq] * (np.sqrt(B_minus + Y_minus ** 2) - Y_minus))

                    u_minus = alpha_minus * (util.doFFT(u_minus))
                    u_minus = beta_minus * (util.doIFFT(u_minus))
                    u_minus = sim.filt * u_minus

                    sim_refl.field[mXstep, :] = u_minus[sim.fNum:-sim.fNum] / (np.sqrt(sim_refl.dx * (kBack+1)) * np.exp(-1.j * sim.k0[iFreq] * sim_refl.dx * (1+kBack))) #TODO: exp(i*k*x_minus) xminus = x[k] - x_refl
                    mXstep -= 1
                #print('shape: sim_field', sim.field[:jXstep, idx_min:idx_max].shape, 'sim_refl: ', sim_refl.field[:, idx_min:idx_max].shape)
                sim.field[:jXstep, :] += sim_refl.field[:jXstep, :]
        tend_i = time.time()
        fstep_time = tend_i - tstart_i
        if len(rxList) > 0:
            for rx in rxList:
                rx.add_spectrum_component(sim.freq[iFreq], sim.get_field(x0=rx.x, z0=rx.z))
        x_get_rx = 10.
        z_get_rx = sim.refDepth
        #print(sim.field)
        print('Amplitude at receiver: x =', x_get_rx, 'z =', z_get_rx, sim.get_field(x0=x_get_rx, z0=z_get_rx))
        print('Solution time: ', fstep_time, 'Remaining time: ', int((fstep_time*(idx_max - iFreq))/60), 'min ', round((fstep_time*(idx_max - iFreq))%60,2), 's')

    return rxList

#TODO: Check all indices are correct -> check that i, j and k are defined
#TODO: Or -> replace them with obvious labels