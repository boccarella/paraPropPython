from numpy import sqrt
from math import pi
import numpy as np

#Fundamental Constants
epsilon0 = 8.85e-12 #Permittivity of Free Space
mu0 = 4*pi*1e-7 #Permeability of Free Space
c0 = 1./sqrt(epsilon0*mu0) #Speed of Light in Vacuum

#Functions
def eps2m(eps_c): #Return complex refractive index (m) from complex relative permittivity (eps_c)
    eps_r = eps_c.real
    eps_i = eps_c.imag
    n_sq = (1/2.) * (sqrt(eps_r**2 + eps_i**2) + eps_r ) 
    k_sq = (1/2.) * (sqrt(eps_r**2 + eps_i**2) - eps_r )
    n = sqrt(n_sq)
    k = sqrt(k_sq)
    m = n + 1j*k
    return m

def cond2eps_im(sigma, f):
    w = 2*pi*f
    return sigma/(w*epsilon0)

def sigma_from_p(p, sigma): #p: porosity, sigma: conductivity for p = 0
    v = 1 - p #volume filling factor
    return sigma*v*(0.68 + 0.32*v)**2



def alpha(eps_c, f):
    m_c = eps2m(eps_c)
    k = m_c.imag
    w = 2*pi*f
    return 4*w*k/c0


def pure_ice(x, z):
    n_vacuum = 1.
    n_ice = 1.78 + 1e-5j
    n_material = n_vacuum
    if z >= 0:
        n_material = n_ice
    else:
        n_material = n_vacuum
    return n_material

def southpole(z):
    A=1.78
    B=-0.43
    C=-0.0132
    return A+B*np.exp(C*z)

def rho2n(rho):
    eps_r = (1 + 0.835*rho)**2
    return eps2m(eps_r)

def poro2n(poro,eps_r0):
    eps_r = eps_r0*poro*(0.68 + 0.32*poro)**2
    return eps2m(eps_r)

    

def epsilon_seawater_real(f, S):
    epsilon_inf = 6.4587
    epsilon_s = 81.820
    tau = (17.303+S*(-6.272e-3))*10**(-12)
    eps_sea_real = epsilon_inf + (epsilon_s-epsilon_inf)/(1+4*pi**2*f**2*tau**2)
    return eps_sea_real

def epsilon_seawater_imag(f, S):
    epsilon_inf = 6.4587
    epsilon_s = 81.820
    tau = 17.303+S*(-6.272e-3)
    sigma = 0.086374+0.077454*S
    eps_sea_imag = ((epsilon_s-epsilon_inf)*2*pi*f*tau)/(1+4*pi**2*f**2*tau**2) + (sigma)/(2*pi*epsilon0*f)
    print('Conductivity=', sigma*1e-6, 'S/m')
    return eps_sea_imag


def epsilon_snow_imag(sig_solid_pure_ice, f, c = 5.114e4, beta = 0.076):
    sigma_pure = sigma_from_p(0.1, sigma_solid_pure_ice)
    sig_sod = c*beta
    sigma_sodium = sigma_from_p(0.1, sig_sod)
    sigma_inf = sigma_pure + sigma_sodium
    print('Conductivity of Snow= ', sigma_inf)
    eps_snow_imag = (sigma_inf*1e-6)/(epsilon0*f)
    return eps_snow_imag

    
    

sigma_solid_pure_ice = 9e-6 #uS/m #8.7
P_surface = 0.9
sigma_pure_snow = sigma_from_p(P_surface, sigma_solid_pure_ice)
print(sigma_pure_snow)

freq_test = 500e6 #100 MHz
eps_i_ice = cond2eps_im(sigma_solid_pure_ice, freq_test)
print(eps_i_ice)

#eps_snow = 1.2 + 0.015j #Permittivity of Enceladus Snow (Type III)
#eps_snow = 1.2 + 0.139*1j # Permittivity of Enceladus Snow (Type III)
eps_snow = 1.2 + 1j*epsilon_snow_imag(sigma_solid_pure_ice, 500e6)
m_snow = eps2m(eps_snow) #

eps_ice = 3.2 + 1j*eps_i_ice
m_ice = eps2m(eps_ice)
print('Snow, eps_r =', eps_snow, 'n = ', m_snow, 'alpha = ', alpha(eps_snow, freq_test))
print('Ice, eps_r =', eps_ice, 'n =', m_ice, 'alpha = ', alpha(eps_ice, freq_test))

eps_meteor = 8.2 + 0.1558j #Taken from Herique et al. (2018) Direct Observations of Asteroid Interior and Regolith Structure: Science Measurement Requirements
eps_vacuum = 1.0
#eps_water = 82 + 899j #Based on the Conductivity of Salt Water -> 5 S/m
#eps_water = 80.778 + 94.9896*1j  
#eps_water = 85.355 + 90.8838*1j
eps_water_real = epsilon_seawater_real(500e6, 30)
eps_water_imag = epsilon_seawater_imag(500e6, 30)
eps_water = eps_water_real + 1j*eps_water_imag
print('Seawater Permittivity=', eps_water_real, '+i', eps_water_imag)
m_water = eps2m(eps_water)
print('Refractive Index of Seawater= ', m_water, 'alpha= ', alpha(eps_water, freq_test))
