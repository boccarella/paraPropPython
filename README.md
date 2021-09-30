# paraPropPython (Prohira et al., 2021)

this is a simple parabolic equation EM solver. It uses the parabolic equation approximation to simulate the propagation of EM waves in a medium with a changing index of refraction. 

Currently it is designed to simulate propagation of waves beneath the ice and in the air on short-ish baselines. 

2 n(z) profiles can be used: south pole, and taylor dome antarctica. These functional forms can be used as smooth n(z) profiles, or can be augmented with experimental data (for the south pole) and random density fluctuations (for the taylor dome site, data hopefully forthcoming) to simulate realistic RF propagation in the firn. 

It is written in python (which i don't really speak), is new, and is being expanded, and will probably break a lot. email prohira.1 attt osu dottt edu with questions.

## installing

no installation, just clone the repo

## using

cd into the repo directory and try:

python3 simpleExample.py <frequency [GHz, keep it below 1]> <source depth [m]> <use density fluctiations? [0=no, 1=yes]>

and you should see a plot. 

# Analyse-Ascans-xxx.ipynb
iPython-notebook(s) that analyses time-domain PE simulations with regard to directly transmitted signals and reflections from boundary layers. They can be used to analyse
the 'small-scale simulations' (S1 in the thesis) which are saved as .npy files

# runSolver.py
creates 'absolute field plots' for 1 frequency and 1 single source depth, without running the simulation itself

# permittivity.py
calculates real and imaginary part of the complex permittivity for different ices and saline water on Enceladus as a function of frequency, salinity (impurity content) 
and porosity

# enceladus-environment.py
sets the refractive index profiles in the simulation based on the calculations done in permittivity.py

# runSimulation.py / run-Simulation.py
runs the simulation for the small-scale / large-scale simulation. The latter has to be of the form: 'python runSimulation.py input_file.txt output_file.h5 <freq> <depth>'
