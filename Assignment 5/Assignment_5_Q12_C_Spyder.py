# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 19:22:45 2022

@author: 13ccu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#Part B Harmonic Oscillator
def eqn2(x, y, energy):
    '''Schrodinger equation split into ODEs, for harmonic oscillator'''
    potential = V0 * (x**2)/a**2
    f_0 = y[1]
    f_1 = (-2 * m_el/hbar**2) * (energy - potential) * y[0]
    return np.array([f_0, f_1], dtype = 'object')

def solve(energy, func):
    '''Used for the root finder to energies of states'''
    psi_initial = 0
    dpsi_initial = 1
    initial_values_array = np.array([psi_initial, dpsi_initial])
    x_values = np.linspace(-10*a, 10*a, 1000)
    solution = solve_ivp(func, [-10*a, 10*a], initial_values_array, t_eval=(x_values), args=([energy]))
    psi = solution.y[0][len(solution.y[0]) - 1]
    return psi

def solve2(energy, func):
    '''Used to find wavefunctions of states'''
    psi_initial = 0
    dpsi_initial = 1
    initial_values_array = np.array([psi_initial, dpsi_initial])
    x_values = np.linspace(-10*a, 10*a, 1000)
    solution = solve_ivp(func, [-10*a, 10*a], initial_values_array, t_eval=(x_values), args=([energy]))
    psi = solution.y[0]
    return psi
    
def energy_finder(initial_energy):
    '''Solving for the ground state energy'''
    root = fsolve(solve, initial_energy, eqn2)
    return root

def result():
    '''For test cell, finds difference between excited energy states'''
    difference = excited2_energy_eV - excited1_energy_eV
    return difference

def normalisation(wavefunction):
    x_values = np.linspace(-5*a, 5*a, 1000)
    integrand = (wavefunction**2) * (x_values[1] - x_values[0])
    integral = np.sum(integrand)
    normalised_wavefunction = wavefunction/np.sqrt(integral)
    return normalised_wavefunction

#Main code   
#Constants
m_el = 9.1094e-31#mass of electron in [kg]
hbar = 1.0546e-34#Planck's constant over 2 pi [Js]
e_el = 1.6022e-19#electron charge in [C]
L_bohr = 5.2918e-11#Bohr radius [m]
a = 1E-11
V0 = 50 * e_el
#Have the harmonic oscillator begin at x = -10a and end at x = 10a

initial_energy_guess_ground = 150 * e_el
ground_energy = energy_finder(initial_energy_guess_ground)
ground_energy_eV = ground_energy/e_el
#print(ground_energy_eV)


initial_energy_guess_excited1 = 300 * e_el
excited1_energy = energy_finder(initial_energy_guess_excited1)
excited1_energy_eV = excited1_energy/e_el
#print(excited1_energy_eV)

initial_energy_guess_excited2 = 600 * e_el
excited2_energy = energy_finder(initial_energy_guess_excited2)
excited2_energy_eV = excited2_energy/e_el
#print(excited2_energy_eV)

result()

#Part C plotting wavefunctions
x_values = np.linspace(-5*a, 5*a, 1000)

ground_wavefunction = solve2(ground_energy, eqn2)
#plt.plot(x_values, ground_wavefunction)
ground_normalised_wavefunction = normalisation(ground_wavefunction)
plt.plot(x_values, ground_normalised_wavefunction, label = "Ground state wavefunction")

excited1_wavefunction = solve2(excited1_energy, eqn2)
#plt.plot(x_values, excited1_wavefunction)
excited1_normalised_wavefunction = normalisation(excited1_wavefunction)
plt.plot(x_values, excited1_normalised_wavefunction, label = "First excited state wavefunction")

excited2_wavefunction = solve2(excited2_energy, eqn2)
#plt.plot(x_values, excited2_wavefunction)
excited2_normalised_wavefunction = normalisation(excited2_wavefunction)
plt.plot(x_values, excited2_normalised_wavefunction, label = "Second excited state wavefunction")

plt.xlabel('x')
plt.ylabel('$\psi$')
plt.title('Normalised wavefunctions of a quantum harmonic oscillator')
plt.legend(loc = 'lower left', fontsize = 9, frameon = False)
plt.show()
