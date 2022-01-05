# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:34:12 2021

@author: 13ccu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.integrate import solve_bvp
from scipy.optimize import fsolve

#Part B Harmonic Oscillator
def eqn2(x, y, energy):
    '''Schrodinger equation split into ODEs, for harmonic oscillator'''
    f_0 = y[1]
    f_1 = (-2 * m_el/hbar**2) * (energy - V0 * (y[3]**2)/a**2) * y[0]
    return np.array([f_0, f_1])

def solve(energy, func):
    '''Solving the Schrodinger Equation for the wavefunction'''
    psi_initial = 0
    dpsi_initial = 1
    initial_values_array = np.array([psi_initial, dpsi_initial])
    x_values = np.linspace(0, L_bohr, 1000)
    solution = solve_ivp(func, [0, L_bohr], initial_values_array, t_eval=(x_values), args=([energy]))
    psi = solution.y[0][len(solution.y[0]) - 1]
    return psi

def energy_finder(initial_energy):
    '''Solving for the ground state energy'''
    root = fsolve(solve, initial_energy, eqn2)
    return root


#Main code   
#Constants
m_el = 9.1094e-31#mass of electron in [kg]
hbar = 1.0546e-34#Planck's constant over 2 pi [Js]
e_el = 1.6022e-19#electron charge in [C]
L_bohr = 5.2918e-11#Bohr radius [m]
a = 1E-11
V0 = 50 * e_el
initial_energy_guess = 150 * e_el
#Have the harmonic oscillator begin at x = -10a and end at x = 10a

ground_energy = energy_finder(initial_energy_guess)
ground_energy_eV = ground_energy/e_el
print(ground_energy_eV)


