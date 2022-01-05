# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:34:12 2021

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
    '''Solving the Schrodinger Equation for the wavefunction'''
    psi_initial = 0
    dpsi_initial = 1
    initial_values_array = np.array([psi_initial, dpsi_initial])
    x_values = np.linspace(-1E-10, 1E-10, 1000)
    solution = solve_ivp(func, [-1E-10, 1E-10], initial_values_array, t_eval=(x_values), args=([energy]))
    psi = solution.y[0][len(solution.y[0]) - 1]
    return psi

def energy_finder(initial_energy):
    '''Solving for the ground state energy'''
    root = fsolve(solve, initial_energy, eqn2)
    return root



def result():
    '''For test cell, finds difference between excited energy states'''
    difference = excited2_energy_eV - excited1_energy_eV
    return difference

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

























