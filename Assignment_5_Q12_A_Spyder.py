# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:09:45 2022

@author: 13ccu
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#Part A, square potential well
def eqn(x, y, energy):
    ''''Schrodinger equation split into ODEs, remembering V(x) = 0 inside potential'''
    f_0 = y[1]
    f_1 = ((-2 * m_el * energy)/hbar**2) * y[0]
    return np.array([f_0, f_1], dtype = object)


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
    root = fsolve(solve, initial_energy, eqn)
    return root

#Main code   
#Constants
m_el = 9.1094e-31#mass of electron in [kg]
hbar = 1.0546e-34#Planck's constant over 2 pi [Js]
e_el = 1.6022e-19#electron charge in [C]
L_bohr = 5.2918e-11#Bohr radius [m]
initial_energy_guess = 1.6E-17
#Have the square well begin at x = 0 and end at x = L

energy_root = energy_finder(initial_energy_guess)
#print(energy_root)
energy_eV = energy_root/e_el
#print(energy_eV)
