# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 18:42:40 2021

@author: 13ccu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def equations(t, y, mu):
    '''Defining 2nd order ODEs with y[0] = x(t), y[1] = dx/dt, y[2] = y(t), y[3] = dy/dt'''
    r = np.sqrt((y[0] + mu)**2 + (y[2])**2)
    s = np.sqrt((y[0] - 1 + mu)**2 + (y[2])**2)
    f_0 = y[1] #dx/dt
    f_1 = y[0] + ((2 * y[3]) - ((1-mu) * (y[0] + mu))/r**3) - ((mu * (y[0] - 1 + mu))/s**3)
    f_2 = y[3] #dy/dy
    f_3 = y[2] - 2 * y[1] - ((1-mu) * (y[2]))/r**3 - (mu * y[2])/s**3
    np.array([t])#To get rid of style checker warning, has no effect on code
    return np.array([f_0, f_1, f_2, f_3])



def satellite(init, time, mu):
    '''Finds solutions to the coupled ODEs'''
    time_values_interval = np.linspace(0, 18, 200)
    solution = solve_ivp(equations, time, init, t_eval = time_values_interval, args=([mu]))
    return solution


def plots(x, y, dx, dy, t):
    '''Plotting trajectory, position and speed of satellite'''
    fig, (trajectory, position, speed) = plt.subplots(1, 3, figsize = (15,5))
    fig.tight_layout(pad = 2.0)

    #Trajectory plot
    trajectory.plot(x, y)
    trajectory.set_title("Trajectory of satellite in xy plane due to 2 large masses", fontsize = 13)
    trajectory.set_xlabel("x(t)", fontsize = 11)
    trajectory.set_ylabel("y(t)", fontsize = 11)

    #Position plot
    position.plot(t, x, label = "x(t)")
    position.plot(t, y, label = "y(t)")
    position.set_title("Variation of x and y coordinates of satellite", fontsize = 13)
    position.set_xlabel("Time(s)", fontsize = 11)
    position.set_ylabel("Position of satellite in xy plane", fontsize = 11)
    position.legend(loc = "lower left")

    #Speed Plot
    speed.plot(dx, dy)
    speed.set_title("Showing variation of speed of satellite in x and y coordinates", fontsize = 13)
    speed.set_xlabel("dx/dt", fontsize = 11)
    speed.set_ylabel("dy/dt", fontsize = 11)


#Main Code
#Defining initial values
mass_ratio = 0.01227471
time_interval = [0,18]
time_values = np.linspace(0, 18, 200)
initial_x = 0.994
initial_dx = 0
initial_y = 0
initial_dy = -2.0015851
initial_values = np.array([initial_x, initial_dx, initial_y, initial_dy])

#Stores the solutions
answer = satellite(initial_values, time_interval, mass_ratio)
x_values = answer.y[0]
y_values = answer.y[2]
dx_values = answer.y[1]
dy_values = answer.y[3]

#Plotting all the graphs
plots(x_values, y_values, dx_values, dy_values, time_values)
