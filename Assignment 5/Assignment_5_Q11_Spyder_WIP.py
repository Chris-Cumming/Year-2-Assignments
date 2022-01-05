# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 16:29:16 2021

@author: 13ccu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from math import atan2


def equations(t, y):
    f_0 = y[1] #dx/dt
    f_1 = -2.0 * (y[2]**2) * y[0] * (1 - (y[0])**2) * np.exp(-(y[0]**2 + y[2]**2))
    f_2 = y[3] #dy/dt
    f_3 = -2.0 * (y[0]**2) * y[2] * (1 - y[2]**2) * np.exp(-(y[0]**2 + y[2]**2))
    return np.array([f_0, f_1, f_2, f_3], dtype = 'object')


def trajectory(impactpar, speed):
    maxtime = 10/speed
    time_values = np.linspace(0, maxtime, 300)
    solution = solve_ivp(equations, [0, maxtime], (impactpar, initial_dx, initial_y, speed), t_eval = time_values)
    return np.array([solution.y[0], solution.y[2]], dtype = 'object')


def scatterangles(allb, speed):
    maxtime = 10/speed
    time_values = np.linspace(0, maxtime, number_impact_parameters)
    array_angles = np.zeros(number_impact_parameters)
    if len(allb) == 1:
        full = solve_ivp(equations, [0, maxtime], (allb, initial_dx, initial_y, speed), t_eval = time_values)
        #print(allb[i])
        #print(full)
        #print(full.y[3][number_impact_parameters - 1])
        #print(full.y[1][number_impact_parameters - 1])
        dx = full.y[1]
        max_dx = dx[len(dx) - 1]
        dy = full.y[3]
        max_dy = dy[len(dy) - 1]
        scatter_angle = atan2(max_dy, max_dx)
        #print(scatter_angle)
        array_angles = scatter_angle
    else:
        for i in range(0, number_impact_parameters):
            full = solve_ivp(equations, [0, maxtime], (allb[i], initial_dx, initial_y, speed), t_eval = time_values)
            #print(allb[i])
            #print(full)
            #print(full.y[3][number_impact_parameters - 1])
            #print(full.y[1][number_impact_parameters - 1])
            dx = full.y[1]
            max_dx = dx[len(dx) - 1]
            dy = full.y[3]
            max_dy = dy[len(dy) - 1]
            scatter_angle = atan2(max_dy, max_dx)
            #print(scatter_angle)
            array_angles[i] = scatter_angle
        plt.plot(allb, array_angles)
        plt.xlabel("Impact parameter", fontsize = 10)
        plt.ylabel(("Scatter Angle"), fontsize = 10)
        plt.title("Demonstration of pinball scattering")
        plt.show()
        #print(array_angles)
    return np.array([array_angles], dtype = 'object')


initial_x = 0.1 #x(0) can be between -0.9 and 0.9
initial_dx = 0.0
initial_y = -2
initial_dy = 0.5 #dy/dt can be between 0 and 0.5

answer = trajectory(initial_x, initial_dy)
plt.plot(answer[0], answer[1])
plt.xlabel("x(t)", fontsize = 10)
plt.ylabel("y(t)", fontsize = 10)
plt.title("Trajectory of pinball due to potential with impact parameter = 0.1")
plt.show()

start_impact_parameter = -0.2
final_impact_parameter = 0.2
number_impact_parameters = int((2 * final_impact_parameter/0.001))
range_impact_parameter = np.linspace(start_impact_parameter, final_impact_parameter, number_impact_parameters)

#print(range_impact_parameter)
angle_scattered = scatterangles(range_impact_parameter, 0.1)

#print(angle_scattered)
