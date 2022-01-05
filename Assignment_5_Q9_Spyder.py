# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 20:48:43 2021

@author: 13ccu
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def equations(t, y, k_1, k_2):
    '''Defining the ODEs'''
    f_0 = -k_1 * y[0]
    f_1 = k_1 * y[0] - k_2 * y[1]
    array = [f_0, f_1]
    np.array([t])#To get rid of style checker warning, has no effect on code
    return array

def rateEqns(init, time, k1, k2):
    '''Solving the ODEs'''
    ysolve = solve_ivp(equations, time, init, args=(k1, k2))
    return ysolve

def plot_solutions(time_range, answer):
    '''Plotting solutions to ODEs'''
    plt.scatter(time_range, answer.y[0], marker = "D", label = "y1(t)")
    plt.scatter(time_range, answer.y[1], marker = "+", label = "y2(t)")
    plt.ylabel('y(t)',fontsize = 10)
    plt.xlabel('Time(s)',fontsize = 10)
    plt.title("Solving 2 coupled First Order ODEs")
    plt.legend(loc = "upper right")
    plt.show()

def loss_feed(time, answer):
    '''Plotting loss feed of solutions'''
    loss = 100 - answer.y[0] - answer.y[1]
    plt.plot(time, loss)
    plt.ylabel("Loss", fontsize = 10)
    plt.xlabel("Time(s)", fontsize = 10)
    plt.title("Loss Feed of Coupled First Order ODEs")
    plt.show()

#Main Code
#Defining initial values
rate1 = 0.2
rate2 = 0.8
time_values = np.linspace(0.0, 20.0, 16)
y1_initial = 100
y2_initial = 0
initial = np.array([y1_initial, y2_initial])

#Solving coupled ODEs
solution = rateEqns(initial, [0., 20.], rate1, rate2)
#Plotting the solutions and the loss feed
plot_solutions(time_values, solution)
loss_feed(time_values, solution)
