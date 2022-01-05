# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:50:29 2021

@author: 13ccu
"""
# YOUR CODE HERE
#Import modules, numpy, scipy, matplotlib


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import  curve_fit



def fitfunc(m_fit, mu_fit, sigma_fit, r_fit, a_fit, b1_fit, b2_fit):
    "Creating fit for curve_fit to find best values of"
    #Background fit
    background = a_fit * np.exp((b1_fit * (m_fit-105.5) + b2_fit * (m_fit-105.5)**2))
    #Signal fit
    signal = (r_fit/(sigma_fit * (2 * np.pi)**0.5)) * np.exp(-((m_fit-mu_fit)**2)/(2*sigma_fit**2))
    combined = background + signal
    return combined

def residual_ratio(experiment_y_values, fit_y_values):
    "Determining values of residual ratios"
    residual_ratios = (experiment_y_values - fit_y_values)/experiment_y_values
    return residual_ratios
# Given
# write the fitfunc function above(!) this since it is being used below.
def fitter(xval, yval, initial):
    ''' function to fit the given data using a 'fitfunc' TBD.
        The curve_fit function is called. Only the best fit values
        are returned to be utilized in a main script.
    '''
    best, _ = curve_fit(fitfunc, xval, yval, p0=initial)
    return best

# Use functions with script below for plotting parts (a) and (b)
# start value parameter definitions, see equations for s(m) and b(m).
# init[0] = mu
# init[1] = sigma
# init[2] = R
# init[3] = A
# init[4] = b1
# init[5] = b2
init = (125.8, 1.4, 470.0, 5000.0, -0.04, -1.5e-4)
xvalues = np.arange(start=105.5, stop=160.5, step=1)
data = np.array([4780, 4440, 4205, 4150, 3920, 3890, 3590, 3460, 3300, 3200, 3000,
                 2950, 2830, 2700, 2620, 2610, 2510, 2280, 2330, 2345, 2300, 2190,
                 2080, 1990, 1840, 1830, 1730, 1680, 1620, 1600, 1540, 1505, 1450,
                 1410, 1380, 1380, 1250, 1230, 1220, 1110, 1110, 1080, 1055, 1050,
                 940, 920, 950, 880, 870, 850, 800, 820, 810, 770, 760])
# YOUR CODE HERE

#Call the fitter function
b_val = fitter(xvalues, data, init)



#Plotting main graph with best fit
plt.title("Determining mass of higgs boson", fontsize = 18)
plt.xlabel("Mass(GeV)", fontsize = 10)
plt.ylabel("Counts", fontsize = 10)
plt.plot(xvalues, data, "+")
plt.plot(xvalues,fitfunc(xvalues,b_val[0],b_val[1],b_val[2],b_val[3],b_val[4],b_val[5]),'r')
plt.show()


#Finding the residual ratios
rdls = residual_ratio(data,fitfunc(xvalues,b_val[0],b_val[1],b_val[2],b_val[3],b_val[4],b_val[5]))
#print(residuals)
#Plotting residual ratios
plt.xlabel("Mass(GeV)", fontsize = 10)
plt.ylabel("Residual Ratio", fontsize = 10)
plt.plot(xvalues, rdls, "+")
plt.show()
