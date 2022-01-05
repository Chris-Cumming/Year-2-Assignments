# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 20:06:52 2021

@author: 13ccu
"""
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # import only curve_fit
'''
Script:
Demonstrate the curve_fit function from SciPy.

This example contains faults but also one physics blunder!
'''


def fitFunc(fit, amplitude, decay, constant):
    "Returns the function that needs to be fitted"
    #Third mistake here(physics mistake), the amplitude should not be negative
    return amplitude * np.exp(-decay * fit) + constant


# produce artificial data first from the function
x = np.linspace(0, 4, 50)
#Second mistake here mixture of local and global variables, function paratmer changed to s
t = fitFunc(x, 2.5, 1.3, 0.5)

# add noise to the data using random numbers
noisy = t + 0.2 * t * np.random.normal(size=len(t))

# invoke the scipy function
# fit the noisy data
results = curve_fit(fitFunc, x, noisy)
fitParams = results[0]
#First mistake was found below, should be results not result as variable name
fitErrors = results[1]

print('True values: amplitude = 2.5; decay = 1.3; constant = 0.5')
print('Best fit parameter: ', fitParams[0], ' ;', fitParams[1], ' ;',
      fitParams[2])
print('Fit errors: ', sqrt(fitErrors[0,0]), ' ;', sqrt(fitErrors[1,1]), ' ;',
      sqrt(fitErrors[2,2]))

# plot the noisy data and the fit result together
plt.title('Temperature in time')
plt.ylabel('Temperature (C)', fontsize=14)
plt.xlabel('time (s)', fontsize=14)
plt.errorbar(x, noisy, fmt='bo', yerr=0.2*t)
plt.plot(x, fitFunc(x, fitParams[0], fitParams[1], fitParams[2]), 'r-')
plt.show()
