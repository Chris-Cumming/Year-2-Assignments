# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 00:40:46 2021

@author: 13ccu
"""

#Import necessary functions

import numpy as np
import matplotlib.pyplot as plt


#Create function that calculates the sum of the fourier function for order, n
def fourier(values, order):
    '''Calculating fourier series values'''
    fourier_value = 0
    for i in range(1, order + 1):
        fourier_value += (1/(2*i-1)) * np.sin((2*i - 1) * 2 * np.pi * values)
    fourier_value = fourier_value * (4/np.pi)
    return fourier_value
#Creating x/T values
xT = np.linspace(0, 1, 200)
#print(xT)

#Plot graphs
plt.title("Demonstrating increasing accuracy of approximation", fontsize = 16)
plt.xlabel("x/T", fontsize = 10)
plt.ylabel("S(x,n)", fontsize = 10)
plt.plot(xT, fourier(xT, 3), label = "n = 3")
plt.plot(xT, fourier(xT, 11), label = "n = 11")
plt.plot(xT, fourier(xT, 40), label = "n = 40")
plt.legend(loc = "lower left")
plt.show()
