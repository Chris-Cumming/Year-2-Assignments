# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 00:40:46 2021

@author: 13ccu
"""
import numpy as np
import matplotlib.pyplot as plt

def moving_average(inputdata, filterorder):
    ''' implement a moving average filter of odd order
        inputdata: to be filtered, NumPy array
        filterorder: integer number as filter order,
                     must be made odd if even, i.e. even+1.
    '''
#First mistake here, the code below was not present, necessary for code to work
    if not filterorder % 2 :
        filterorder += 1
    response = 1 / filterorder  # a float normalization factor
    output = []
    #padding the input for averaging data borders
#Second mistake, should be integer division, //, instead of normal division.
    leftextension = np.flip(inputdata[:filterorder // 2], 0)
    rightextension = np.flip(inputdata[-filterorder // 2 + 1:], 0)
    padded = np.concatenate([leftextension, inputdata, rightextension])

    output.append(np.sum(padded[:filterorder]) * response) # first average
#Third mistake, same mistake as second one should be integer division.
    n = filterorder // 2 + 1
    for idx in range(n, len(inputdata) + n - 1):
        term = output[-1]  # previous
        term += padded[idx + filterorder // 2] * response
        term -= padded[idx - filterorder // 2 - 1] * response
        output.append(term)
    return np.array(output)

def simple_pulse(length, amplitude, risetime, decaytime):
    '''Creating pulse model using numpy'''
    time = np.linspace(0, length, length + 1)
    onset = 0.3 * length # start 30% into the length
    pulse = np.exp(-(time - onset)/risetime) - np.exp(-(time - onset)/decaytime)
    pulse[np.where(time < onset)] = 0.0 # not defined before onset time, set 0
    return -amplitude * pulse

#make the data
AMP = 10.0
RISETIME = 3.0
DECAYTIME = 300.0
data = simple_pulse(2000, AMP, RISETIME, DECAYTIME)
noisy = data + 0.4 * AMP * np.random.normal(size = len(data))

#filter and plot
smooth = moving_average(noisy, 12)
plt.plot(noisy)
plt.plot(smooth, 'r-')
plt.title('Pulse with noise and smoothing')
plt.ylabel('Amplitude', fontsize = 14)
plt.xlabel('Samples', fontsize = 14)
plt.show()
