# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:00:59 2021

@author: 13ccu
"""
#Import modules
from random import gauss
import numpy as np
import matplotlib.pyplot as plt

#Find value of gravity
def gravity(height, time):
    'Returns value of gravity from a certain height and time'
    #First mistake found here, SUVAT equation rearranged wrong
    #Also needed to add functionality to get rid of large values for g
    grav = 2 * height / time**2
    if grav > 29:
        return 0
    else:
        return grav
#Finding variation in gravity
def fallsim(attempts, height, height_error, time, time_error):
    'Simulates the free fall of the object many times'
    collector = ()
    for _ in range(attempts): # no counter variable needed
        distance = gauss(height, height_error)
        watch    = gauss(time, time_error)
        #collector.append(gravity(distance, watch))
        collector += (gravity(distance, watch), )
        #print(collector)
    #Need to get rid of all the values of g that are 0 as this is unphysical
    array_g_values = np.array(collector)
    final_values = np.delete(array_g_values, np.where(array_g_values == 0))
    return final_values

#Defining constants and plotting the histogram
pisa = 58 # [m]
falltime = 3.4  # [s] in standard Earth gravity
herror = 0.5 # [m] uncertainty from where to drop to the floor
werror = 1.0 # [s] watches were rather uncertain
measurements = fallsim(10000, pisa, herror, falltime, werror)
plt.hist(measurements, 21)
plt.title('Gravity constant measurements', fontsize = 16)
plt.ylabel('measurements', fontsize = 14)
plt.xlabel('g [ms$^{-2}$]', fontsize = 14)
plt.show()
