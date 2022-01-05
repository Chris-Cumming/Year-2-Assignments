# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 23:15:08 2021

@author: 13ccu
"""

#Import Modules

import random as rnd
import numpy as np
from scipy import stats
from scipy import constants
import matplotlib.pyplot as plt

#Defining key constants
k = constants.k

#print(atomic_mass)
helium_mass = 4.0 * constants.u


def samples(T1, T2, mass):
    'Creating 2 data sets for 2 different temperatures of helium molecules'
    #Defining a from maxwell distribution
    a_T1 = np.sqrt((k*T1)/(mass))
    a_T2 = np.sqrt((k*T2)/(mass))
    #Generating the maxwell distribution of speeds
    sample_T1 = stats.maxwell.rvs(scale = a_T1, size = 1000)
    sample_T2 = stats.maxwell.rvs(scale = a_T2, size = 1000)
    #print(np.mean(sample_T1))
    #print(np.mean(sample_T2))
    #print(sample_T1)
    #print(sample_T2)
    return sample_T1, sample_T2

def doCollision(ncoll, sample1, sample2):
    'Finding speeds of random particles, then their difference in energy and adjusting accordingly'
    for i in range(ncoll):
        #For T1
        index1 = rnd.randint(0, 999)
        random_speed_T1 = sample1[index1]
        #print(random_speed_T1)
        #For T2
        index2 = rnd.randint(0, 999)
        random_speed_T2 = sample2[index2]
        #print(random_speed_T2)
        #Finding energies and then energy difference
        energy_T1 = 0.5 * helium_mass * (random_speed_T1)**2
        energy_T2 = 0.5 * helium_mass * (random_speed_T2)**2
        energy_difference = abs(energy_T1 - energy_T2)
        #print(energy_difference)
        #Adjusting energies
        if energy_T1 < energy_T2:
            new_energy_T1 = energy_T1 + (0.5 * energy_difference)
            new_energy_T2 = energy_T2 - (0.5 * energy_difference)
        else:
            new_energy_T1 = energy_T1 - (0.5 * energy_difference)
            new_energy_T2 = energy_T2 + (0.5 * energy_difference)
        #Calculating new speeds
        new_speed_T1 = np.sqrt((2 * new_energy_T1)/helium_mass)
        new_speed_T2 = np.sqrt((2 * new_energy_T2)/helium_mass)
        #print(new_speed_T1)
        #print(new_speed_T2)
        #Rewriting original speed array
        sample1[index1] = new_speed_T1
        sample2[index2] = new_speed_T2
    total_distribution = np.append(sample1, sample2)
    #print(total_distribution)
    return total_distribution

def final_temperature(full_sample):
    'Calculates the final temperature of the combined samples'
    mean_final_distribution = np.mean(full_sample)
    sqrt_temp = ((mean_final_distribution)/2) * np.sqrt((np.pi)/2) * np.sqrt((helium_mass)/k)
    temp = (sqrt_temp)**2
    return temp


#Collecting samples from samples function
both_samples = samples(290, 4.2, helium_mass)
#print(samples[0])
#print(samples[1])

#Getting total distriubution
full_distribution = doCollision(10000, both_samples[0], both_samples[1])
#Getting final tempertuare of combined samples
final_temp = final_temperature(full_distribution)
#print(final_temp)

plt.xlabel("Speeds(ms$^{-1}$)")
plt.ylabel("Speed Density")
plt.title("Speed distribution of 2 samples of helium mixing(290K and 4.2K)")
plt.hist(full_distribution, bins = 20)
plt.show()
