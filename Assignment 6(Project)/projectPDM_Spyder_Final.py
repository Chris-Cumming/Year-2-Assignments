# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 14:37:15 2022

@author: 13ccu
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic


def light_curve(time, flux, flux_error):
    '''Plots the light curve for the given data'''
    plt.errorbar(time, flux, yerr=(flux_error), marker = '+', linestyle = '', ecolor = "orange")
    plt.xlabel("Time(days)")
    plt.ylabel("Flux")
    plt.title("Light curve")
    plt.show()

def phase_folding(time_values, period_guess):
    '''Converts time values into phase'''
    phi = (np.mod(time_values, period_guess))/period_guess
    return phi

def binning(phi_values, flux_values):
    '''Bins all the data and finds the statistics for each bin'''
    mean_bins = binned_statistic(phi_values, flux_values, statistic = 'mean', bins = 20)[0]
    count_bins = binned_statistic(phi_values, flux_values, statistic = 'count', bins = 20)[0]
    std_bins = binned_statistic(phi_values, flux_values, statistic = 'std', bins = 20)[0]
    variance_bins = std_bins**2
    global_mean = np.mean(flux_values)
    results = np.array([mean_bins, count_bins, variance_bins, global_mean], dtype = 'object')
    return results

def variance_mean_fluxes(mean_bin, count_bin, total_mean):
    '''Finds variance of each bin with global mean'''
    A = np.sum(count_bin * (mean_bin - total_mean)**2)
    return A

def sum_variance_bins(variance_bin):
    '''Calculates sum of the variance of each bin'''
    E = np.sum(variance_bin)
    return E

def ratio_s(N, M, A, E):
    '''Calculates the value of the S statistic to find best period'''
    statistic = ((N-M)*A)/((M-1)*E)
    return statistic

def determining_optimal_period(time_values, time_period_values, flux_values):
    '''Finds optimal period for star and then plots periodogram'''
    for i in range(20000):
        phase_folded_phi = phase_folding(time_values, time_period_values[i])
        bins = binning(phase_folded_phi, flux_values)
        A = variance_mean_fluxes(bins[0], bins[1], bins[3])
        E = sum_variance_bins(bins[2])
        S_period_guess = ratio_s(len(time_values), 10, A, E)
        array_S[i] = S_period_guess
    optimal_S = np.amax(array_S)
    optimal_period_array = np.array([period_values[np.where(array_S == optimal_S)]])
    optimal_period = optimal_period_array[0][0]
    #Periodogram plot
    plt.scatter(period_values, array_S)
    plt.xlabel("Period(days)")
    plt.ylabel("S(p)")
    plt.title("Periodogram")
    plt.show()
    return optimal_period

def optimal_phase_folded(period, time, flux_phase):
    '''Plots the phase folded curve with the optimal period'''
    phase_values = phase_folding(time, period)
    plt.scatter(phase_values, flux_phase)
    plt.xlabel("Phase Value")
    plt.ylabel("Flux")
    plt.title("Phase folded light curve")
    plt.show()

def fractional_difference(period_guess, true_period):
    fractional_difference = 100 * ((period_guess - true_period)/true_period)
    return np.abs(fractional_difference)

#Load in all data files from within same folder as code
star1 = np.loadtxt("TIC-8170664_EA2_FLUX.dat")
star2 = np.loadtxt("TIC-37157588_RU_lep_FLUX.dat")
star3 = np.loadtxt("TIC-61213992_EA2_FLUX.dat")
star4 = np.loadtxt("TIC-437253391_RV_crt_FLUX.dat")
star5 = np.loadtxt("TIC-61332742_Ellipsoidal_secondary_FLUX.dat")
star6 = np.loadtxt("WASP-5_TOI-250_FLUX.dat")
star7 = np.loadtxt("WASP-7_TOI-2197_FLUX.dat")
star8 = np.loadtxt("WASP-31_TOI-683_FLUX.dat")
star9 = np.loadtxt("WASP-111_TOI-143_FLUX.dat")
star10 = np.loadtxt("WASP-112b_TOI-126_FLUX.dat")
#Find time values
timevalues_star1 = star1[:,0]
timevalues_star2 = star2[:,0]
timevalues_star3 = star3[:,0]
timevalues_star4 = star4[:,0]
timevalues_star5 = star5[:,0]
timevalues_star6 = star6[:,0]
timevalues_star7 = star7[:,0]
timevalues_star8 = star8[:,0]
timevalues_star9 = star9[:,0]
timevalues_star10 = star10[:,0]
#Finding new time values
new_timevalues_star1 = timevalues_star1 - np.amin(timevalues_star1)
new_timevalues_star2 = timevalues_star2 - np.amin(timevalues_star2)
new_timevalues_star3 = timevalues_star3 - np.amin(timevalues_star3)
new_timevalues_star4 = timevalues_star4 - np.amin(timevalues_star4)
new_timevalues_star5 = timevalues_star5 - np.amin(timevalues_star5)
new_timevalues_star6 = timevalues_star6 - np.amin(timevalues_star6)
new_timevalues_star7 = timevalues_star7 - np.amin(timevalues_star7)
new_timevalues_star8 = timevalues_star8 - np.amin(timevalues_star8)
new_timevalues_star9 = timevalues_star9 - np.amin(timevalues_star9)
new_timevalues_star10 = timevalues_star10 - np.amin(timevalues_star10)
#Find flux values
flux_values_star1 = star1[:, 1]
flux_values_star2 = star2[:, 1]
flux_values_star3 = star3[:, 1]
flux_values_star4 = star4[:, 1]
flux_values_star5 = star5[:, 1]
flux_values_star6 = star6[:, 1]
flux_values_star7 = star7[:, 1]
flux_values_star8 = star8[:, 1]
flux_values_star9 = star9[:, 1]
flux_values_star10 = star10[:, 1]
#Flux error values
flux_error_values_star1 = star1[:, 2]
flux_error_values_star2 = star2[:, 2]
flux_error_values_star3 = star3[:, 2]
flux_error_values_star4 = star4[:, 2]
flux_error_values_star5 = star5[:, 2]
flux_error_values_star6 = star6[:, 2]
flux_error_values_star7 = star7[:, 2]
flux_error_values_star8 = star8[:, 2]
flux_error_values_star9 = star9[:, 2]
flux_error_values_star10 = star10[:, 2]

#Plotting light curve finding best period value by finding optimal S statistic,
#Plotting periodgram and plotting phase folded light curve

#TIC-8170664
light_curve(new_timevalues_star1, flux_values_star1, flux_error_values_star1)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star1, period_values, flux_values_star1)
optimal_phase_folded(optimal_p, new_timevalues_star1, flux_values_star1)
print("Optimal period value was found to be", optimal_p, "days")

#TIC-37157588_RU_lep_FLUX
light_curve(new_timevalues_star2, flux_values_star2, flux_error_values_star2)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star2, period_values, flux_values_star2)
optimal_phase_folded(optimal_p, new_timevalues_star2, flux_values_star2)
print("Optimal period value was found to be", optimal_p, "days")

#TIC-61213992_EA2_FLUX
light_curve(new_timevalues_star3, flux_values_star3, flux_error_values_star3)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star3, period_values, flux_values_star3)
optimal_phase_folded(optimal_p, new_timevalues_star3, flux_values_star3)
print("Optimal period value was found to be", optimal_p, "days")

#TIC-437253391_RV_crt_FLUX
light_curve(new_timevalues_star4, flux_values_star4, flux_error_values_star4)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star4, period_values, flux_values_star4)
optimal_phase_folded(optimal_p, new_timevalues_star4, flux_values_star4)
print("Optimal period value was found to be", optimal_p, "days")

#TIC-61332742_Ellipsoidal_secondary_FLUX
light_curve(new_timevalues_star5, flux_values_star5, flux_error_values_star5)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star5, period_values, flux_values_star5)
optimal_phase_folded(optimal_p, new_timevalues_star5, flux_values_star5)
print("Optimal period value was found to be", optimal_p, "days")

#WASP-5_TOI-250_FLUX
light_curve(new_timevalues_star6, flux_values_star6, flux_error_values_star6)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star6, period_values, flux_values_star6)
optimal_phase_folded(optimal_p, new_timevalues_star6, flux_values_star6)
print("Optimal period value was found to be", optimal_p, "days")
difference = fractional_difference(optimal_p, 1.6284246)
print("Difference from true value is:", difference, "%")

#WASP-7_TOI-2197_FLUX
light_curve(new_timevalues_star7, flux_values_star7, flux_error_values_star7)
period_values = np.linspace(4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star7, period_values, flux_values_star7)
optimal_phase_folded(optimal_p, new_timevalues_star7, flux_values_star7)
print("Optimal period value was found to be", optimal_p, "days")
difference = fractional_difference(optimal_p, 4.9546416)
print("Difference from true value is:", difference, "%")


#WASP-31_TOI-683_FLUX
light_curve(new_timevalues_star8, flux_values_star8, flux_error_values_star8)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star8, period_values, flux_values_star8)
optimal_phase_folded(optimal_p, new_timevalues_star8, flux_values_star8)
print("Optimal period value was found to be", optimal_p, "days")
difference = fractional_difference(optimal_p, 3.405909)
print("Difference from true value is:", difference, "%")

#WASP-111_TOI-143_FLUX
light_curve(new_timevalues_star9, flux_values_star9, flux_error_values_star9)
period_values = np.linspace(1.5, 5, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star9, period_values, flux_values_star9)
optimal_phase_folded(optimal_p, new_timevalues_star9, flux_values_star9)
print("Optimal period value was found to be", optimal_p, "days")
difference = fractional_difference(optimal_p, 2.310965)
print("Difference from true value is:", difference, "%")

#WASP-112b_TOI-126_FLUX
light_curve(new_timevalues_star10, flux_values_star10, flux_error_values_star10)
period_values = np.linspace(0.4, 10, 20000)
array_S = np.empty(20000)
optimal_p = determining_optimal_period(new_timevalues_star10, period_values, flux_values_star10)
optimal_phase_folded(optimal_p, new_timevalues_star10, flux_values_star10)
print("Optimal period value was found to be", optimal_p, "days")
difference = fractional_difference(optimal_p, 3.0353992)
print("Difference from true value is:", difference, "%")
