# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 19:26:24 2021

@author: 13ccu
"""
#Import modules
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#Define functions needed for main code
def simple_pulse(time, onset, amplitude, rt, dt):
    ''' Generates a singular pulse and sets any pulse before onset to equal 0'''
    pulse = np.exp(-(time-onset)/rt)-np.exp(-(time-onset)/dt)
    #Sets any pulse that has time less than the start time to 0
    pulse[np.where(time<onset)] = 0.0
    return -amplitude * pulse

def oscillation(length, amplitude, freq, decaytime):
    '''Function for final 10% of data, as background data using different function to single_pulse()'''
    time = np.linspace(0, length-1, length)
    S = amplitude * (np.sin(2*np.pi*freq*time))*(np.exp(-time/decaytime))
    return S

def pulse_noise_added(time, onset, rt, dt):
    '''Adding gaussian noise to each section of pulses with different amplitudes randomly'''
    array_pulses = np.zeros((0, len(time)))
    gaussian_noise = np.random.normal(size = 1000)
    #For first 45% of data, noise added with sqaure root of amplitudes between 1 and 100
    for i in range(int(2000*0.45)):
        amplitude = np.random.uniform(1,100)
        pulse = simple_pulse(time, onset, amplitude, rt, dt)
        for x in range(0, len(time)):
            pulse[x] += np.sqrt(amplitude) * np.random.choice(gaussian_noise)
        array_pulses = np.concatenate((array_pulses, np.array([pulse])))
    #For second 45% of data, noise added with select random choice of amplitudes from (11,16,31,82)
    for i in range(int(2000*0.45)):
        amplitude = np.random.choice([11,16,31,82],1)
        pulse = simple_pulse(time, onset, amplitude, rt, dt)
        for x in range(0, len(time)):
            pulse[x] += np.sqrt(amplitude) * np.random.choice(gaussian_noise)
        array_pulses = np.concatenate((array_pulses, np.array([pulse])))
    #For final 10% of data, uses oscillation function(amplitude between 1 and 20) to generate background data with gaussian noise added
    for i in range(int(2000*0.1)):
        amplitude = np.random.uniform(1,20)
        pulse = oscillation(1000, amplitude, 1/80, 500)
        for x in range(0, len(time)):
            pulse[x] += np.sqrt(amplitude) * np.random.choice(gaussian_noise)
        array_pulses = np.concatenate((array_pulses, np.array([pulse])))
    return array_pulses

def fitting(function, data, initial_values):
    '''Attempting to fit data to simple_pulse function, finds optimal parameters and the errors, otherwise discards pulse'''
    array_parameters = np.zeros((0, 4))
    array_errors = np.zeros((0, 4, 4))
    #Discards any data that can't be fitted to simple_pulse function
    for i in range(0, 2000):
        try:
            parameters, parameters_errors = curve_fit(function, time_values, data[i], p0 = np.array(initial_values)
                                        ,bounds = ([0, 0, 0, 0], [1000, np.inf, 1000, 1000]))
            array_parameters = np.concatenate((array_parameters, np.array([parameters])))
            array_errors = np.concatenate((array_errors, np.array([parameters_errors])))
        except (ValueError, RuntimeError):
            pass
    return array_parameters, array_errors

def amplitude_versus_error(x_values, y_values):
    '''Plotting graph of amplitude values of remaining pulses against the error'''
    array_amplitudes = np.zeros((0,1))
    #Removes any bad(too large) values for amplitude and onset relative error and then begins to plot good values
    for i in range(0, len(y_values)):
        if x_values[i, 0, 0]/y_values[i, 0] < 0.1 and y_values[i, 1] <100:
            array_amplitudes = np.append(array_amplitudes, y_values[i, 1])
            plt.scatter(x_values[i, 0, 0]/y_values[i, 0], y_values[i, 1], color = 'red', marker = "s", s = 6)
    #print('Data left = ',len(arr))
    plt.title('Amplitude vs Onset relative error')
    plt.xlabel('Onset Relative Error')
    plt.ylabel('Amplitude(V)')
    plt.show()
    return array_amplitudes

def bins_histogram(amplitude):
    '''Beginning to find number of bins and values of centres of bins for the histogram'''
    bins = plt.hist(amplitude, bins = 200)[1]
    counters = plt.hist(amplitude, bins = 200)[0]
    return bins, counters

def gaussian_fit_peaks(x, *parameters):
    '''Defining gaussian function'''
    function = np.zeros_like(x)
    for i in range(0, len(parameters), 3):
        centre_peak = parameters[i]
        amplitude = parameters[i+1]
        std_dev = parameters[i+2]
        function += amplitude * np.exp( -((x - centre_peak)/std_dev)**2)
    return function

def histogram_peaks_fitted(function, x_values, data, initial_parameters):
    '''Plotting histogram and fitting the peaks to gaussian functions'''
    for i in range(10):
        try:
            parameters = curve_fit(function, x_values, data, p0 = initial_parameters)[0]
            fit = function(x_values, *parameters)
        except (ValueError, TimeoutError):
            print('pass')
            pass
    plt.title('Histogram with gaussian functions fitted to peaks')
    plt.xlabel('Amplitude')
    plt.ylabel('Frequency density')
    plt.plot(x_values, fit)
    plt.show()
    return parameters

def peak_width_against_peak_position(data):
    '''Plotting energy response'''
    width = np.zeros((0,1))
    peak_position = np.zeros((0,1))
    for i in range(0, len(data), 3):
        width = np.append(width, data[i+2])
        peak_position = np.append(peak_position, data[i+1])
    plt.scatter(peak_position, width, c="red")
    plt.title("Energy resolution response for detector")
    plt.xlabel("Width of peak")
    plt.ylabel("Position of peak")
    plt.show()
'Main body of code'
#Generates time values and stores the data for all pulses generated
time_values = np.linspace(0, 999, 1000)
all_data = pulse_noise_added(time_values, 250, 6, 200)
#Tries to find the parameters of each fit for the data and their errors, given some initial parameters
fit_result = fitting(simple_pulse, all_data, [200, 45, 6, 300]) # onset, amplitude, risetime, decaytime
parameters = fit_result[0]
errors_parameters = fit_result[1]
#Prints number of remaining pulses after discarding those that can't be fitted
print('N = ',len(parameters))

first_plot = amplitude_versus_error(errors_parameters, parameters)
#Finding values for histogram
histogram_data = bins_histogram(first_plot)
bins = histogram_data[0]
counters = histogram_data[1]
bin_width = bins[1]-bins[0]
bin_centre_list = [bins[0] + 0.5* bin_width]


for idx in range (1, len(bins)-1):
    bin_centre_list.append(bin_centre_list[-1] + bin_width)    
bin_centres = np.array(bin_centre_list)
#print(len(bin_centres))
#Plots histogram of amplitudes remaining and fits peaks, at 11,16,31,82, to a gaussian
second_plot = histogram_peaks_fitted(gaussian_fit_peaks, bin_centres, counters, [11, 40, 2, 16, 55, 3, 31, 75, 1, 82, 35, 2])
#Plots energy resolution response of peaks
third_plot = peak_width_against_peak_position(second_plot)
