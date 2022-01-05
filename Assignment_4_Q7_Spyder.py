'''
SiPM single event modelling
(a) Make a pulse, always 50 ns after the start of the event.
(b) Randomly decide with the dark count rate how many
    additional pulses should be created.
(c) for more than zero, add additional pulses with random amplitude
    in discrete units of scale and random position in time
    according to exponential distribution and dark count rate.
(d) Analyse event: first filter with matched filter and find peaks.
(e) Use results on peak positions and number to fit all peaks, especially if there are more than one
(f) Draw data and fit.
'''
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

def matched_filter(data, template):
    '''Creates the filter'''
    return np.correlate(data, template, mode = 'full')

def pulse_sequence(t, *pos):
    '''Sums all the pulses'''
    total = np.zeros_like(t)
    #Mistake found here, need to remove anything of length less than 4
    if len(pos) <4 :
        pos = pos[0]
    for idx in range(0, len(pos), 4): # superposition of pulses
        total += pulse(t, pos[idx], pos[idx+1], pos[idx+2], pos[idx+3])
    return total

def pulse(time, amplitude, start, rise_time, decay_time):
    '''Creates singular pulse'''
    singlepulse = (np.exp(-(time - start) / rise_time) - np.exp(-(time - start) / decay_time))
    singlepulse[np.where(time < start)] = 0.0 # not defined before onset time, set 0
    #Mistake found here, needed to be -amplitude instead of +amplitude
    return - amplitude * singlepulse

def make_template(rise_time, decay_time):
    '''Creates the template data'''
    #Mistake found here, should start time values at 50ns
    time = np.linspace(0.05, 50.05, 101) # 0.05 mus, 0.5 unit step size
    scale_1 = 1.0   # some scale factor giving reasonable values
    onset_1 = 0.0    # pulse start [ns]
    dummy = pulse(time, scale_1, onset_1, rise_time, decay_time)
    template = dummy / np.trapz(dummy, x = time) # normalized
    return time, template

def data_production(time, cfg):
    '''Produces data arrays'''
    amp = cfg[0]   # some scale factor giving reasonable values
    start = cfg[1]    # pulse start [ns]
    rtime = cfg[2]   # realistic rise time
    dtime = cfg[3] # realistic decay time
    noiselevel = cfg[4] # noise level scale
    dcr = cfg[5] # [1/ns] => 2 MHz

    framestop = time[-1] # final time bin
    Npulses = np.random.poisson(framestop * dcr)
    print ('n pulses: ', Npulses)

    pp = pulse(time, amp, start, rtime, dtime)
    noisy = np.random.normal(pp, scale = noiselevel)
    frame = noisy # first triggered pulse at onset
    for _ in range(Npulses): # additional pulses
        npe = np.random.poisson(1.0) # n photo electrons given DCR
        print('npe: ', npe)
        pretrigger = start
        triggertime = random.expovariate(dcr) # rate parameter
        start = pretrigger + triggertime
        if start > framestop-300:
            break
        if npe > 0:
            print('next onset: ', start)
            #Mistake found here, start was added as a parameter as it was missing
            pp = pulse(time, npe * amp, start, rtime, dtime)
            frame += pp
    return frame

def analysis(tvalues, data, cfg):
    '''Does analysis on data'''
    scale_0 = cfg[0]   # some scale factor giving reasonable values
    rtime = cfg[2]   # realistic rise time
    dtime = cfg[3] # realistic decay time

    # prepare the analysis with the matched filter - get a template pulse
    time, tplt = make_template(rtime, dtime)
    time -= time[-1]
    filtered = matched_filter(data, tplt) # filter
    responsetime = np.concatenate((time[:-1], tvalues), axis=None)

    # search the filtered result for peaks in response
    #Mistake found here, needed first entry of array and filtered instead of data as parameter
    peakfilt = find_peaks(filtered, height = 5.0,distance = 6.0)[0]
    print('in filtered: ', responsetime[peakfilt])

    # fit the pulse, try similar initial values to construction
    # and the identified peak number and positions
    if responsetime[peakfilt].size == 0:
        return None # failed peak finding
    init = []

    # construct the initial parameter array
    #Mistake found here, should be iterating over responsetime array.
    for val in responsetime[peakfilt]:
        init.append([scale_0, val, rtime, dtime])
    try:
        fit_params = curve_fit(pulse_sequence, tvalues, data, p0 = np.array(init))[0]
        print(fit_params)
    except (RuntimeError, ValueError):
        fit_params = None # failed fit
    return fit_params

# Start main script
# make a pulse, consider times in nano seconds [ns]
#Mistake found here should start at 1.5 microseconds
timevalues = np.linspace(1.5, 1501.5, 3001) # 1.5 mus, 0.5 unit step size
scale = 10.0   # some scale factor giving reasonable values
risetime = 2.0   # realistic rise time
decaytime = 150.0 # realistic decay time
onset = 50.0    # pulse start [ns]
nlevel = 0.2  # noise level scale
darkCountRate = 0.002 # [1/ns] => 2 MHz
#Mistake found here, needed to add darkCountRate to config
config = [scale, onset, risetime, decaytime, nlevel, darkCountRate]

# Data production first
result = data_production(timevalues, config)

# then analyse the event
bestfit = analysis(timevalues, result, config)

# finally plotting
plt.plot(timevalues, result, 'r-')
if bestfit is not None:
    plt.plot(timevalues, pulse_sequence(timevalues, bestfit), 'b-')
plt.title('SiPM pulse', size=12)
plt.xlabel('Time [ns]', size=12)
plt.ylabel('Amplitude [mV]', size=12)    
plt.show()
