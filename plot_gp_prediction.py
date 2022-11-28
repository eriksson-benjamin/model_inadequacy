#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 08:45:31 2022

@author: beriksso
"""

import useful_defs as udfs
import numpy as np
import matplotlib.pyplot as plt
udfs.set_nes_plot_style()

def load_fit(fit_file, dat_file):
    """Return the residuals of the fit in f_name."""
    # Import fit data
    fit = udfs.unpickle(fit_file)['tof_spectrum']

    # Calculate total fit
    tot_fit = (fit['TH (DD)'] + fit['NBI (DT)'] +
               fit['NBI (DD)'] + fit['scatter'])

    # Import experimental data
    dat = udfs.import_data(dat_file)
    
    # Select positive flight times
    counts = dat[0][499:]
    u_counts = np.sqrt(counts)
    background = dat[2][499:]
    u_background = np.sqrt(background)
    signal = dat[0][499:] - dat[2][499:]
    u_signal = np.sqrt(u_counts**2 + u_background**2)
    
    

    return fit['tof_axis'], signal, u_signal, tot_fit


def model_inadequacy(f_name):
    """Load component to compensate for model inadequacy."""
    dat = np.loadtxt(f_name)
    tof_axis = dat[:, 0]
    counts = dat[:, 1]
    
    return tof_axis, counts


if __name__ == '__main__':
    # Load data/model
    fit_file = 'output_files/fit_output.pickle'
    dat_file = 'input_files/model_inadequacy.pickle'
    tof_ax, counts, u_counts, model = load_fit(fit_file, dat_file)
    
    # Load model inadequacy component
    _, counts_mi = model_inadequacy('output_files/gp_prediction.txt')
    
    # Plot
    plt.figure()
    plt.plot(tof_ax, counts, 'k.')
    plt.errorbar(tof_ax, counts, u_counts, color='k', linestyle='None')
    plt.plot(tof_ax, model + counts_mi, 'r-', label='new total')
    plt.plot(tof_ax, model, color='r', linestyle='-.', label='old total')
    plt.plot(tof_ax, counts_mi, 'b--', label='GP prediction')
    
    # Set labels
    plt.xlabel('$t_{TOF}$ (ns)') 
    plt.ylabel('counts')
    plt.legend()
    plt.yscale('log')
    plt.xlim(20, 75)
    plt.ylim(10, 1E5)
    
    
    
    