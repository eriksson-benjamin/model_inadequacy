# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:35:39 2022

@author: bener807
Analysis of the residuals of the fit to the data in
output_files/fit_output.pickle. Gaussian process regression is applied to fit
the residual component.
"""

import sys
sys.path.insert(0, 'C:/python/useful_definitions/')
import useful_defs as udfs
udfs.set_nes_plot_style()
import matplotlib.pyplot as plt
import numpy as np


def load_residuals(fit_file, dat_file):
    """Return the residuals of the fit in f_name."""
    # Import fit data
    fit = udfs.unpickle(fit_file)['tof_spectrum']

    # Calculate total fit
    tot_fit = (fit['TH (DD)'] + fit['NBI (DT)'] +
               fit['NBI (DD)'] + fit['scatter'])

    # Import experimental data
    dat = udfs.import_data(dat_file)

    # Select positive flight times
    counts = dat[0][499:] - dat[2][499:]
    bins = dat[1][499:]

    # Calculate residuals
    res = counts - tot_fit
    u_res = np.sqrt(counts)

    return fit['tof_axis'], res, u_res


def plot_residuals(x_axis, res, u_res):
    """Plot the fit residuals."""
    plt.figure('Residuals')
    # plt.plot(x_axis, residuals, 'k.')
    plt.errorbar(x_axis, res, u_res, color='k',
                 linestyle='None', marker='.', markersize=1)
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('$r$')
    plt.xlim(27.5, 56)


if __name__ == '__main__':
    # Calculate residuals
    fit_file = 'output_files/fit_output.pickle'
    dat_file = 'input_files/model_inadequacy.pickle'
    tof_axis, res, u_res = load_residuals(fit_file, dat_file)

    # Plot residuals
    plot_residuals(tof_axis, res, u_res)

    # Set everything outside of area of interest to zero
    mask = ((tof_axis < 27.5) | (tof_axis > 56))
    x = np.copy(tof_axis)
    y = np.copy(res)
    uy = np.copy(u_res)

    y[mask] = 0
    uy[mask] = 0
