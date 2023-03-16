#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 08:08:18 2023

@author: beriksso
"""


"""
Fit D(T) spectrum using only TOF components provided by NES. 
Also include the model inadequacy component.
"""

import numpy as np
import matplotlib.pyplot as plt
import useful_defs as udfs
udfs.set_nes_plot_style()
import scipy


def import_data(dat_file, fit_file, mi_file):
    """Import data, fit from NES and model inadequacy component."""
    # TOFu data
    x_dat, y_dat, bgr_dat = np.loadtxt(dat_file, delimiter=',').T
    dat = (x_dat, y_dat-bgr_dat, np.sqrt(y_dat))
    
    # Fit data
    fit = udfs.numpify(udfs.json_read_dictionary(fit_file))
    
    # Model inadequacy component
    x_mi, y_mi = np.loadtxt(mi_file).T
    fit['model_inadequacy'] = y_mi
    
    if not np.all(x_mi == fit['tof_axis']):
        raise ValueError('Model inadequacy component and NES x-axes mismatch.')
    
    return dat, fit


def cash(dat, mod):
    """Return fit statistic."""
    mask = ((dat > 0) & (mod > 0))
    cash = (-2 * np.sum(dat[mask] * np.log(mod[mask]/dat[mask]) 
            + dat[mask] - mod[mask]))
    
    return cash
    

def chi2N(dat, mod):
    """Calculate reduced chi2."""
    mask = ((dat > 0) & (mod > 0))
    chi2 = np.sum((dat[mask] - mod[mask])**2 / mod[mask]) / (len(dat[mask])-4)
    return chi2


def fit_function(params, x, y, mod, xlim):
    """Function for minimizer."""
    model = calculate_model(params, mod)
    
    # Set x-limit
    mask = (x >= xlim[0]) & (x <= xlim[1])

    # Calculate chi2N
    chi2n = chi2N(y[mask], model[mask])
    C_stat = cash(y[mask], model[mask])
    print(f'Chi2N: {chi2n}')
    print(f'Cash: {C_stat}')
    
    return C_stat


def calculate_model(params, mod, return_comp=False):
    """Multiply parameter intensities with corresponding model component."""
    # Component intensities
    I_bt_dt, I_th_dd, I_bt_dd, I_mi = params
    
    bt_dt_comp = I_bt_dt * mod['NBI (DT)'] 
    th_dd_comp = I_th_dd * mod['TH (DD)']
    bt_dd_comp = I_bt_dd * mod['NBI (DD)']
    scatter_comp = mod['scatter'] 
    mi_comp = I_mi*mod['model_inadequacy']
    
    # Calculate model
    model = (bt_dt_comp + th_dd_comp + bt_dd_comp + scatter_comp + mi_comp)
    
    if return_comp:
        return bt_dt_comp, th_dd_comp, bt_dd_comp, scatter_comp, mi_comp
    else:
        return model
    
    
def plot_model(dat, mod, params):
    """Plot model with data."""
    plt.figure('Best fit')
    plt.plot(dat[0], dat[1], 'k.', markersize=1.5)
    plt.errorbar(dat[0], dat[1], dat[2], color='k', linestyle='None')
    
    total = calculate_model(params, mod)
    comps = calculate_model(params, mod, return_comp=True)
    
    plt.plot(dat[0], comps[0], label='BT (DT)', color='g', linestyle='-.')
    plt.plot(dat[0], comps[1], label='TH (DD)', color='b', linestyle='--')
    plt.plot(dat[0], comps[2], label='BT (DT)', color='C1', linestyle='dotted')
    plt.plot(dat[0], comps[3], label='scatter', color='k', 
             linestyle=udfs.get_linestyle('densely dashdotdotted'))
    plt.plot(dat[0], comps[4], label='model inadequacy', color='magenta', 
             linestyle=udfs.get_linestyle('densely dashed'))
    plt.plot(dat[0], total, 'r-', label='total')
    
    # Set labels etc.
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1, 4E4)
    plt.xlim(20, 130)

def set_lims(scale):
    """Set limits for plots."""
    if scale == 'log':
        plt.yscale('log')
        plt.ylim(1, 2E4)
        plt.xlim(20, 120)
    elif scale == 'linear':
        plt.yscale('linear')
        plt.ylim(-200, 5500)
        plt.xlim(27, 80)


if __name__ == '__main__':
    # Import data/model
    shot = 100387    
    
    fit_file = f'input_files/dt_spectrum/{shot}_fit.json'
    dat_file = f'input_files/dt_spectrum/{shot}_dat.txt'
    mi_file = 'output_files/gp_prediction.txt'
    dat, fit = import_data(dat_file, fit_file, mi_file)
    
    # Minimize
    parameters = (1, 1, 1, 0.1)
    xlim = (20, 100)
    popt = scipy.optimize.minimize(fit_function, parameters, 
                                   args=(dat[0], dat[1], fit, xlim))
    plot_model(dat, fit, popt.x)
    
    