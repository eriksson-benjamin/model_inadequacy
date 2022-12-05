#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:07:08 2022

@author: beriksso
"""

"""
Fit TT spectrum with He-5 resonance components using only TOF components 
provided by NES. Also include the model inadequacy component.
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
    """Calculate Cash-statistic."""
    return -2 * np.sum(dat*(np.log(mod/dat) + 1) - mod - 1)


def chi2N(dat, mod):
    """Calculate reduced chi2."""
    return np.sum((dat - mod)**2 / mod / (len(dat) - 6))


def fit_function(params, x, y, mod, xlim):
    """Function for minimizer."""
    model = calculate_model(params, mod)
    
    # Set x-limit
    mask = (x >= xlim[0]) & (x <= xlim[1])

    # Calculate chi2N
    chi2n = chi2N(y[mask], model[mask])
    print(f'Chi2N: {chi2n}')
    
    return chi2n


def calculate_model(params, mod, return_comp=False):
    """Multiply parameter intensities with corresponding model component."""
    # Component intensities
    I_bt_dt, I_bt_tt, I_gs_tt, I_es_tt, I_scatter, I_mi = params
    
    bt_dt_comp = I_bt_dt*mod['D(T,n)He4'] 
    bt_tt_comp = I_bt_tt*mod['T(T,2n)He4'] 
    gs_tt_comp = I_gs_tt*mod['T(T,n)He5(GS)']
    es_tt_comp = I_es_tt*mod['T(T,n)He5(ES)']
    scatter_comp = I_scatter*mod['scatter'] 
    mi_comp = I_mi*mod['model_inadequacy']
    
    # Calculate model
    model = (bt_dt_comp + bt_tt_comp + gs_tt_comp + es_tt_comp + 
             scatter_comp + mi_comp)
    
    if return_comp:
        return bt_dt_comp, bt_tt_comp, gs_tt_comp, es_tt_comp, scatter_comp, mi_comp
    else:
        return model
    
def plot_model(dat, mod, params):
    """Plot model with data."""
    plt.figure('Best fit')
    plt.plot(dat[0], dat[1], 'k.', markersize=1.5)
    plt.errorbar(dat[0], dat[1], dat[2], color='k', linestyle='None')
    
    total = calculate_model(params, mod)
    comps = calculate_model(params, mod, return_comp=True)
    
    
    plt.plot(dat[0], comps[0], label='D(T,n)He4', color='g', 
             linestyle='-.')
    plt.plot(dat[0], comps[1], label='T(T,2n)He4', color='b', 
             linestyle=udfs.get_linestyle('long dash with offset'))
    plt.plot(dat[0], comps[2], label='T(T,n)He5(GS)', color='C1', 
             linestyle='dotted')
    plt.plot(dat[0], comps[3], label='T(T,n)He5(ES)', color='C3', 
             linestyle='--')
    plt.plot(dat[0], comps[4], label='scatter', color='k', 
             linestyle=udfs.get_linestyle('densely dashdotdotted'))
    plt.plot(dat[0], comps[5], label='model inadequacy', color='magenta', 
             linestyle=udfs.get_linestyle('densely dashed'))
    plt.plot(dat[0], total, 'r-', label='total')
    
    # Set labels etc.
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('counts')
    plt.legend()
    plt.yscale('log')
    plt.ylim(1, 4E4)
    plt.xlim(20, 130)

if __name__ == '__main__':
    # Import data/model
    name = 'nbi'
    
    # Set which mode to use for fit components
    mode = 1
    """
    mode = 0 
    --------
    Regular 3-body TT reaction + resonance spectrum (ES + GS) generated using 
    Breit-Wigner distribution for He-5 mass.
    
    mode = 1
    --------
    TT reactions (3-body, GS, ES) from Gerry Hale's R-matrix theory.
    """


    if mode == 0:
        fit_file = f'input_files/tt_spectrum/{name}_fit.json'
    elif mode == 1:
        fit_file = f'input_files/tt_spectrum/hale_fit.json'

    dat_file = f'input_files/tt_spectrum/{name}_dat.txt'
    mi_file = 'output_files/gp_prediction.txt'
    dat, fit = import_data(dat_file, fit_file, mi_file)
    
    # Minimize
    parameters = (1, 1, 1, 1, 1, 1)
    xlim = (20, 100)
    popt = scipy.optimize.minimize(fit_function, parameters, 
                                   args=(dat[0], dat[1], fit, xlim))
    plot_model(dat, fit, popt.x)
    
    