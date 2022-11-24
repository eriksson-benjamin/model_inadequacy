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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF



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
    counts = dat[0][499:]
    u_counts = np.sqrt(counts)
    background = dat[2][499:]
    u_background = np.sqrt(background)
    signal = dat[0][499:] - dat[2][499:]
    u_signal = np.sqrt(u_counts**2 + u_background**2)
    
    # Calculate residuals
    res = signal - tot_fit
    u_res = u_signal

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


def gp_prediction(l, sigma_f, sigma_n , X_train, y_train, X_test):
    # Kernel definition 
    kernel = (ConstantKernel(constant_value=sigma_f, 
                            constant_value_bounds=(1e-2, 1e2)) *
              RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2)))
    
    # GP model 
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, n_restarts_optimizer=10, )
    # Fitting in the gp model
    gp.fit(X_train, y_train)
    # Make the prediction on test set.
    y_pred = gp.predict(X_test)
    return y_pred, gp

def clean_negatives(y_pred, tof_axis, aoi):
    """
    Removes oscillations in GP prediction outside aoi.
    
    Notes
    -----
    Finds first negative value to the right and left of aoi bounds, sets all 
    values before/after these to zero.
    """
    mask_1 = tof_axis <= aoi[0]
    mask_2 = tof_axis >= aoi[1]
    
    arg_1 = np.argwhere(y_pred[mask_1] < 0).max()
    
    right_ind = len(mask_2)-mask_2.sum()
    arg_2 = np.argwhere(y_pred[mask_2] < 0).min() + right_ind
    
    y_copy = np.copy(y_pred)
    y_copy[0:arg_1+1] = 0
    y_copy[arg_2:] = 0
    
    return y_copy

if __name__ == '__main__':
    # Calculate residuals
    fit_file = 'output_files/fit_output.pickle'
    dat_file = 'input_files/model_inadequacy.pickle'
    tof_axis, res, u_res = load_residuals(fit_file, dat_file)

    # Plot residuals
    plot_residuals(tof_axis, res, u_res)
    
    # Copy data arrays
    x_train = np.copy(tof_axis)
    y_train = np.copy(res)
    uy_train = np.copy(u_res)


    aoi = (27.5, 56) #  area-of-interest
    aof = (20.0, 70.0) #  area-of-fitting
    
    # Masks
    aoi_mask = ((tof_axis >= aoi[0]) & (tof_axis <= aoi[1]))
    aof_mask = ((tof_axis >= aof[0]) & (tof_axis <= aof[1]))
    
    # Set alternative
    alternative = 1
    
    '''
    Alternative 1: set everything outside aoi to zero, include some points 
    outside aoi given by aof.
    '''
    if alternative == 1:
        # Set y-values and uncertainties outside aoi
        y_train[np.invert(aoi_mask)] = 0
        uy_train[np.invert(aoi_mask)] = np.mean(uy_train)
        
        # Select area of fitting
        y_train = y_train[aof_mask]
        uy_train = uy_train[aof_mask]
        x_train = x_train[aof_mask]
    '''
    Alternative 2: only use the area of interest
    '''
    if alternative == 2:
        y_train = y_train[aoi_mask]
        uy_train = uy_train[aoi_mask]
        x_train = x_train[aoi_mask]

    
    d = 1 #  dimensionality
    
    # Reshaping training/test dataset
    x_train = x_train.reshape(len(x_train), d)
    x_test = tof_axis.reshape(len(tof_axis), d)
    
    l_init = 1
    sigma_f_init = 3
    sigma_n = np.sqrt(uy_train)
    
    y_pred, gp = gp_prediction(l_init, sigma_f_init, sigma_n, 
                               x_train, y_train, x_test)    
    
    
    # Generate samples from posterior distribution. 
    y_hat_samples = gp.sample_y(x_test, n_samples=len(x_test))
    
    # Compute the mean of the sample. 
    y_hat = np.apply_over_axes(func=np.mean, a=y_hat_samples, axes=1).squeeze()
    
    # Compute the standard deviation of the sample. 
    y_hat_sd = np.apply_over_axes(func=np.std, a=y_hat_samples, axes=1).squeeze()
    
    # Find where prediction goes negative outside aoi
    
    
    # Plotting the training data.
    plt.figure('GP regression of residuals')
    plt.title('GP regression of residuals', loc='left')
    plt.plot(x_train.squeeze(), y_train, 'k.', label='residuals')
    plt.errorbar(x_train.squeeze(), y_train, yerr=uy_train, color='k', 
                 linestyle='None')
    
    
    # Plot corridor. 
    ax = plt.gca()
    ax.fill_between(x=x_test.squeeze(), y1=(y_hat - 2*y_hat_sd), 
                    y2=(y_hat + 2*y_hat_sd), color='green',alpha=0.3, 
                    label='$\pm 2 \sigma$')
    
    # Plot prediction
    plt.plot(x_test.squeeze(), y_pred, color='green', label='GP regression')


    # Check where prediction goes negative outside aoi
    y_clean = clean_negatives(y_pred, tof_axis, aoi)
    plt.plot(x_test.squeeze(), y_clean, color='r', linestyle='--', 
             label='GP final')

    # Labeling axes
    plt.legend()
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('$r$')

    # Save prediction to file
    to_save = np.array([tof_axis, y_clean]).T
    np.savetxt('gp_prediction.txt', to_save)

    # Save unit matrix to DRF-like file
    unit_matrix = np.diag(np.ones(len(tof_axis)))
    matrix = [list(row) for row in unit_matrix]
    tof_axis_list = list(np.round(tof_axis, 1))
    drf_x, drf_y = tof_axis_list, tof_axis_list
    x_unit, y_unit = 'ns', 'ns'
    info = ('Identity matrix to be used as the DRF for folding a '
            'TOF component in NES.')
    name = 'Identity matrix'

    to_save = {'matrix': matrix, 'drf_x': drf_x, 'drf_y': drf_y, 
               'x_unit': x_unit, 'y_unit': y_unit, 'info': info, 'name': name}
    udfs.json_write_dictionary('identity_matrix.json', to_save)

