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
import scipy as sp


def load_residuals(fit_file, dat_file):
    """
    Return the residuals of the fit in f_name.

    Parameters
    ----------
    fit_file : str,
        Path to the file containing the fit data.
    dat_file : str
        Path to the file containing the experimental data.

    Returns
    -------
    tuple : (tof_axis, res, u_res),
        tof_axis : ndarray,
            Time-of-flight axis.
        res : ndarray,
            Residuals of the fit.
        u_res : ndarray,
            Uncertainties in the residuals.
    """ 
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
    plt.errorbar(x_axis, res, u_res, color='k',
                 linestyle='None', marker='.', markersize=1)
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('$r$')
    plt.xlim(27.5, 56)


def gp_prediction(l, sigma_f, sigma_n , X_train, y_train, X_test):
    """
    Apply Gaussian Process regression to fit a function to the data.

    Parameters
    ----------
    l : float,
        length scale for the RBF kernel
    sigma_f : float,
        multiplicative factor for the constant kernel
    sigma_n : float,
        additive factor for the variance in the prediction
    X_train : array_like,
        independent variable of training data
    y_train : array_like,
        dependent variable of training data
    X_test : array_like,
        independent variable of test data

    Returns
    -------
    tuple : (y_pred, gp)
        y_pred : array_like,
            predicted values for X_test
        gp : GaussianProcessRegressor object,
            fitted model
    """    
    
    # Kernel definition 
    kernel = (ConstantKernel(constant_value=sigma_f, 
                            constant_value_bounds=(1e-2, 1e2)) *
              RBF(length_scale=l, length_scale_bounds=(1e-2, 1e2)))
    
    # GP model 
    gp = GaussianProcessRegressor(kernel=kernel, alpha=sigma_n**2, 
                                  n_restarts_optimizer=10)
        
    # Fitting in the gp model
    gp.fit(X_train, y_train)
    
    # Make the prediction on test set.
    y_pred, y_std = gp.predict(X_test, return_std=True)
    return y_pred, y_std, gp


def clean_negatives(y_pred, tof_axis, aoi):
    """
    Removes oscillations in GP prediction outside area of interest.

    Parameters
    ----------
    y_pred : array_like,
        predicted values from GP regression.
    tof_axis : array_like,
        Time-of-flight axis.
    aoi : tuple(float),
        Area of interest boundaries.

    Returns
    -------
    array_like:
        Copy of y_pred with all negative values outside of aoi set to zero.

    Notes
    -----
    Finds first negative value to the right and left of area of interest 
    bounds, sets all values before/after these to zero.
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


def prediction_interval(x_train, y_train, x_test, y_hat, y_hat_std):
    """Calculate the 95% prediction interval."""
    # Mean squared error
    args = [np.argmin(np.abs(x_test - x)) for x in x_train]
    mse =  np.sum((y_hat[args] - y_train)**2) / (len(y_train) - 1)
    
    # Students t crit. value
    t_stat = sp.stats.t.ppf(0.95, len(y_train))
    
    # The rest
    x_m = np.mean(x_test)
    term = (x_test - x_m)**2 / np.sum((x_test - x_m)**2)
    factor = np.sqrt(1 + 1/len(y_train) + term).squeeze()
    
    # Prediction interval
    pred_int = t_stat * np.sqrt(mse) * factor
    
    u = y_hat + pred_int
    l = y_hat - pred_int
    
    return u, l


def plot_gp(x_train, y_train, uy_train, x_test, y_pred, y_std):
    """Plot average Gaussian process regression (+/-2 sigma) on top of data."""
    # Plotting the training data.
    plt.figure('GP regression of residuals')
    plt.title('GP regression of residuals', loc='left')
    plt.errorbar(x_train.squeeze(), y_train, yerr=uy_train, color='k', 
                 label='residuals', linestyle='None', marker='.', 
                 markersize=4)
    
    # Prediction interval
    y_u, y_l = prediction_interval(x_train, y_train, x_test, y_pred, y_std)
    ax = plt.gca()
    ax.fill_between(x=x_test.squeeze(), y1=y_l, y2=y_u, color='g',alpha=0.3, 
                    label='95% prediction interval')
    
    # Plot prediction
    plt.plot(x_test.squeeze(), y_pred, color='green', label='GP regression')

    # Labeling axes
    plt.legend(frameon=True)
    plt.xlabel('$t_{TOF}$ (ns)')
    plt.ylabel('$r$')
    plt.xlim(25, 60)
    plt.ylim(-200, 600)
    

def main(alternative=1):
    # Calculate residuals
    fit_file = 'output_files/fit_output.pickle'
    dat_file = 'input_files/model_inadequacy.pickle'
    tof_axis, res, u_res = load_residuals(fit_file, dat_file)

    # Copy data arrays
    x_train = np.copy(tof_axis)
    y_train = np.copy(res)
    uy_train = np.copy(u_res)

    aoi = (27.5, 56) #  area-of-interest
    aof = (20.0, 70.0) #  area-of-fitting
    
    # Masks
    aoi_mask = ((tof_axis >= aoi[0]) & (tof_axis <= aoi[1]))
    aof_mask = ((tof_axis >= aof[0]) & (tof_axis <= aof[1]))
    
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
    
    y_pred, y_std, gp = gp_prediction(l_init, sigma_f_init, sigma_n, 
                               x_train, y_train, x_test)    
    
    
    # Plot
    plot_gp(x_train, y_train, uy_train, x_test, y_pred, y_std)

    # Check where prediction goes negative outside aoi
    y_clean = clean_negatives(y_pred, tof_axis, aoi)

    # Save prediction to file
    to_save = np.array([tof_axis, y_clean]).T
    np.savetxt('gp_prediction.txt', to_save)
    
    
if __name__ == '__main__':
    main(1)