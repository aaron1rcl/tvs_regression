
import numpy as np
from scipy.stats import poisson
from scipy.special import iv
from scipy.stats import norm
from patsy import dmatrix
import pandas as pd

def generate_example_waves(n):
    np.random.seed(12)
    x = np.arange(1,n+1)/100
    x = 2*np.pi*x
    y1 = np.sin(x)
    y2 = np.sin(x) + np.random.normal(0, 0.5, size=n)
    return x, y1, y2


def generate_sample_perturbations(scale, n, type="gaussian", truncate=False, sr=10):
    perturbations = (np.random.normal(loc=0, scale=(scale - 1), size=n))
    perturbations = perturbations.astype("int")
    if truncate is True:
        perturbations[perturbations <= -1*sr] = -1*sr
    p_cum = np.cumsum(perturbations)
    return perturbations, p_cum


def apply_perturbations(y, perturbations, sr):
    perturbations = np.cumsum(perturbations)
    y_p = np.repeat(np.nan, (len(y)*sr))
    y_p[0] = y[0]
    for i in range(1,len(y)):
        y_i = y[i]
        p = sr*(i+1) + perturbations[i]
        # Check whether the perturbed profile matches the time series length, if not then break the loop
        if p >= len(y_p) or p <= 0:
            iterations = i
            break
        else:
            y_p[p] = y_i

    # Now interpolate the values between the data points
    y_p = np.array(pd.Series(y_p).interpolate(method="linear"))

    # Down sample a series (select every nth element based on the sample rate)
    downsample = np.arange(1,len(perturbations))*sr
    y_p = y_p[downsample]
    
    # Pad the perturbed series
    if (len(y_p) < len(y)):
        d = len(y) - len(y_p)
        for i in range(0,d):
            y_p = np.append(y_p, np.nan)


    if 'iterations' not in locals():
        iterations = len(y_p)


    return y_p[:len(perturbations)], iterations


def calc_shift_splines(params, x, y):

    t_range = range(0, len(x))
    # Create the cubic spline basis functions for modelling time shifts
    cubic_splines = dmatrix("cr(x, df=6)", {"x": t_range})
    basis_coeff = params

    # Sum of the splines for the fit
    y_p = np.dot(cubic_splines, basis_coeff)

    # Punish the cumulative sum of the differences to control the endpoint positions
    y = np.cumsum(np.diff(y))
    y_p = np.cumsum(np.diff(y_p))

    res = np.sum(np.sqrt((y - y_p)**2))
    return res


def fit_random_walk(y, sd):
    y_fit = np.zeros(len(y))
    y_fit[0] = np.round(y[0])
    samples = np.zeros(len(y))

    
    for i in range(1, len(y)):
        sample = np.round(np.random.normal(0, scale=sd))

        drift = y_fit[i-1] - y[i-1]
        # Check for drift - if so the standard deviation is too small
        if (drift <= 0):
            y_fit[i] = y_fit[i-1] + np.ceil(np.abs(drift))
            samples[i] = np.ceil(np.abs(drift))
        else:
            y_fit[i] = y_fit[i-1] - np.ceil(np.abs(drift))
            samples[i] = -1*np.ceil(np.abs(drift))


    return y_fit, samples.astype("int")