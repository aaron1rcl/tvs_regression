import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.special import iv

# Create the binomial input sequence
def create_input(n, p = 0.1, binary=False):
    ''' Creates a binomial vector of length n) '''
    x = np.random.random(n)
    x[x >= (1-p)] = 1
    x[x < (1-p)]  = 0
    
    if binary is False:
        x = np.random.randn(x.shape[0])*x
    return x

# Decompose a vector into the matrix representation
def decompose_vector(x, return_bounds = False):
    ''' Decompose a vector in a matrix representation, where
    x = sum(i, n) xi 
    The formula returns a matrix of the n vectors of xi '''
    # Convert x to a numpy array
    x = np.array(x)
    x_out = np.zeros((x.shape[0], x.shape[0]))
    bounds = np.zeros((x.shape[0], 2))

    # Fill out the matrix
    for i in range(0, x.shape[0]):
        x_i = np.zeros((x.shape[0]))
        x_i[i] = x[i]
        x_out[i,:] = x_i
        
        # Fill the bounds array
        bounds[i,0] = -1*i
        bounds[i,1] = (x.shape[0] - i)

    # Drop all the columns which are complete zeros
    non_zero = np.apply_along_axis(sum , 0, x_out) != 0
    x_out = x_out[non_zero,:]
    bounds = bounds[non_zero, :]
    if return_bounds is True:
        return x_out, bounds
    else:
        return x_out




# Define a lagging function
def shift(x, shift=0):
    x = np.roll(x, shift)
    
    if shift < 0:
        x[shift:] = np.nan
    elif shift > 0:
        x[:shift ] = np.nan
    
    return x



def shift_array(X, shift_seq, fill_na=True):
    # Shift the values by the given sequence (this is actual shift value for inference)
    X_shift = X.copy()
    for i in range(0, X.shape[0]):
        X_shift[i,:] = shift(X[i,:], shift=shift_seq[i])
        
    if fill_na is True:
        X_shift[np.isnan(X_shift)] = 0
        
    return X_shift


def hor_mul(X, A):
    for i in range(0, X.shape[0]):
        X[i,:] =  X[i,:] * A
    return X

def log_likelihood(x, u, sd):
    l = np.log(norm.pdf(x,loc=u, scale=sd))
    # Very Hack workaround for low log likelihoods
    if np.all(np.isnan(l)):
        return -10000
    l = l[~np.isnan(l)]

    return np.sum(l)

def log_pmf_discrete(x, u, sd):
    x = np.array(x)
    # Centre the variable first by subtracting the mean (to make it mean zero)
    x = x - u
    # Get the log likelihood
    l = np.log(discrete_gaussian_kernel(x, sd))
    
    if np.all(np.isnan(l)):
        return -10000
    # Remove na values
    l = l[~np.isnan(l)]
    return np.sum(l)

def linear_objective_fn(X, y, shift_seq, A, tu, tsd, u, sd):
    
    X = shift_array(X, shift_seq=shift_seq)
    
    # Multiply the array vectors by the A
    for i in range(0, X.shape[0]):
        X[i,:] =  X[i,:] * A
    
    # Create the prediction
    y_p = np.sum(X, axis=0)

    res = y - y_p
    
    e_l = log_likelihood(x=res, u=0, sd=1)
    t_l = log_likelihood(x=shift_seq, u=0, sd=1)
    
    return e_l + t_l
    

# Modified bessel function from wikipedia for discrete gaussian case
def discrete_gaussian_kernel(x, sd):
    return np.exp(-sd) * iv(x, sd)


def refine_bounds(bounds, sd):
    t_min = np.min(bounds)
    t_max = np.max(bounds)
    
    # Find the bound range
    bound_range = np.arange(t_min, t_max)
    
    # Calculate the pmfs
    pmf = discrete_gaussian_kernel(bound_range, sd)
    
    bound_range = bound_range[pmf > 0.0001]
    
    # Define the max and min bounds
    t_min = np.min(bound_range)
    t_max = np.max(bound_range)
    
    if t_min < -5:
        t_min = -5
        
    if t_max > 5:
        t_max = 5
    
    # Refine the input bounds for each section
    bounds[bounds < t_min] = t_min
    bounds[bounds > t_max] = t_max
    
    return np.array(bounds, dtype="int32")
    