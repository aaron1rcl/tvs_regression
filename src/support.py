import numpy as np
from scipy.stats import norm
from scipy.stats import mode
import itertools

# Import custom distributions
from src.distributions import discrete_gaussian_kernel
from src.distributions import discrete_poisson_kernel

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
    # Rolls elements back around if they go off the edge
    # This will ensure that the likelihood isnt very high at those points
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
    ''' Calculate continous PMF log likelihood 
        Currently setup only for gaussian error
        Inputs:
            x: the individual observations
            u: gaussian mean
            sd: gaussian standard deviation
    '''
    l = np.log(norm.pdf(x,loc=u, scale=sd))
    
    if np.all(np.isnan(l)):
        return -100

    l = l[~np.isnan(l)]

    return np.sum(l)

def log_pmf_discrete(x, u, sd, family="gaussian"):
    ''' Calculate the discrete PMF log likelihood for given data points
        Inputs:
            x: individual observations
            u: ** currently set to zero - need to generalise **
            sd: standard deviation for gaussian, mu for poisson
            family: either gaussian or poisson
    '''
    x = np.array(x)
    # Centre the variable first by subtracting the mean (to make it mean zero)
    x = x - u
    # Get the log likelihood
    if family == "gaussian":
        l = np.log(discrete_gaussian_kernel(x, sd))
    elif family == "poisson":
        l = np.log(discrete_poisson_kernel(x, sd))
    else:
        raise("Error: family not found")
    
    if np.all(np.isnan(l)):
        return -100
    
    # Remove na values
    l = l[~np.isnan(l)]
    return np.sum(l)


    

# --- Segmentation Functions ------


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
    
    
    # Refine the input bounds for each section
    bounds[bounds < t_min] = t_min
    bounds[bounds > t_max] = t_max
    
    return np.array(bounds, dtype="int32")


def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}
    

def standardise_f0(x, f_0):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x - mode(x).mode[-1]

def standardise(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x
    