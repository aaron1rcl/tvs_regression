import numpy as np

def upsample(x, sr):
    n = len(x)
    x_i = np.repeat(np.nan, n*sr)
    pos = np.arange(0, n)*sr
    np.put(x_i, pos , x)
    return x_i

# Take the wave and multiply it by its 1.T to get a square sparse matrix decomposition
def matrix_decomp(x, n, sr):
    ones = np.repeat(1, n*sr)
    # Create 1.T
    ones = ones[np.newaxis,:]
    x = x[:, np.newaxis]
    X = np.dot(x, ones)
    I = np.identity(n*sr)
    I[I == 0] = np.nan
    X = X*I
    return X

def shift(x: np.array, k=1):
    ''' shift the arrays'''
    if k > 0:
        y = np.concatenate([np.repeat(np.nan, k), x])
        y = y[:len(x)]
    elif k < 0:
        k = np.abs(k)
        y = np.concatenate([x, np.repeat(np.nan, k)])
        y = y[k:]
    else:
        y = x
    return y

def X_shift(X, sr, p):
    # First enlarge the array to include the largest positive time shift
    if np.max(p) > 0:
        X_out = np.pad(X,[(0,0),(0,np.max(p))], mode='constant', constant_values=np.nan)
    # Shift the values

    for i in np.arange(1, len(p)):
        x_r = X_out[i*sr,:]
        x_shift = shift(x_r, p[i])
        X_out[i*sr,:] = x_shift

    return X_out

def ffl_interpolation(x, size):
    ''' Fill forward linear interpolation function using numpy methods'''
    nan_idx = np.isnan(x)
    y = np.arange(len(x))
    x[nan_idx] = np.interp(y[nan_idx], y[~nan_idx], x[~nan_idx])
    return x[:size]


def perturb(x, p, sr):
    ''' Perturb a discrete, contiguous vector with a given perturbation profile and sample rate
    
    Inputs:
        x: Input vector to be perturbed
        p: Cumulative perturbation profile
        sr: Sample rate for upsampling. sr=10 fills 9 nan values between each element
    
    Returns:
        The perturbed vector in its original sample rate.
    '''

    if len(x) != len(p):
        raise("Error: given perturbation length does not equal the input vector length")

    # Upsample
    x_i = upsample(x, sr)
    # Decompose to sparse matrix form
    n = len(x)
    X = matrix_decomp(x_i, n, sr)
    # Apply shifts
    X_s = X_shift(X.copy(), sr, p)
    # Reduce
    x_s = np.nanmean(X_s, axis=0)
    # Linearly interpolate
    x_out = ffl_interpolation(x_s, n*sr)
    # Downsample and return
    return x_out[np.arange(0,n)*sr]

    
