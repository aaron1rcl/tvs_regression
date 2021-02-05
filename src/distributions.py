import numpy as np
from scipy.stats import poisson
from scipy.special import iv



# Modified bessel function from wikipedia for discrete gaussian case
def discrete_gaussian_kernel(x, sd):
    try:
        return np.exp(-sd) * iv(x, sd)
    except:
        return 1000
    
def discrete_poisson_kernel(x, mu):
    try:
        dist = poisson(mu)
        return np.array(dist.pmf(x))
    except:
        return 1000
    
