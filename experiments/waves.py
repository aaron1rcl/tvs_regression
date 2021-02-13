import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import resample
import time


os.chdir("/Users/aaronpickering/Desktop/projects/cronus/")

# Custom imports
from src.model.linearObjective import linearTauSolver
from src.model.linearTVSR import linearTVSRModel
import src.support as src


pi = 3.1415


t = np.linspace(0,20*pi, num=100)

#lynx = pd.read_csv("./experiments/lynx.csv")
#lynx.columns = ["year","value"]

#plt.plot(lynx['value'])

#y = np.array(lynx['value'], dtype="float64")
# Standardise
#y = y/np.max(y)
#t = np.linspace(0,19*pi, num=114)

#x = np.sin(t)
#x[0] = 0.0001


x = np.sin(t)
x[0]=0.0001
phase = 0.05*t
y = 2*np.sin(t + phase)
y[0] = 0.0001

plt.plot(x)
plt.plot(y)


# Generate a vector with resample freq x 10
r = np.linspace(0, 20*np.pi, 1000)

# Interpolate the signal and upsample * 10
x_r = np.interp(r, t, x)
y_r = np.interp(r, t, y)

na_pos = np.arange(0,100)
non_na_pos = np.arange(0,10)*10
# add the last value - makes interpolation better
non_na_pos = np.append(non_na_pos, 99)
na_pos = na_pos[~np.isin(na_pos, non_na_pos)]


# Convert to matrix form
XR = src.decompose_vector(x_r)
XR[na_pos,:] = np.nan


# Generate the shift sequence - assume the that the shifts are defined by some random walk sequence
# With gaussian step sequences
min_res = -10000000
min_shifts = np.isnan
min_sd = np.nan
min_a = np.nan
for i in range(0,100000):
    sd = np.random.uniform()
    a = np.random.randn()*1.5
    tsd = np.random.uniform()*2
    print(i)
    jumps = np.round(np.random.randn(XR.shape[0])*tsd)
    jumps = np.array(jumps, dtype="int")
    C = 0
    
    shift = [C]
    for j in range(1,XR.shape[0]):
        next_shift = shift[len(shift) - 1] + jumps[j]
        shift.append(next_shift)
        
        
    # Use the shift sequence on the matrix XR
    XR_shift = src.shift_array(XR, np.array(shift, dtype="int"))
    
    XR_shift[XR_shift == 0] = np.nan
    # Sum across the positions
    x_pred = np.nanmean(XR_shift, axis=0)
    x_pred = x_pred*a
    
    np.nanmean
    limit = np.max(np.where(~np.isnan(x_pred)))
    # Now remove all zeroes and interpolate between them
    #x_pred[x_pred == 0] = np.nan
    x_pred = pd.Series(x_pred).interpolate()
    x_pred = np.array(x_pred)
    x_pred = x_pred[:limit]
    # Fill the na values at the end (just extrapolate the gradient)
    
    
    time_shifts = np.diff(shift)
    t_l = src.log_pmf_discrete(time_shifts, 0, tsd)
    
    res = y_r[:limit] - x_pred
    res[0] = 0
    e_l = src.log_likelihood(res, 0, sd)
    
    l = t_l + e_l
    
    if l > min_res:
        min_res = l
        min_shifts = shift
        min_sd = sd
        min_a = a
        min_tsd = tsd

# Minimize the residuals   
min_res
min_shifts
min_a
min_tsd


# Plot the result
XR_shift = src.shift_array(XR, np.array(min_shifts, dtype="int"))
    
XR_shift[XR_shift == 0] = np.nan
    # Sum across the positions
x_pred = np.nanmean(XR_shift, axis=0)*min_a

limit = np.max(np.where(~np.isnan(x_pred)))
# Now remove all zeroes and interpolate between them
#x_pred[x_pred == 0] = np.nan
x_pred = pd.Series(x_pred).interpolate()
x_pred = np.array(x_pred)
x_pred = x_pred[:limit]

# Plot y vs x_pred
plt.plot(y_r[:limit])
plt.plot(x_pred)
plt.plot(min_shifts)
