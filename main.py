import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Custom imports
from src.model.linearObjective import linearTauSolver
from src.model.linearTVSR import linearTVSRModel
import src.support as src


# Define additional variables
A = np.array(2, dtype="float")
np.random.seed(20)
x = src.create_input(100, 0.2, binary=False)
X, bounds = src.decompose_vector(x, return_bounds=True)
# Generate a random shift seq
np.random.seed(21)
real_shifts = np.round(np.random.randn(X.shape[0]))
real_shifts = np.array(real_shifts, dtype="int")
# Shift the series
X_shift = src.shift_array(X, np.array(real_shifts, dtype="int"))
X_shift = src.hor_mul(X_shift, A)
# Retrieve the observed series, only for reference.
xi = np.sum(src.shift_array(X, np.array(real_shifts, dtype="int")), axis=0)
y = np.sum(X_shift, axis=0) + np.random.randn(X_shift.shape[1])

#Define the parameters, again only for reference.
tu=0
tsd=1
u=0
sd=1

# For comparison, the log likelihood of the actual maximum value (tau-error + y axis error)?
f = linearTauSolver(X, y, 2, 0, 1, 0, 1)

# This line calculates the alikelihood of the actual parameters (approximately)
f.objective_function(real_shifts)


# Plot the sequences
plt.plot(y)
plt.plot(x)
plt.plot(xi)
plt.show()

## TVS Regression - initialise a model
tvs = linearTVSRModel()


# method is gradient descent like (L-BFGS-B) or genetic (differential_evolution)
# This line does the optimisation.
# The grid search is only set up to reproduce this specific example. If you change the example, change the method too.
# The solution space has saddle points
#   im playing around with more global methods which can handle these local minima
tvs.fit(x, y, method="L-BFGS-B", split=False)

# Print the summary
tvs.summary

#  Linear regression model summary for comparison (from statsmodels)
tvs.basic_lin_summary

# Estimated best shift sequences (taus.)
tvs.shift_seq

# Plot the sample chain
plt.scatter(tvs.posterior['A'], tvs.posterior['likelihood'])
plt.ylim(155, 220)

