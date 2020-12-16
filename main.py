import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import rbfopt
from linearObjective import linearTauSolver
import src.support as src
from linearTVSR import linearTVSRModel
import statsmodels.api as sm

os.chdir("/Users/aaronpickering/Desktop/projects/cronus/")


# Define additional variables
A = np.array(8, dtype="float")
np.random.seed(20)
x = src.create_input(100, 0.2)
X, bounds = src.decompose_vector(x, return_bounds=True)
# Generate a random shift seq
np.random.seed(21)
real_shifts = np.round(np.random.randn(X.shape[0]))
real_shifts = np.array(real_shifts, dtype="int")
X_shift = src.shift_array(X, np.array(real_shifts, dtype="int"))
X_shift = src.hor_mul(X_shift, A)

xi = np.sum(src.shift_array(X, np.array(real_shifts, dtype="int")), axis=0)
y = np.sum(X_shift, axis=0) + np.random.randn(X_shift.shape[1])

tu=0
tsd=1
u=0
sd=1

# For comparison, the log likelihood of the actual maximum value (tau-error + y axis error)?
f = linearTauSolver(X, y, 8, 0, 1, 0, 1)
f.objective_function(real_shifts)



# Plot the sequences
plt.plot(y)
plt.plot(x)
plt.plot(xi)
plt.show()

# Define optimizer settings
settings = rbfopt.RbfoptSettings(max_evaluations=100, 
                                 max_noisy_evaluations=10,
                                 minlp_solver_path='/Users/aaronpickering/Desktop/bonmin-osx/bonmin',
                                 print_solver_output=False)


## TVS Regression
tvs = linearTVSRModel(settings)
tvs.fit(x, y)
# Print the summary
tvs.summary
tvs.basic_lin_summary
tvs.shift_seq


# Statsmodels comparison (no intercept, same as above)
mod = sm.OLS(y, x)
res = mod.fit()
print(res.summary())
res.params
res.summary()


# Plot the fit
plt.scatter(x, y)
plt.scatter(xi, y)
plt.scatter(np.arange(-20,20)/10,(np.arange(-20,20)/10)*0.89)
plt.show()


plt.scatter(tvs.posterior['A'], np.exp(-tvs.posterior['likelihood']/170))
plt.ylim(0,0.5)
plt.show()



plt.scatter(tvs.posterior['tsd'], tvs.posterior['likelihood'])
plt.ylim(200,500)
plt.xlim(0,15)
plt.show()

