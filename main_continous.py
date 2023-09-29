#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Custom imports
import src.support as src
import src.cont_support as cont_src
from src.model.continuousObjective import contSolver
from src.model.contTVSR import contTVSRModel



#%%
n=1000
z, x, y1 = cont_src.generate_example_waves(n)
plt.plot(x)


# %%
sr=10
p, p_cum = cont_src.generate_sample_perturbations(5, n, truncate=False, sr=10)
y, iterations = cont_src.apply_perturbations(x, p, sr)
y= y + np.random.normal(0, 0.3, n) + 1.5*y**2 #- 0.2*y + 0.01*y**3 +
y = src.standardise(y[~np.isnan(y)])

#%%
plt.plot(y)
plt.plot(x)
plt.show()


# %%
plt.hist(p)
plt.show()
plt.plot(p_cum)


# %%
# Create a model object
wave_reg = contTVSRModel(verbose=True)
# %%
wave_reg.fit(y_train=y[:(n-1)], x_train=x[:(n-1)], method="differential_evolution", cubic_df=7, cubic_t_df=6)


# %%
wave_reg.summary

# %%
y_pred = wave_reg.predict()



# %%
plt.plot(y)
plt.plot(y_pred)
plt.show()
# %%

wave_reg.plot_time_basis()

# %%
wave_reg.basic_linear_regression()

# %%
wave_reg.basic_lin_summary
# %%

#%
wave_reg.history
# %%
plt.hist(tvs.history['likelihood'][~np.isinf(tvs.history['likelihood'])], 50)


# %%
plt.scatter(tvs.history['coeff'].apply(np.abs).apply(np.sum), tvs.history['likelihood'])


# Remove all likelihoods less than a thousand
history = tvs.history[tvs.history['likelihood'] > -800]

# %%
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(history['coeff'].apply(np.abs).apply(np.sum), history['t_coeff'].apply(np.abs).apply(np.sum), history['likelihood'])
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
# %%

fig = plt.scatter(history['coeff'].apply(np.abs).apply(np.sum), history['likelihood'].apply(np.exp))
ax = plt.set_xlim(2.3, 2.6)
plt.show()

# %%
