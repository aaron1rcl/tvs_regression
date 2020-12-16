import pandas as pd
import numpy as np
import os
import rbfopt
from scipy.optimize import differential_evolution
import scipy.optimize as o
import statsmodels.api as sm

# Local libraries
import src.support as src
from linearObjective import linearTauSolver


class linearTVSRModel:
    
    
    
    def __init__(self, rbf_settings, verbose=False):

        self.shift_seq = None
        self.likelihood = None
        
        self.posterior = pd.DataFrame({"A":0,"tsd":0,"sd":0,"likelihood":0}, index=[0])
        
        # Establish the settings
        if rbf_settings is None:
            self.settings = rbfopt.RbfoptSettings(max_evaluations=100, 
                                 max_noisy_evaluations=10,
                                 minlp_solver_path='/Users/aaronpickering/Desktop/bonmin-osx/bonmin',
                                 print_solver_output=False)
        else:
            self.settings = rbf_settings
        
        # Create a dev null output to silence the print output
        if verbose is False:
            self.dev_null = open(os.devnull, 'w')
            
            
    def basic_linear_regression(self):
        # Statsmodels comparison (no intercept, same as above)
        mod = sm.OLS(self.y, self.x)
        res = mod.fit()
        self.basic_lin_summary = res.summary()
        self.min_A = res.params[0]
        self.max_sd = np.std(res.predict(self.x)  - self.y)
        
    
    def inner_objective(self, params):
                     
        print(params)       
        A = params[0]
        tsd = params[1]
        sd = params[2]
        
        if tsd <= 0:
            return 10000
        
        if sd <= 0:
            return 10000
        
        tu = 0
        u = 0
        f = linearTauSolver(self.X, self.y, A, tu, tsd, u, sd)
        
        loop_bounds = src.refine_bounds(self.X_bounds.copy(), tsd)
        print(loop_bounds)
        
        # Create a user black box function
        bb = rbfopt.RbfoptUserBlackBox(self.dimension, 
                                       loop_bounds[:,0], 
                                       loop_bounds[:,1],
                                       np.array(['I']*self.dimension), 
                                       f.objective_function)

        # Crreate the algorithm from the black box and settings
        alg = rbfopt.RbfoptAlgorithm(self.settings,bb)
        alg.set_output_stream(self.dev_null)
        
        val, x_out, itercount, evalcount, fast_evalcount = alg.optimize()
        
        # Save the best state parameter from the internal loop
        if self.likelihood is None:
            self.likelihood = val
        elif val < self.likelihood:
            self.likelihood = val
            self.shift_seq = x_out
            
        print("Likelihood: " + str(round(val,1)))
        
        single_sample = pd.DataFrame({"A":A,"tsd":tsd,"sd":sd,"likelihood":val}, index=[0])
        self.posterior = pd.concat([self.posterior, single_sample], axis=0)
        
        return val
    
    def fit(self, x_train, y_train, method="L-BFGS-B"):
        self.x = x_train
        self.X, self.X_bounds = src.decompose_vector(self.x, return_bounds=True)
        self.dimension = self.X.shape[0]
        self.y = y_train
        
        # Run a standard linear regression first
        self.basic_linear_regression()
        
        # Assign bounds
        bnds = ((self.min_A,10),(0.00,self.x.shape[0]/2), (0.00, self.max_sd))
        
        if method == "L-BFGS-B":
            self.summary = o.minimize(self.inner_objective, [0, 2, 2], method='L-BFGS-B', bounds=bnds)
        elif method == "differential_evolution":
            self.summary = differential_evolution(self.inner_objective,
                                                  bnds, 
                                                  polish=True,
                                                  maxiter=50,
                                                  popsize=1)
        else:
            raise("Error: no compatible optimizer found")

        
        