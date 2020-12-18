import pandas as pd
import numpy as np
import os
import rbfopt
from scipy.optimize import differential_evolution
import scipy.optimize as o
import statsmodels.api as sm

# Local libraries
import src.support as src
from src.model.linearObjective import linearTauSolver
from src.optimisation.dataSegments import dataSegments

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
        # Refine bounds and define the objective function
        loop_bounds = src.refine_bounds(self.X_bounds.copy(), tsd)
        


        ##### TO-DO
        # Define an input to select whether to split or not
        
        segments = dataSegments(self.X.copy(), self.x.copy(), self.y.copy(), loop_bounds)
        segments.refine_bounds()
        
        # Loop through all the segments
        vals = np.array([])
        taus = np.array([])
        print("gets here")
        for j in range(0, len(segments.X_segments)):
            X = segments.X_segments[j]
            y = segments.y_segments[j]
            dim = segments.dimension[j]
            # Temporary fix until the bounds are better organised
            bnds = 0
 
            # Define our solver for the internal loop
            f = linearTauSolver(X, y, A, tu, tsd, u, sd)
            
            # Run inner optimisation loop
            val, tau = self.inner_optimisation(X, y, dim, bnds, f, loop_bounds)
            print(val, tau)
            vals = np.append(vals, val)
            taus = np.append(taus, tau)
        
        max_likelihood = np.sum(vals)
        # Save the best state parameter from the internal loop
        if self.likelihood is None:
            self.likelihood = max_likelihood 
        elif np.sum(vals) < self.likelihood:
            self.likelihood = max_likelihood 
            self.shift_seq = taus
            
        print("Likelihood: " + str(round(max_likelihood ,1)))
                
        return max_likelihood
    

    def outer_optimization(self, method, bounds):
        # Run the outer optimisation loop
        if method == "L-BFGS-B":
            self.summary = o.minimize(self.inner_objective, [0, 2, 2], method='L-BFGS-B', bounds=bounds)
        elif method == "differential_evolution":
            self.summary = differential_evolution(self.inner_objective,
                                                  bounds, 
                                                  polish=True,
                                                  maxiter=50,
                                                  popsize=1)
        else:
            raise("Error: no compatible optimizer found")
            
    def inner_optimisation(self, X, y, dim, bnds, f, loop_bounds):
        # Create a user black box function
        if np.all(loop_bounds == 0):
            x_out = np.array([0]*dim[0])
            val = f.objective_function(x_out)
        else:
            bb = rbfopt.RbfoptUserBlackBox(dim[0], 
                                           [-4]*dim[0], 
                                            [4]*dim[0],
                                            np.array(['I']*dim[0]), 
                                            f.objective_function)
    
            # Crreate the algorithm from the black box and settings
            alg = rbfopt.RbfoptAlgorithm(self.settings,bb)
            alg.set_output_stream(self.dev_null)
                
            val, tau, itercount, evalcount, fast_evalcount = alg.optimize()

            return val, tau
            
            
    def fit(self, x_train, y_train, method="L-BFGS-B", split=True):
        
        # Load the data into the object memory and decompose the vector
        self.x = x_train
        self.X, self.X_bounds = src.decompose_vector(self.x, return_bounds=True)
        self.dimension = self.X.shape[0]
        self.y = y_train
        self.split = split
        
        # Run a standard linear regression first
        self.basic_linear_regression()
        
        # Assign bounds
        bnds = ((self.min_A,10),(0.00,self.x.shape[0]/2), (0.00, self.max_sd))
        
        # Run the outer loop to optimize the global parameters
        self.outer_optimization(method="L-BFGS-B", bounds=bnds)
