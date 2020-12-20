import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
import scipy.optimize as o
import statsmodels.api as sm

# Local libraries
import src.support as src
from src.model.linearObjective import linearTauSolver
from src.optimisation.segmentation import create_segments
from src.optimisation.custom_tau_optimiser import tau_optimiser

class linearTVSRModel:
    
    
    
    def __init__(self, verbose=False):

        self.shift_seq = None
        self.likelihood = None
        
        self.posterior = pd.DataFrame({"A":0,"tsd":0,"sd":0,"likelihood":0}, index=[0])
        
            
            
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
        
        loop_bounds = src.refine_bounds(self.X_bounds.copy(), tsd)

        ##### TO-DO
        if self.split == True:
            # Define an input to select whether to split or not
            X_segs, y_segs, dim_segs, bnds_segs = create_segments(self.X, self.x, self.y, loop_bounds)
            # Loop through all the segments
            vals = np.array([])
            taus = np.array([])
    
            for j in range(0, len(X_segs)):
                print("Segment " + str(j + 1) + " from " + (str(len(X_segs))))
                X_seg = X_segs[j]
                y_seg = y_segs[j]
                dim_seg = dim_segs[j]
                # Temporary fix until the bounds are better organised
                bnds_seg = [[-4]*dim_seg[0], [4]*dim_seg[0]]
     
                # Define our solver for the internal loop
                f = linearTauSolver(X_seg, y_seg, A, tu, tsd, u, sd)
                
                # Run inner optimisation loop
                val, tau = self.inner_optimisation(dim_seg, f, bnds_seg)
                
                print("------ Seg Likelihood: " + str(val))
                vals = np.append(vals, val)
                taus = np.append(taus, tau)
        else:
            input_obj = linearTauSolver(self.X.copy(), self.y.copy(), A, tu, tsd, u, sd)
            bnds_seg = [[-4]*self.X.shape[0], [4]*self.X.shape[0]]
            vals, taus = self.inner_optimisation(self.X.shape, input_obj, bnds_seg)

        
        max_likelihood = np.sum(vals)
        # Save the best state parameter from the internal loop
        if self.likelihood is None:
            self.likelihood = max_likelihood 
        elif np.sum(vals) < self.likelihood:
            self.likelihood = max_likelihood 
            self.shift_seq = taus
            
        print("Likelihood: " + str(round(max_likelihood ,3)))
        # Append the iteration to a dataframe
        single_sample = pd.DataFrame({"A":A,"tsd":tsd,"sd":sd,"likelihood":max_likelihood}, index=[0])
        self.posterior = pd.concat([self.posterior, single_sample], axis=0)
                
        return max_likelihood
    

    def outer_optimization(self, method, bounds):
        # Run the outer optimisation loop
        if method == "L-BFGS-B":
            self.summary = o.minimize(self.inner_objective, 
                                      [0, 1, 1],
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      options={"eps":0.2})

        elif method == "differential_evolution":
            self.summary = differential_evolution(self.inner_objective,
                                                  bounds, 
                                                  polish=True,
                                                  maxiter=500,
                                                  popsize=20)
        elif method == "grid_search":
            A_test = np.arange(0.1, stop=4, step=0.2)
            t_test = np.arange(0.1, stop=2, step=0.2)
            e_test = np.arange(0.5, stop=3, step=0.2)
            
            df = pd.DataFrame(src.expandgrid(A_test, t_test, e_test))
            df.columns = ["A","tsd","sd"]
            
            for i in df.index:
                self.inner_objective(params=[df.loc[i,'A'], df.loc[i,'tsd'], df.loc[i,'sd']])
        elif method == "basin_hopping":
            minimizer_kwargs = {"method": "L-BFGS-B"}
            self.summary=o.basinhopping(self.inner_objective,
                                      [0,1,1],niter=100,minimizer_kwargs=minimizer_kwargs)
        else:   
            raise("Error: no compatible optimizer found")
            
    def inner_optimisation(self,  dim, f, loop_bounds):
        # Create a user black box function
        if np.all(loop_bounds == 0):
            tau = [0]*dim[0]
            val = f.objective_function(tau)
        else:
            # ->>>>>> Bounds are hard coded into the model - still figuring out a smart way of doing this
            val, tau = tau_optimiser([0]*dim[0], f, 1000, 3)

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
        # I think there should be a bound on the standard deviations in relation to the standard regression
        # I tried some minimal examples and it makes the optimisation get stuck - need to revisit this.
        bnds = ((self.min_A,None),(0.00,None), (0.00, None))
        
        # Run the outer loop to optimize the global parameters
        self.outer_optimization(method=method, bounds=bnds)
