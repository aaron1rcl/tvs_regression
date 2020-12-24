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
    
    '''
    The algorithm flow is as follows:
        fit:
            -> outer optimisation (optimises the continuous parameters)
                -> inner_objective (sets up the inner likelihood)
                    -> inner_optimisation (sets up the inner optimiser)
                        -> tau optimiser (actually does the inner optimisation of Taus)
                        
        1. Basically, Beta, sd, tsd are initilised and fixed (by an optimiser)
        2. for the fixed beta, sd, tsd the best set of taus are found with the inner optimiser
        3.  for the best set of taus the max likelihood is returned
        4. the max likelihood is fed back to the step 1, where a new set of params is initialised (hopefully closer to the true values)
    
    '''
    
    
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
        
        '''     
        
         Inner Objective:
             This function calls the tau optimiser to maximilise the inner likelihood loop.
             Note that the input params are fixed- this is the best set taus for a fixed A, tsd and sd.
             The outer optimisation loop controls the variation  of A, tsd, and sd
             
        Input: 
        params: numpy array
            fixed set of params 
            
        Return:
            max_likelihood: the maximum likelihood estimated for the given A, tsd and sd
            Note: the optimizer minimizes values, therefore the estimates are multiplied by -1
            
        Self assigns (to the object):
            self.likelihood: the best value of the likelihood overall (from all outer optimisation calls)
            self.shift_seq: the tau sequence for the best likelihood estimate overall
            self.posterior: a record of all likelihood optimisation steps. this helps to see the full solutions space for analysis
            
         
        '''
        print(params)       
        A = params[0]
        tsd = params[1]
        sd = params[2]
        C = params[3]
        
        # Hack workaround for values outside of the range
        # Need a cleaner solution
        if tsd <= 0:
            return 10000
        
        if sd <= 0:
            return 10000
        
        tu = 0
        u = 0
        
        # Apply some rules to the loop boundaries
        loop_bounds = src.refine_bounds(np.copy(self.X), tsd)

        ##### TO-DO
        if self.split == True:
            # Define an input to select whether to split or not
            X_segs, y_segs, dim_segs, bnds_segs = create_segments(np.copy(self.X), np.copy(self.x), np.copy(self.y), loop_bounds)
            # Loop through all the segments
            vals = np.array([])
            taus = np.array([])
    
            # Loop through the segments and apply the segmentation independently
            for j in range(0, len(X_segs)):
                print("Segment " + str(j + 1) + " from " + (str(len(X_segs))))
                X_seg = X_segs[j]
                y_seg = y_segs[j]
                dim_seg = dim_segs[j]

                # Define our solver for the internal loop
                f = linearTauSolver(X_seg, y_seg, A, tu, tsd, u, sd, C)
                
                # Run inner optimisation loop
                val, tau = self.inner_optimisation(dim_seg, f)
                
                print("------ Seg Likelihood: " + str(val))
                vals = np.append(vals, val)
                taus = np.append(taus, tau)
        else:
            # Define the solver function
            input_obj = linearTauSolver(np.copy(self.X), np.copy(self.y), A, tu, tsd, u, sd, C)
            # Run the inner optimisation.
            vals, taus = self.inner_optimisation(np.copy(self.X.shape), input_obj)

        
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
                                      [0, 1, 1, 0.3],
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      options ={"eps":0.2})

        elif method == "differential_evolution":
            self.summary = differential_evolution(self.inner_objective,
                                                  bounds, 
                                                  polish=True,
                                                  maxiter=5,
                                                  popsize=10)
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
            
    def inner_optimisation(self,  dim, f):
        # Create a user black box function
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
        bnds = ((-2,2),(0.00,3), (0.00, 1), (-1,1))
        
        # Run the outer loop to optimize the global parameters
        self.outer_optimization(method=method, bounds=bnds)
