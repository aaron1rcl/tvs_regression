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
    
    scaled = False
    rescaled = False
    shift_seq = None
    likelihood = None
    history = None
    family = "gaussian"

    
    def __init__(self, verbose=False):
        self.epoch = 0
        
               
    def basic_linear_regression(self):
        
        ''' Runs a linear regression (y = Ax + C) with statsmodels 
            The linear regression model can be useful for comparison.
        
        '''
        # Statsmodels comparison (no intercept, same as above)
        X_in = np.copy(self.x)
        X_in = sm.add_constant(X_in)
        mod = sm.OLS(self.y, X_in)
        res = mod.fit()
        self.basic_lin_summary = res.summary()
        self.min_A = res.params[0]
        self.max_sd = np.std(res.predict(X_in)  - self.y)
        
        
    def scale_inputs(self, max_t):
        
        ''' Scales the input parameter space to a range between zero and 1.
        
            #TO-DO:
                - deal with the fixed values of scale_t
                - allow for input to f_0  
        '''
        if self.scaled is False:
            self.x_min = np.min(self.x)
            self.x_max = np.max(self.x)
            self.x = src.standardise_f0(self.x, f_0=0)
            self.y_min = np.min(self.y)
            self.y_max = np.max(self.y)
            self.y = src.standardise(self.y)
    
            # Scale the time dimension to have a max value of 1
            if max_t is None:
                # TO-DO: remove hardcoded 5% value and replace with something general
                #Make the max t 5% of the length by default
                self.scale_t = self.x.shape[0]*0.05
            else:
                self.scale_t = max_t
                
            self.scaled = True
            
    def rescale_input_and_params(self):
        ''' Rescales all parameters and results back to the original space '''
        if self.rescaled is False:
            self.x = (self.x*(self.x_max - self.x_min)) + self.x_min
            self.y = (self.y*(self.y_max - self.y_min)) + self.y_min
            
            # Update the summary object
            self.summary.x[3] = (self.y_max - self.y_min)*self.summary.x[3] + self.y_min
            self.summary.x[2] = self.summary.x[2]*(self.y_max - self.y_min)
            self.summary.x[1] = self.summary.x[1]*self.scale_t
            
            scale_factor = (self.x_max - self.x_min) / (self.y_max - self.y_min)
            self.summary.x[0] = self.summary.x[0]/scale_factor
            
            self.rescaled = True
            
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
            self.history: a record of all likelihood optimisation steps. this helps to see the full solutions space for analysis
            
        '''
        self.epoch = self.epoch + 1
        print("Epoch: " , self.epoch)   
        A = params[0]
        tsd = params[1]*self.scale_t
        sd = params[2]
        C = params[3]
        
        # TO-DO: Currently this is a quick hack workaround for values outside of the range
        # Need a clean general solution.
        if tsd <= 0:
            return 10000
        
        if sd <= 0:
            return 10000
        
        tu = 0
        u = 0
        
        # Apply some rules to the loop boundaries
        loop_bounds = src.refine_bounds(np.copy(self.X), tsd)

        #Either split into orthogonal segments or not
        if self.split == True:
            # Define an input to select whether to split or not
            X_segs, y_segs, dim_segs, bnds_segs = create_segments(np.copy(self.X), np.copy(self.x), np.copy(self.y), loop_bounds, tsd)
            self.segs = dim_segs
 
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
                f = linearTauSolver(X_seg, y_seg, A, tu, tsd, u, sd, C, self.family)
                
                # Run inner optimisation loop
                val, tau = self.inner_optimisation(dim_seg, f)
                
                print("------ Seg Likelihood: " + str(val))
                # Add the segment values a list
                vals = np.append(vals, val)
                taus = np.append(taus, tau)
        else:
            # Define the solver function
            input_obj = linearTauSolver(np.copy(self.X), np.copy(self.y), A, tu, tsd, u, sd, C, self.family)
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
        
        # Append the iteration results to a dataframe (to track the history)
        if self.history is not None:
            single_sample = pd.DataFrame({"A":A,"tsd":tsd,"sd":sd,"likelihood":-1*max_likelihood}, index=[0])
            self.history = pd.concat([self.history, single_sample], axis=0)
        else:
            self.history = pd.DataFrame({"A":A,"tsd":tsd,"sd":sd,"likelihood":-1*max_likelihood}, index=[0])
                
        return max_likelihood
    

    def outer_optimization(self, method, bounds):
        # Run the outer optimisation loop
        if method == "L-BFGS-B":
            self.summary = o.minimize(self.inner_objective, 
                                      [0, 1, 1, 0],
                                      method='L-BFGS-B',
                                      bounds=bounds)

        elif method == "differential_evolution":
            self.summary = differential_evolution(self.inner_objective,
                                                  bounds, 
                                                  polish=True,
                                                  maxiter=20,
                                                  popsize=10)
        else:   
            raise("Error: no compatible optimizer found")
            
    def inner_optimisation(self,  dim, f):
        # Run the inner optimisation algorithm and retrieve the max likelihood
        val, tau = tau_optimiser([0]*dim[0], f, self.iter, 3, self.family)

        return val, tau
            
            
    def fit(self, x_train, y_train, method="L-BFGS-B", split=True, max_t=None, family=None, iterations=1000):
        
        # Define the setting
        self.split = split
        self.iter = iterations
        
        # Set the family
        if family is not None:
            self.family = family
        
        # Load the data into the object memory and decompose the vector
        self.x = x_train
        self.y = y_train
        
        # Run a standard linear regression first for comparison
        self.basic_linear_regression()
        
        # Scale the input parameter space and save the transforms (makes optimisation easier)
        self.scale_inputs(max_t=max_t)
        
        # Decompose the input series
        self.X, self.X_bounds = src.decompose_vector(self.x, return_bounds=True)
        self.dimension = self.X.shape[0]

        

        
        # Assign bounds
        # Bounds can be fixed because the input space has been scaled
        bnds = ((-2,2),(0.00,1), (0.00, 1), (-1,1))
        
        # Run the outer loop to optimize the global parameters
        self.outer_optimization(method=method, bounds=bnds)
        
        # Rescale params and input sequences back to the originals
        self.rescale_input_and_params()
