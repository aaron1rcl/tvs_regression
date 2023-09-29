import numpy as np
import pandas as pd

from scipy.optimize import differential_evolution
import scipy.optimize as o
import statsmodels.api as sm

# Local libraries
import src.support as src
from src.model.continuousObjective import contSolver


class contTVSRModel():
    
    
    scaled = False
    rescaled = False
    shift_seq = None
    likelihood = None
    history = None
    family = "gaussian"
    cubic_df = None
    cubic_t_df = None

    
    def __init__(self, verbose=False):
        self.epoch = 0
        self.verbose = verbose
        
               
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
            self.x = src.standardise(self.x)
            self.y_min = np.min(self.y)
            self.y_max = np.max(self.y)
            self.y = src.standardise(self.y)
    
            # Scale the time dimension to have a max value of 1
            if max_t is None:
                # TO-DO: remove hardcoded 5% value and replace with something general
                #Make the max t 5% of the length by default
                self.scale_t = self.x.shape[0]*0.1
            else:
                self.scale_t = max_t
                
            self.scaled = True
            
    def rescale_input_and_params(self):
        ''' Rescales all parameters and results back to the original space '''
        if self.rescaled is False:
            self.x = (self.x*(self.x_max - self.x_min)) + self.x_min
            self.y = (self.y*(self.y_max - self.y_min)) + self.y_min
            
            # Update the summary object
            #self.summary.x[3] = (self.y_max - self.y_min)*self.summary.x[3] + self.y_min
            #self.summary.x[2] = self.summary.x[2]*(self.y_max - self.y_min)
            #self.summary.x[1] = self.summary.x[1]*self.scale_t
            
            scale_factor = (self.x_max - self.x_min) / (self.y_max - self.y_min)
            #self.summary.x[0] = self.summary.x[0]/scale_factor
            
            self.rescaled = True
    
    def read_params(self, params):
        coeff = params[:(self.cubic_df)]
        t_coeff = params[(self.cubic_df):(self.cubic_df + self.cubic_t_df)]*self.scale_t
        tsd = params[(self.cubic_df + self.cubic_t_df)]*self.scale_t
        sd = params[(self.cubic_df + self.cubic_t_df + 1)]

        return dict({"coeff":coeff,"t_coeff":t_coeff, "tsd":tsd, "sd":sd})
            
    def objective(self, params):
        
        '''    
            - Calculate the likelihood for a given set of parameters, including both a time and fit error 
        '''
        self.epoch = self.epoch + 1

        if self.verbose is True:
            print("Epoch: " , self.epoch)

        coeff = params[:(self.cubic_df)]
        t_coeff = params[(self.cubic_df):(self.cubic_df + self.cubic_t_df)]*self.scale_t
        tsd = params[(self.cubic_df + self.cubic_t_df)]*self.scale_t
        sd = params[(self.cubic_df + self.cubic_t_df + 1)]

        # TO-DO: Currently this is a quick hack workaround for values outside of the range
        # Need a clean general solution.
        if tsd <= 0:
            return 100000
        
        if sd <= 0:
            return 100000
        
        tu = 0
        u = 0
        

        l = contSolver(np.copy(self.x), np.copy(self.y), coeff,t_coeff, tu, tsd, u, sd, self.family)
        likelihood = l.objective_function()
        
        if self.verbose is True:
            print("Likelihood: " + str(round(likelihood ,3)))
        
        # Append the iteration results to a dataframe (to track the history)
        if self.history is not None:
            single_sample = pd.DataFrame({"coeff":[coeff],"t_coeff":[t_coeff],"tsd":tsd,"sd":sd,"likelihood":-1*likelihood,"e_l":l.e_l,"t_l":l.t_l}, index=[0])
            self.history = pd.concat([self.history, single_sample], axis=0)
        else:
            self.history = pd.DataFrame({"coeff":[coeff],"t_coeff":[t_coeff],"tsd":tsd,"sd":sd,"likelihood":-1*likelihood,"e_l":l.e_l,"t_l":l.t_l}, index=[0])
                
        return likelihood
    

    def optimize(self, method, bounds):
        # Run the outer optimisation loop
        if method == "differential_evolution":
            self.summary = differential_evolution(self.objective,
                                                  bounds, 
                                                  polish=True,
                                                  maxiter=50,
                                                  popsize=20)
        elif method == "L-BFGS-B":
            self.summary = o.minimize(self.objective, 
                                      np.repeat(0.1, 16),
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      options ={"maxiter":100000})
        else:   
            method = ""
            raise("Error: no compatible optimizer found")
                      
            
    def fit(self, x_train, y_train, method="L-BFGS-B", cubic_df = 3, cubic_t_df=3, max_t=None, family=None, iterations=1000):
        
        # Define the setting
        self.iter = iterations
        self.cubic_df = cubic_df
        self.cubic_t_df = cubic_t_df
        
        # Set the family
        if family is not None:
            self.family = family
        
        # Load the data into the object memory
        self.x = x_train
        self.y = y_train
        
        # Run a standard linear regression first for comparison
        self.basic_linear_regression()
        
        # Scale the input parameter space and save the transforms (makes optimisation easier)
        self.scale_inputs(max_t=max_t)
        
        # Assign bounds
        # Bounds can be fixed because the input space has been scaled
        bnds = ((-1,1),(-1,1), (-1,1), (-1,1), (-1,1), (-1,1), (-1,1),
            (-3,3),
        (-3,3),
        (-3,3),
        (-3,3),
        (-3,3),
        (-3,3),
        (-3,3),
        (0.00,  0.5), (0.00, 1))
        
        # Run the outer loop to optimize the global parameters
        self.optimize(method=method, bounds=bnds)
        
        # Rescale params and input sequences back to the originals
        self.rescale_input_and_params()

    def predict(self, h=0):
        params = self.summary.x
        params = self.read_params(params)

        f = contSolver(np.copy(self.x), np.copy(self.y), params['coeff'],params['t_coeff'],0, params['tsd'],0, params['sd'], self.family)
        return f.predict()

    def plot_time_basis(self):
        params = self.summary.x
        params = self.read_params(params)

        f = contSolver(np.copy(self.x), np.copy(self.y), params['coeff'],params['t_coeff'],0, params['tsd'],0, params['sd'], self.family)
        f.plot_time_basis()
