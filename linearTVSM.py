import src.support as src
import numpy as np
import scipy.optimize as o
from linearObjective import linearTauSolver
import rbfopt
import os

class linearTVSRModel:
    
    def __init__(self, rbf_settings, verbose=False):

        
        # Establish the settings
        if rbf_settings is None:
            self.settings = rbfopt.RbfoptSettings(max_evaluations=100, 
                                 max_noisy_evaluations=10,
                                 minlp_solver_path='/Users/aaronpickering/Desktop/bonmin-osx/bonmin',
                                 print_solver_output=False)
        
        # Create a dev null output to silence the print output
        if verbose is False:
            self.dev_null = open(os.devnull, 'w')
    

    def inner_objective(self, params):
                     
        print(params)       
        A = params[0]
        tsd = params[1]
        sd = params[2]
        tu = 0
        u = 0
        f = linearTauSolver(self.X, self.y, A, tu, tsd, u, sd)
        
        # Create a user black box function
        bb = rbfopt.RbfoptUserBlackBox(self.dimension, 
                                       np.array([-2] * self.dimension), 
                                       np.array([2] * self.dimension),
                                       np.array(['I']*self.dimension), 
                                       f.objective_function)

        # Crreate the algorithm from the black box and settings
        alg = rbfopt.RbfoptAlgorithm(self.settings,bb)
        alg.set_output_stream(self.dev_null)
        
        val, x_out, itercount, evalcount, fast_evalcount = alg.optimize()
        print("Likelihood: " + str(round(val,1)))
        return val
    
    def fit(self, x_train, y_train):
        self.x = x_train
        self.X = src.decompose_vector(self.x)
        self.dimension = self.X.shape[0]
        self.y = y_train
            
        self.summary = o.minimize(self.inner_objective, [0, 2, 2], method='L-BFGS-B')
        

        
        