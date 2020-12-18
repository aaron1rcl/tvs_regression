import src.support as src
import numpy as np


class linearTauSolver:
    
    
    def __init__(self, X, y, A, tu, tsd, u, sd):
        self.X = X
        self.y = y
        self.A = A
        self.tu = tu
        self.tsd = tsd
        self.u = u
        self.sd = sd

    
    
    
    def objective_function(self, shift_seq):
        
        
        X_shift = src.shift_array(self.X, shift_seq=np.array(shift_seq, dtype="int"))

        X_shift = src.hor_mul(X_shift, self.A)
        
        
        # Create the prediction
        y_p = np.sum(X_shift, axis=0)
        
        # Calculate the y axis residuals
        res = self.y - y_p

        # Calculate the y axis log likelihood
        e_l = src.log_likelihood(x=res, u=self.u, sd=self.sd)
        # Calculate the t axis log likelihood
        # The t Axis is a discrete parameter - using continuous likelihood leads to the parameter shrinkage
        t_l = src.log_pmf_discrete(x=shift_seq, u=self.tu, sd=self.tsd)
        
        # Assume that the y and t errors are independent
    
        # Want to get the maximum likelihood! Hence the negative operator
        return -(e_l + t_l)