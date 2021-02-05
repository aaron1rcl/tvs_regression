import src.support as src
import numpy as np


class linearTauSolver:
    
    
    def __init__(self, X, y, A, tu, tsd, u, sd, C, family):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.A = A
        self.tu = tu
        self.tsd = tsd
        self.u = u
        self.sd = sd
        self.C = C
        self.family = family
        
    
    
    def objective_function(self, shift_seq):
        ''' Calculates the likelihood for a given set of X, y, A, C and shift seq 
        
            The likehood is the sum of the likelihood in the time domain (t-axis)
            and the prediction error (y-axis)
        '''
        
        # If there is a shift, apply it
        if not all(shift_seq == 0):
            X_shift = src.shift_array(np.copy(self.X), shift_seq=np.array(shift_seq, dtype="int"))
        else:
            X_shift = np.copy(self.X)
        
        # Apply the coefficient
        X_shift = src.hor_mul(np.copy(X_shift), self.A)
        
        # Create the prediction
        y_p = np.sum(X_shift, axis=0) + self.C

        
        # Calculate the y axis residuals
        res = self.y - y_p

        # Calculate the y axis log likelihood
        e_l = src.log_likelihood(x=res, u=self.u, sd=self.sd)
        # Calculate the t axis log likelihood
        # The t Axis is a discrete parameter - using continuous likelihood leads to the parameter shrinkage
        t_l = src.log_pmf_discrete(x=shift_seq, u=self.tu, sd=self.tsd, family=self.family)
        

        # Want to get the maximum likelihood! Hence the negative multiplier
        return -(e_l + t_l)
    


    
    
    def predict(self, shift_seq):
        
        ''' Produces a 'prediction' for a given shift sequence, A and C.
        
            The results is not a true prediction because the shift sequence in
            the time domain in stochastic, but it can be used for assessing the fit.
            Additionally, the last predicted shift might be beneficial for true predictions, if the input
            has a lagged effect.
        
        '''
        
        # If there is a shift, apply it
        if not all(shift_seq == 0):
            X_shift = src.shift_array(np.copy(self.X), shift_seq=np.array(shift_seq, dtype="int"))
        else:
            X_shift = np.copy(self.X)
        
        # Apply the coefficient
        X_shift = src.hor_mul(np.copy(X_shift), self.A)
        
        # Create the prediction
        y_p = np.sum(X_shift, axis=0) + self.C

        return y_p