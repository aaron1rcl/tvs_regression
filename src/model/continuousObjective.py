import src.support as src
import numpy as np
import src.cont_support as cont_src
from patsy import dmatrix
import matplotlib.pyplot as plt


class contSolver():

    def __init__(self, X, y, coeff, t_coeff, tu, tsd, u, sd, family):
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.coeff = coeff
        self.t_coeff = t_coeff
        self.tu = tu
        self.tsd = tsd
        self.u = u
        self.sd = sd
        self.family = family
        self.e_l = None
        self.t_l = None
        

    def f_x(self, x_in):
        
        x_in = x_in[~np.isnan(x_in)]
        # Apply a cubic basis function to the x value
        spline_basis = dmatrix("cr(x, df=6)", {"x": x_in})
        basis_coeff = self.coeff 
        return np.dot(spline_basis, basis_coeff)
    

    def objective_function(self):

        t_range = range(0, len(self.X))
        # Create the cubic spline basis functions for modelling time shifts

        cubic_splines = dmatrix("bs(x, df=6, degree=3, include_intercept=False) - 1", {"x": t_range})

        basis_coeff = self.t_coeff

        # Sum of the splines for the fit
        t_fit = np.dot(cubic_splines, basis_coeff)

        shift_fit, shift_seq = cont_src.fit_random_walk(t_fit, self.tsd)

        # If there is a shift, apply it
        x_shift, iterations = cont_src.apply_perturbations(self.X, perturbations=shift_seq, sr=10)
        # Apply the coefficient

        # Calculate the fit
        y_p = self.f_x(x_shift)

        res = self.y[:-1] - y_p
        # Calculate the y axis log likelihood
        e_l = src.log_likelihood(x=res, u=self.u, sd=self.sd)

        # Calculate the t axis log likelihood
        # The t Axis is a discrete parameter - using continuous likelihood leads to the parameter shrinkage
        t_l = src.log_pmf_discrete(x=shift_seq[:iterations], u=self.tu, sd=self.tsd, family="gaussian")
        
        self.e_l = e_l
        self.t_l = t_l

        # Want to get the maximum likelihood! Hence the negative multiplier
        return -(e_l + t_l)
    

    def plot_time_basis(self):
        t_range = range(0, len(self.X))
        # Create the cubic spline basis functions for modelling time shifts
        cubic_splines = dmatrix("bs(x, df=6, degree=3, include_intercept=False) - 1", {"x": t_range})

        basis_coeff = self.t_coeff

        # Sum of the splines for the fit
        t_fit = np.dot(cubic_splines, basis_coeff)
        plt.plot(t_fit)
        plt.show()
    
    def predict(self):
        
        ''' 
        Produces a 'fit' for a given set of coefficients.
        '''
        t_range = range(0, len(self.X))
        # Create the cubic spline basis functions for modelling time shifts
        cubic_splines = dmatrix("bs(x, df=6, degree=3, include_intercept=False) - 1", {"x": t_range})

        basis_coeff = self.t_coeff

        # Sum of the splines for the fit
        t_fit = np.dot(cubic_splines, basis_coeff)
        
        
        shift_fit, shift_seq = cont_src.fit_random_walk(t_fit, self.tsd)
        
        # If there is a shift, apply it
        x_shift, iterations = cont_src.apply_perturbations(self.X, perturbations=shift_seq, sr=10)
        # Apply the coefficient
        
        # Calculate the fit
        y_p = self.f_x(x_shift)

        return y_p
    

    def forecast_time_shift(time_fit, n,sd):
        ''' Forecast future time shifts based on inferred parameters'''

        lower = -2*sd*np.sqrt(np.arange(1,n)) + time_fit[len(time_fit)-1]
        u = np.repeat(0, n) + time_fit[len(time_fit)-1]
        upper = 2*sd*np.sqrt(np.arange(1,n)) + time_fit[len(time_fit)-1]


        return lower, u, upper


lower, u, upper = forecast_time_shift(t_fit[:574], 200, 6.32)