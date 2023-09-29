import numpy as np

# Establish the equation for a Piecewise Linear Spline
def piecewise_linear(x :np.array, k :int):
    ''' Linear spline is 0 for all x < k
        and Bx for all x >= k
     For the basis - set B = 1 and let the model learn the parameter'''
    y = x - k
    y = np.where(x < 0, 0, y)
    y[[x < k for x in x]] = 0
    return y

def create_basis(x, k):
    ''' Create a linear spline basis'''
    b = np.empty((len(x), len(k)))
    for i in range(0,len(k)):
        b_i = piecewise_linear(x, k[i])
        b[:,i] = b_i
    return b


def build_piecewise_linear(length, weights, n_basis):
    ''' Creates a piecewise linear function with n_basis splines, 
        equally spaced across the vector length. The first basis is = 0 '''

    # Create basis functions
    k = np.linspace(0,length, num=n_basis)
    t = np.arange(0, length)
    b = create_basis(t, k = k)

    # Dot product between weights and basis functions
    y = np.dot(b, weights)

    return y