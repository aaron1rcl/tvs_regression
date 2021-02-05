import numpy as np


def tau_optimiser(init, f, max_iterations, max_dependency, family):
  
    '''
    Optimizer for best Tau Selection
       
     Start with an initial vector [0]*16
     loop over all the elements and randomly add or subtract 1 to 2 elements
     if the result has a higher likelihood -> keep as best parameter
     if the result has a lower likelihood -> reject and resample
    
    Parameters
    ----------
    init : numpy array, length = number of impulses
        the intial value vector of shifts (taus) to optimize
    f : linearTauSolver
        the objective function object to optimize
    max_iterations : int
        Max number of iterations for the algorithm. > 1000 is effective
    max_dependency : int
        The max number of impulses which are dependent on each other
        Higher values are harder to optimize but might be necessary for the problem

    Returns
    -------
    i_l : float
        The maximum likelihood value for the inner loop
    init : numpy array
        The estimated tau shifts for the maximum likelihood value
    '''
    
    # If splitting is set, there will be shorter segments,
    #   potentially with less impulses than the max dependency value
    #   - therefore check and set appropriately
    if max_dependency > len(init):
        max_dependency = len(init)
    
    # Define a initial set of tau values - mean of the current tau distribution
    i_l = f.objective_function(np.copy(init))
    
    assert(max_iterations is not None)
    
    # Do this loop for n iterations 
    for i in range(0, max_iterations):
        

        # Randomly select a number of impulses defined by the max_dependency parameters
        # TO-DO. Define the max dependency parameter from the data
        #   We can use the space between impulses and the tau distribution sd to estimate the likely crossover
        # TO-DO - make impulses closer together more likely to be selected than a straight random draw
        selection = np.random.choice(np.arange(0,len(init)), size=max_dependency, replace=False)
        
        # For the randomly selected impulses, generate a shift proposal by adding a random value of tau
        # The random tau selection controlled by the tau sd given on line 30
        proposal = np.copy(init)
        for sel in selection:
            if family == "gaussian":
                flip = np.round(np.random.randn(1)*f.tsd)
            elif family == "poisson":
                flip = np.random.poisson(f.tsd)
            else:
                raise("Error: family not found")
            proposal[sel] = flip
        # Check whether the proposal is better than the initial
        
        p_l = f.objective_function(np.copy(proposal))
        
        # If its better, keep it as the 'best' and continue looping
        # If so jump to the proposal
        # Else ignore the proposal
        if p_l <= i_l:
            i_l = np.copy(p_l)
            init = np.copy(proposal)
            
        # This way the tau optimisation iterates to better and better solutions
        # This is similar to MCMC sampling or SANN except its greedy
        #   in that proposals are never rejected if they are better
            
    return i_l, init