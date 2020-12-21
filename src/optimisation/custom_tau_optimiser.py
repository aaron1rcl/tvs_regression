import numpy as np
# Define the optimizer

# start with an initial vector [0]*16
# loop over all the elements and randomly add or subtract 1 to 2 elements
# if the result has a higher likelihood -> keep as best parameter
# if the result has a lower likelihood -> reject and resample

def tau_optimiser(init, f, max_iterations, max_dependency=4):
    
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
            flip = np.round(np.random.randn(1)*f.tsd)
            proposal[sel] = proposal[sel] + flip
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