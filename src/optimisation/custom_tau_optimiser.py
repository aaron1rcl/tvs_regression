import numpy as np
# Define the optimizer

# start with an initial vector [0]*16
# loop over all the elements and randomly add or subtract 1 to 2 elements
# if the result has a higher likelihood -> keep as best parameter
# if the result has a lower likelihood -> reject and resample

def tau_optimiser(init, f, max_iterations, max_dependency=4):
    
    i_l = f.objective_function(np.copy(init))
    
    assert(max_iterations is not None)
    
    for i in range(0, max_iterations):
        
        selection = np.random.choice(np.arange(0,len(init)), size=max_dependency, replace=False)
        
        # Generate a proposal
        proposal = np.copy(init)
        for sel in selection:
            flip = np.round(np.random.randn(1)*f.tsd)
            proposal[sel] = proposal[sel] + flip
        # Check whether the proposal is better than the initial
        p_l = f.objective_function(np.copy(proposal))
        # If so jump to the proposal
        if p_l <= i_l:
            i_l = np.copy(p_l)
            init = np.copy(proposal)
            
    return i_l, init