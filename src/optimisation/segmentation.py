import src.support as src
import numpy as np

 
def create_segments(X, x, y, bounds, tsd ):        
    loc, loc_diff = get_impulse_locations(x)
    splits = get_split_points(bounds, loc, loc_diff, tsd)
    # Split the dependent variable
    y_segments = np.split(y, splits)
    
    # Split the input matrix
    X_out = np.split(X, splits, axis=1)
    # Remove elements with all zeroes
    X_out = [z[z.any(axis=1)] for z in X_out]
    # Assign the segments to the object
    X_segments = X_out
    

    # Get the new updated object dimensions and bounds
    dimension = [z.shape for z in X_out]
    
    # Refine the bounds
    bounds = refine_segment_bounds(bounds, loc, dimension)
    
    return X_segments, y_segments, dimension, bounds

        
    
def get_impulse_locations(x):
        
    # Get the non zero element positions of the input vector
    loc = np.where(x != 0)[-1]
    loc_diff = np.diff(loc)
        
    return loc, loc_diff
    
    
def get_split_points(bounds, loc, loc_diff, tsd):
    

    # Find the spaces which are greater than twice the boundary (2 points movin toward each other)
    # To-Do: remove hardcoded hyperparameters
    # Set a min spacing for the split difference
    if tsd <= 2:
        split_diff = 2
    else:
        split_diff = tsd*3
    splits = np.where(loc_diff > np.min([tsd*3,20]))
    # Fit the array location where these splits occur
    split_locations = loc[splits] + np.round(loc_diff[splits]/2)
        
    return split_locations.astype("int16")
    
    
def refine_segment_bounds(bounds, loc, dimension):
    
    global_max_bound = np.max(bounds)
    global_min_bound = np.min(bounds)
        
    # Get the cumulative sum of the various split locations
    cum_sum = [0]
    for i in range(0, len(dimension)):
        cum_sum.append(dimension[i][1] + cum_sum[i])
    cum_sum = np.array(cum_sum)
    
    # Loop through the absolute values and refine the global boundaries
    #   into local values that change per split
        
    bounds = []
    for l in loc:
        diff = l - cum_sum
        min_bound = -1*np.min(diff[diff > 0])
        max_bound = -1*np.max(diff[diff < 0])
        # Check against the global max bound
        if max_bound > global_max_bound:
            max_bound = global_max_bound
        if min_bound < global_min_bound:
            min_bound = global_min_bound 
        # Add to the output array
        np.append(bounds, [min_bound, max_bound])
        
    return bounds
                
                
        
                
            
        
    
    
    
    
    
    
                
    
