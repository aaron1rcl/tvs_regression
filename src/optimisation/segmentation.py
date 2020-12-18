import src.support as src
import numpy as np


test = dataSegments(X, x, bounds)
test.refine_bounds()

class dataSegments(object):
    

    def __init__(self, X, x, bounds):        
        self.loc, self.loc_diff = self.get_impulse_locations(x)
        splits = self.get_split_points(bounds)
        
        X_out = np.split(X, splits, axis=1)
        
        # Remove elements with all zeroes
        X_out = [z[z.any(axis=1)] for z in X_out]
        
        # Assign the segments to the object
        self.segments = X_out
        # Get the new updated object dimensions and bounds
        self.dimension = [z.shape for z in X_out]

        
    
    def get_impulse_locations(self, x):
        
        # Get the non zero element positions of the input vector
        loc = np.where(x != 0)[-1]
        loc_diff = np.diff(loc)
        
        return loc, loc_diff
    
    
    def get_split_points(self, bounds):
        # Extract the maximum boundary
        self.global_max_bound = np.max(bounds)
        self.global_min_bound = np.min(bounds)
        # Find the spaces which are greater than twice the boundary (2 points movin toward each other)
        splits = np.where(self.loc_diff > np.max(np.abs(self.global_min_bound), self.global_max_bound)*2)
        # Fit the array location where these splits occur
        split_locations = self.loc[splits] + np.round(self.loc_diff[splits]/2)
        
        return split_locations.astype("int16")
    
    
    def refine_bounds(self):
        
        # Get the cumulative sum of the various split locations
        cum_sum = [0]
        for i in range(0, len(self.dimension)):
                cum_sum.append(self.dimension[i][1] + cum_sum[i])
        cum_sum = np.array(cum_sum)
        
        # Loop through the absolute values and refine the global boundaries
        #   into local values that change per split
        
        self.bounds = []
        for l in self.loc:
            diff = l - cum_sum
            print(diff)
            min_bound = -1*np.min(diff[diff > 0])
            max_bound = -1*np.max(diff[diff < 0])
            # Check against the global max bound
            if max_bound > self.global_max_bound:
                max_bound = self.global_max_bound
            if min_bound < self.global_min_bound:
                min_bound = self.global_min_bound 
            # Add to the output array
            self.bounds.append((min_bound, max_bound))
                
                
        
                
            
        
    
    
    
    
    
    
                
    