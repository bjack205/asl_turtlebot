# EXPLORER

#!/usr/bin/env python

import numpy as np
from grids import StochOccupancyGrid2D
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches

###

###

def find_closest_unexplored(rmap, location, width, height, resolution):
    known = np.zeros(rmap.shape)
    known[np.where(rmap == 0)] = 1
    known_dilated = scipy.ndimage.morphology.binary_dilation(known)
    test_cases = known_dilated - known
    
    distance = np.zeros(rmap.shape)
    x_ind, y_ind = location_ind(location, width, height, resolution)
    for i in range(rmap.shape[0]):
        for j in range(rmap.shape[1]):
            if test_cases[i,j] == 1 and rmap[i,j] != 1:
                distance[i,j] = np.sqrt((x_ind - i)**2 + (y_ind -j)**2)
            else:
                distance[i,j] = np.infty
    closest_ind = np.unravel_index(distance.argmin(), distance.shape)
    
    return ind_to_cartesian(closest_ind, width, height, resolution)
    
def find_explore_location(rmap, location, width, height, resolution):
    unexplored = np.zeros(rmap.shape) # binary: explored (0), unexplored (1) matrix
    unexplored[np.where(rmap==-1)] = 1 # set all unexplored regions to 1
    lbl = scipy.ndimage.label(unexplored)[0] # automatically labels "clusters" with different values (1,2,3...)
    com_options_ind = scipy.ndimage.measurements.center_of_mass(unexplored, lbl,np.unique(lbl)[1:]) # find center(s) of mass

    #### if no centroid is found, simply go to closest unexplored location
    if com_options_ind == []:
        #print('no centroid found: going to nearest frontier')
        return find_closest_unexplored(rmap, location, width, height, resolution)
    ###
    
    ### if options exist, select the cluster with the most area
    else:
        counts = np.bincount(lbl[np.where(lbl != 0)].ravel())
        best_idx = np.argmax(counts)-1
        com = ind_to_cartesian(com_options_ind[best_idx], width, height, resolution)
        #print('Number of centroids:')
        #print(len(counts)-1)
        #print('Current centroid:')
        #print(com)
        
        # if center of mass is within the unexplored region, new explore location has been found
        if unexplored[location_ind(com, width, height, resolution)] == 1: 
            #print('best center of mass works')
            #print('Current centroid:')
            #print(com)
            return com
        else:
            # if center of mass is outside unexplored region, we try splitting cluster via erosion to find new com
            # NOTE: may be better to try other calculated com's from above instead of directly trying erosion
            temp = np.copy(unexplored)
            while unexplored[location_ind(com, width, height, resolution)] != 1:
                print('erosion...')
                temp = scipy.ndimage.morphology.binary_erosion(temp) # erode unexplored region
                lbl = scipy.ndimage.label(temp)[0] # automatically labels "clusters" with different values (1,2,3...)
                com_options_ind = scipy.ndimage.measurements.center_of_mass(temp, lbl,np.unique(lbl)[1:]) # find new center(s) of mass

                # if no centroid is found, simply go to closest unexplored location
                if com_options_ind == []:
                    #print('frontier exploration broken')
                    return find_closest_unexplored(rmap, location, width, height, resolution)

                # if multiple clusters have been found, select the one with the most area
                else:
                    counts = np.bincount(lbl[np.where(lbl != 0)].ravel())
                    best_idx = np.argmax(counts)-1
                    com = ind_to_cartesian(com_options_ind[best_idx], width, height, resolution)
                    if unexplored[location_ind(com, width, height, resolution)] == 1: 
                        #print('Current centroid:')
                        #print(com)
                        return com  
                        
def snap_to_grid(x, width, height, resolution):
    return (resolution*round(x[0]/resolution), resolution*round(x[1]/resolution))

def location_ind(x, width, height, resolution):
    return (int(round(x[0]/resolution)), int(round(x[1]/resolution)))

def ind_to_cartesian(ind, width, height, resolution):
    x_car = resolution*ind[0]
    y_car = resolution*ind[1]
    return (x_car, y_car)

