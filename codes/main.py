# Contains the adjacency learning models

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import softmax
from helper import *
from itertools import *
from datetime import datetime


### Talk with Misi

# Model 1: co_occurence_model

def co_occurence_model(solution_array, 

                       alpha=0.1, 
                       n_prim=6):

    """
    Learn the adjacency matrix using observed frequencies of chunks in the solutions array. 
    Update after every observation 

    Model parameter/s:
    alpha: learning rate, should be within [0,1] 

    1. First they start with uniform probabilities
    2. Then the agent starts observing one trial/silhouette at a time
    3. At each set of observation, the probabilities are recalculated
    4. The computation ends either after observing all "silhouettes" or by some number of trials
    """

    # Input
    # solution_array: (a matrix of n_sils x 1 (list of building blockd)) 
    # prior: (a matrix of n_prim x n_prim)
    # alpha: (scalar) learning rate
    # n_prim: (scalar) the number of primitive building blocks
    # t: (scalar) number of observations to learn

    # Output
    # posterior: (dictionary)

    # Initialization:

    # Dictionary to keep trial by trial posterior

    posterior_dict = {}

    # a check that alpha in [0,1]

    # if no prior is provided
    prior = np.ones((n_prim,n_prim), float)*(12/math.comb(6,4))
    np.fill_diagonal(prior,0)

    for k, sil in enumerate(solution_array):
        # posterior
        posterior = prior

        # list of pairs of blocks in a silhouette
        tuple_list = list(combinations(sil,2)) #2 since we're concerned with chunks composed of 2 blocks, list of tuples

        for i in np.arange(n_prim):
            for j in np.arange(n_prim):
                if (i,j) in tuple_list:
                    posterior[i,j] = posterior[i,j] + alpha*(1-posterior[i,j]) 
                    posterior[j,i] = posterior[j,i] + alpha*(1-posterior[j,i]) # they should have similar values, since this is supposed to be symmetric
                else:
                    posterior[i,j] = posterior[i,j] - alpha*posterior[i,j] # maybe move back to uniform prior rather than 0? ; meeting with Philipp
                    posterior[j,i] = posterior[j,i] - alpha*posterior[j,i] # they should have similar values, since this is supposed to be symmetric

        posterior_dict[f"{k}"] = posterior
        prior = posterior

    return posterior_dict

### Talk with Philipp

# # Model 2: connectivity_model

# def connectivity_model(solution_array, 
#                        alpha=0.1, 
#                        n_prim=6, 
#                        t=150):

#     """
#     Learn the adjacency matrix using observed frequencies of connected chunks in the solutions array. 
#     Update after every observation 

#     Model parameter/s:
#     alpha: learning rate, should be within [0,1] 

#     1. First they start with uniform probabilities
#     2. Then the agent starts observing one trial/silhouette at a time
#     3. At each set of observation, the probabilities are recalculated
#     4. The computation ends either after observing all "silhouettes" or by some number of trials
#     """

#     # Input
#     # solution_array: (a matrix of n_sils x n_blocks) 
#     # prior: (a matrix of n_prim x n_prim)
#     # alpha: (scalar) learning rate
#     # n_prim: (scalar) the number of primitive building blocks
#     # t: (scalar) number of observations to learn

#     # Output
#     # posterior: (matrix of n_prim x n_prim dimensions)

#     # Initialization:

#     # Dictionary to keep trial by trial posterior

#     posterior_dict = {}

#     # a check that alpha in [0,1]

#     # if no prior is provided
#     prior = np.ones((n_prim,n_prim), float)*(12/math.comb(6,4))
#     np.fill_diagonal(prior,0)



#     # do not exhaust all stimulus/silhouettes, just learn from t-silhouettes
#     solution_array = solution_array[:t,:]

#     for sil in np.arange(solution_array.shape[0]):

#         # posterior
#         posterior = prior

#         # collect the pairs of connected blocks in the silhouette
#         # this is a dictionary, the keys are tuples (pairs) of the connected blocks present in the silhouette
#         # while the values are the number of their occurence, I only need to access the keys
#         tuple_dict = tuple_counter(solution_array[sil]) # collects the tuple per row/observation

#         for i in np.arange(n_prim):
#             for j in np.arange(n_prim):
#                 if (i,j) in tuple_dict.keys():
#                     posterior[i,j] = posterior[i,j] + alpha*(1-posterior[i,j]) # if the connection is observed, move closer to 1
#                     posterior[j,i] = posterior[j,i] + alpha*(1-posterior[j,i]) # they should have similar values, since this is supposed to be symmetric
#                 else:
#                     posterior[i,j] = posterior[i,j] - alpha*posterior[i,j] # if not, more closer to 0, 
#                                                                            # maybe move back to uniform prior rather than 0? ; meeting with Philipp
#                                                                            # this just asks, should the prior be a lower bound? idts 
#                     posterior[j,i] = posterior[j,i] - alpha*posterior[j,i] # they should have similar values, since this is supposed to be symmetric

#         posterior_dict[f"{sil}"] = posterior
#         prior = posterior

#     return posterior_dict


######################## MODIFIED ####################

### Talk with Philipp

# matmed: matrix method

# Model 2: connectivity_matmed

def connectivity_matmed (solution_array, 
                        alpha=0.1, 
                        n_prim=6):

    """
    Learn the adjacency matrix using observed frequencies of chunks in the solutions array. 
    Update after every observation 

    Model parameter/s:
    alpha: learning rate, should be within [0,1] 

    1. First they start with uniform probabilities
    2. Then the agent starts observing one trial/silhouette at a time
    3. At each set of observation, the probabilities are recalculated
    4. The computation ends either after observing all "silhouettes" or by some number of trials
    """

    # Input
    # solution_array: (array of the "coordinates of to-be-drawn buiding blocks (format: [x,y,building block id])") 
    # prior: (a matrix of n_prim x n_prim)
    # alpha: (scalar) learning rate
    # n_prim: (scalar) the number of primitive building blocks

    # Output
    # posterior: (matrix of n_prim x n_prim dimensions)

    # Initialization:

    # Dictionary to keep trial by trial posterior
    posterior_dict = {}

    # if no prior is provided
    prior = np.ones((n_prim,n_prim), float)*(12/math.comb(6,4))
    np.fill_diagonal(prior,0)

    # Global var
    ones = np.ones((n_prim,n_prim))
    ones_inv = np.ones((n_prim,n_prim))
    np.fill_diagonal(ones_inv, 0)

    for i, sil in enumerate(solution_array):
        # posterior
        posterior = prior
        BBs = make_ShapeCoord(sil)
        adjacency,BB_built,_ = draw_silh(BBs)
        
        # positive update
        posterior = posterior + ((ones-posterior) * (alpha*adjacency))

        # negative update
        posterior = posterior - (posterior * (alpha*(ones_inv-adjacency)))

        posterior_dict[f"{i}"] = [posterior, adjacency]
        prior = posterior

    return posterior_dict



def connectivity_matmed_key_sil (silhouette_array, 
                                dict_keys, # trial_ids
                                session_id, # session_id /participant  
                                alpha=0.1, 
                                n_prim=6):

    """
    Learn the adjacency matrix using observed frequencies of chunks in the solutions array. 
    Update after every observation 

    Model parameter/s:
    alpha: learning rate, should be within [0,1] 

    1. First they start with uniform probabilities
    2. Then the agent starts observing one trial/silhouette at a time
    3. At each set of observation, the probabilities are recalculated
    4. The computation ends either after observing all "silhouettes" or by some number of trials
    """

    # Input
    # silhouette_array: (array of the "coordinates of to-be-drawn buiding blocks (format: [x,y,building block id])") 
    # dict_keys: should be trial id
    # session_id: scalar
    # alpha: (scalar) learning rate
    # n_prim: (scalar) the number of primitive building blocks

    # Output
    # posterior: (matrix of n_prim x n_prim dimensions)

    # assumption both silhouette_array and dict_keys are ordered

    # Initialization:

    # Dictionary to keep trial by trial posterior
    posterior_dict = {}

    # if no prior is provided
    prior = np.ones((n_prim,n_prim), float)*(12/math.comb(6,4))
    np.fill_diagonal(prior,0)

    # Global var
    ones = np.ones((n_prim,n_prim))
    ones_inv = np.ones((n_prim,n_prim))
    np.fill_diagonal(ones_inv, 0)

    for i, sil in enumerate(silhouette_array):
        # posterior
        posterior = prior
        BBs = make_ShapeCoord(sil)
        adjacency,BB_built,_ = draw_silh(BBs)
        
        # positive update
        posterior = posterior + ((ones-posterior) * (alpha*adjacency))

        # negative update
        posterior = posterior - (posterior * (alpha*(ones_inv-adjacency)))

        posterior_dict[f"{dict_keys[i]}"] = [posterior, session_id] #maybe posterior, session_id
        prior = posterior

    return posterior_dict