import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import softmax
from helper import *
from itertools import *
from datetime import datetime

# contains the adjacency learning models


# Load data

final_Sil = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\final_Sil20220803-104805.npy")
solutions_1 = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\solutions_20220803-104805.npy")
coord_Blocks = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\coord_Blocks_20220803-104805.npy")
adjacency = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\adjacency\\Adjacency_20220803-104805.npy")

### Talk with Misi

# Model 1: co_occurence_model

def co_occurence_model(solution_array, 
                       t=150,
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
    # solution_array: (a matrix of n_sils x n_blocks) 
    # prior: (a matrix of n_prim x n_prim)
    # alpha: (scalar) learning rate
    # n_prim: (scalar) the number of primitive building blocks
    # t: (scalar) number of observations to learn

    # Output
    # posterior: (matrix of n_prim x n_prim dimensions)

    # Initialization:

    # a check that alpha in [0,1]

    # if no prior is provided
    prior = np.ones((n_prim,n_prim), float)*(12/math.comb(6,4))
    np.fill_diagonal(prior,0)

    # posterior
    posterior = prior

    # do not exhaust all stimulus/silhouettes, just learn from t-silhouettes
    solution_array = solution_array[:t,:]

    for sil in np.arange(solution_array.shape[0]):

        # list of pairs of blocks in a silhouette
        tuple_list = list(combinations(solution_array[sil],2)) #2 since we're concerned with chunks composed of 2 blocks, list of tuples

        for i in np.arange(n_prim):
            for j in np.arange(n_prim):
                if (i,j) in tuple_list:
                    posterior[i,j] = posterior[i,j] + alpha*(1-posterior[i,j])
                else:
                    posterior[i,j] = posterior[i,j] - alpha*posterior[i,j] #maybe move back to uniform prior rather than 0? ; meeting with Philipp

    return posterior

# posterior = co_occurence_model(solutions_1)

# # print("adjacency: ", adjacency)
# # #print("prior: ", prior)
# # print("posterior: ", np.around(posterior, 4))

# np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\learned_adjacency\\posterior_trial.npy", posterior)

### Talk with Philipp

# Model 2: connectivity_model

def connectivity_model(solution_array, alpha=0.1, n_prim=6, t=150):

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
    # solution_array: (a matrix of n_sils x n_blocks) 
    # prior: (a matrix of n_prim x n_prim)
    # alpha: (scalar) learning rate
    # n_prim: (scalar) the number of primitive building blocks
    # t: (scalar) number of observations to learn

    # Output
    # posterior: (matrix of n_prim x n_prim dimensions)

    # Initialization:

    # a check that alpha in [0,1]

    # if no prior is provided
    prior = np.ones((n_prim,n_prim), float)*(12/math.comb(6,4))
    np.fill_diagonal(prior,0)

    # posterior
    posterior = prior

    # do not exhaust all stimulus/silhouettes, just learn from t-silhouettes
    solution_array = solution_array[:t,:]

    for sil in np.arange(solution_array.shape[0]):

        # collect the pairs of connected blocks in the silhouette
        # this is a dictionary, the keys are tuples (pairs) of the connected blocks present in the silhouette
        # while the values are the number of their occurence, I only need to access the keys
        tuple_dict = tuple_counter(solution_array[sil]) # collects the tuple per row/observation

        for i in np.arange(n_prim):
            for j in np.arange(n_prim):
                if (i,j) in tuple_dict.keys():
                    posterior[i,j] = posterior[i,j] + alpha*(1-posterior[i,j])
                else:
                    posterior[i,j] = posterior[i,j] - alpha*posterior[i,j] #maybe move back to uniform prior rather than 0? ; meeting with Philipp

    return posterior


posterior = connectivity_model(solutions_1)

# print("adjacency: ", adjacency)
# #print("prior: ", prior)
# print("posterior: ", np.around(posterior, 4))

np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\learned_adjacency\\posterior_trial_connected.npy", posterior)
