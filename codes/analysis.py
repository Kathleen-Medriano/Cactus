# Analysis

import numpy as np
import math
from scipy.special import softmax
from helper import *
from itertools import *
from datetime import datetime
import matplotlib.pyplot as plt

# Analysis 1: Response time analysis.

# The RT for each silhouette could be approximated as the log of the product of the corresponding probabilities for each pair of chunks in the silhouette
# Note: log of the product = sum of the logs

# Input: posterior (a 6x6 matrix)

def rt_prediction(solution_array, posterior, scale_factor=1/1.7):

    """
    Predict the RT for each silhouette trial

    1. check the solutions array
    2. identify all possible pairs 
    3. lookup the corresponding probability of each pair in the posterior
    3. multiply all of them
    """

    # Input
    # solution_array: (a matrix of n_sils x n_blocks) 
    # posterior: (a matrix of n_prim x n_prim) belief of the under
    # scale_factor: (scalar) current scale factor was derived from scaling the max of predicted RTs and obbserved RTs
    # noise: ()

    # Output
    # predicted_rt: (an array of length n_sils) reports the predicted RTs for each sil

    # Initialization
    predicted_rt = np.zeros(solution_array.shape[0])

    for sil in np.arange(solution_array.shape[0]):
        # list of pairs of blocks in a silhouette
        tuple_list = list(combinations(solution_array[sil],2)) #2 since we're concerned with chunks composed of 2 blocks, list of tuples
        for (i,j) in tuple_list:
            if posterior[i,j] != 0:
            #    predicted_rt[sil] *= posterior[i,j]
                predicted_rt[sil] += np.log(posterior[i,j])
                predicted_rt[sil] = -scale_factor*predicted_rt[sil] #+ np.random.normal(0) # I want for each silhouette to have a an associated error term 

    return predicted_rt

# RT distribution

# Load Data

#final_Sil = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\final_Sil20220803-104805.npy")
solutions_1 = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\solutions_20220803-104805.npy")
#coord_Blocks = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\coord_Blocks_20220803-104805.npy")
#adjacency = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\adjacency\\Adjacency_20220803-104805.npy")
posterior = np.load(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\learned_adjacency\\posterior_trial.npy")

predicted_rt = rt_prediction(solutions_1, posterior)

#np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\predicted_rt\\predicted_rt_trial.npy", predicted_rt)

max_rt = np.max(predicted_rt)
min_rt = np.min(predicted_rt)

print(f"max rt: {max_rt}")
print(f"min rt: {min_rt}")
#print(f"histogram of summation of log probabilities")

plt.hist(predicted_rt, bins=30)
plt.show()
