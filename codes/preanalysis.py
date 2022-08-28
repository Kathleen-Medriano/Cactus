
# Imports
import numpy as np
import math
from scipy.special import softmax
from helper import *
from itertools import *
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Analysis 0: Transforming summed log probabilities into RTs (something without dimension to something with)

# We suppose that the RT for each silhouette could be predicted via linear regression with the summed log probabilities as the predictor

# RT = b_0 + b_1*x where b_0 and b_1 are parameters and x is the summed log probabilities

# This pre-analysis is to determine a suitable value parameters that could transform an essentially dimensionless value (summed log probabilities) 
# into one with dimension (time in seconds)

def rt_glm_per_individual(solution_array, posterior, rt_collected):

    """
    Identify individual participant parameters for the model "RT = b_0 + b_1*X where b_0 and b_1 are parameters and X is the summed log probabilities" 
    ... 
    1. check the solutions array
    2. identify all possible pairs 
    3. lookup the corresponding probability of each pair in the posterior
    3. multiply all of them
    """

    # Input
    # X
    # solution_array: (a matrix of n_sils x n_blocks) trials/silhouettes x correct blocks
    # posterior: (a matrix of n_prim x n_prim) learned adjacency matrix / belief of the underlying adjacency matrix

    # Y
    # rt_collected: (an array of length n_sils) 

    # Output
    # b_0: constant term in the linear model
    # b_1: coefficient in the linear model

    # initialization of X
    summed_log_prob = np.zeros(solution_array.shape[0])

    for sil in np.arange(solution_array.shape[0]):
        # list of pairs of blocks in a silhouette
        tuple_list = list(combinations(solution_array[sil],2)) # 2 since we're concerned with chunks composed of 2 blocks, list of tuples
        for (i,j) in tuple_list: 
            if posterior[i,j] != 0:
                summed_log_prob[sil] += np.log(posterior[i,j])
    
    summed_log_prob = summed_log_prob.reshape(-1,1)  # now I have a new summed_log_prob

    # reshaping of Y, just in case
    rt_collected = rt_collected.reshape(-1,1)

    # Regression
    reg = LinearRegression().fit(summed_log_prob,rand_sample_rt)
    b_0 = reg.intercept_
    b_1 = reg.coef_

    return b_0, b_1


