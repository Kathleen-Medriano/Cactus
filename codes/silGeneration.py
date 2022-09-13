# Silhouette generation script

import numpy as np
from simTrial import simTrial
from helper import *
from datetime import datetime

####  Basic Settings

n_trial_types = 1
n_stims_use = 30 # how many different stims per condition
n_rep_stim = 5 # rep per stimulus
n_sim   = n_stims_use*n_rep_stim*n_trial_types # how many stims to simulate

n_sim_2AFC = 60

n_sim_minesweeper = 45

do_plot = True; # plot stims and solutions

n_grid_reduced = 10 # grid size of output
n_grid         = 20 # simulate in large grid, then change to n_grid_reduced for output

n_blocks       = 4 # how many primitive blocks to use in a silhouette

# find the middle of the big grid to place the first building block
starting_point = int(n_grid**2/2-(n_grid/2))

# define different trial types
# 0 <=> silhouette consists of 2 chunks
# 1 <=> silhouette consists of 1 chunk and 2 primitive building blocks
# 2 <=> silhouette consists of 4 primitive building blocks
# trial_type = np.random.permutation(np.tile(np.array([0,1,2]), int(n_sim/3))

trial_type = np.add(np.ones((n_stims_use*n_rep_stim)),1).astype(int)
trial_type_minesweeper = np.zeros((int(n_sim_minesweeper))).astype(int)

####  Generate Building Blocks

# define shape of primite building blocks:
form_BB = [
          [starting_point,   starting_point-1,          starting_point+(n_grid-1)],
          [starting_point-1, starting_point+(n_grid-1), starting_point+n_grid],
          [starting_point,   starting_point+n_grid,     starting_point+(n_grid-1)],
          [starting_point,   starting_point-1,          starting_point+n_grid],
          [starting_point,   starting_point+n_grid,     starting_point+n_grid*2],
          [starting_point,   starting_point-1,          starting_point-2]
          ]

####  Adjacency: currently the adjacency is defined as whether blocks are allowed to be right next to each other

Adjacency = np.zeros((len(form_BB),len(form_BB)))

# Adjacency[0,[1,4,5]] = 1
# Adjacency[1,[0,3,4]] = 1
# Adjacency[2,[3,4,5]] = 1
# Adjacency[3,[1,2,4]] = 1
# Adjacency[4,[0,1,2,3,5]] = 1
# Adjacency[5,[0,2,4]] = 1

Adjacency[0,[4,5]] = 1
Adjacency[1,[2,3]] = 1
Adjacency[2,[1,3,5]] = 1
Adjacency[3,[1,2]] = 1
Adjacency[4,[0,5]] = 1
Adjacency[5,[0,2,4]] = 1

starting_set = [0,4,1,3]

print('checking the adjacency run')
tstp = datetime.now().strftime("%Y%m%d-%H%M%S")
np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\adjacency\\Adjacency_{tstp}.npy", Adjacency)
np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\adjacency\\starting-set_{tstp}.npy", starting_set)


### Find random stims
n_sim = 200

final_Sil_random    = np.zeros((n_sim,n_grid_reduced,n_grid_reduced))
final_Coord_random  = np.zeros((n_sim,n_grid_reduced,n_grid_reduced))
solutions_random    = np.zeros((n_sim,4))
coord_Blocks_random = np.zeros((n_sim,4,4))

idx_unique = 0

BBs = np.arange(6)

for idx_sim in np.arange(n_sim):

  # introduce bias to match marginal probs of BBs
  # use_blocks    = np.random.permutation(np.random.choice(BBs, 4, replace=False))
  use_blocks    = block_sequence(Adjacency,starting_set) # sample a sequence from allowed adjacencies

  block_code = [(x+1) for x in use_blocks] # +1 because 0 = background

  single_sol = False
  correct_sol = False
  while (single_sol==False) or (correct_sol==False): # ensure silhouette only has one solution
    # Sil_Trial, final_Coord = simTrial(n_grid,use_blocks,form_BB,block_code) # get silhouette
    Sil_Trial, final_Coord = simTrial(n_grid,use_blocks,form_BB,block_code,True) # get silhouette, respect order of blocks
      
    single_sol, _ = find_solution(mkCrop(Sil_Trial),form_BB)

    _, _, neighBour_Ident = mkCoords(final_Coord,form_BB)

    adjacency_trial = np.zeros((len(BBs),len(BBs)))

    for idx_BB in np.arange(n_blocks):
      non_zero = neighBour_Ident[idx_BB,neighBour_Ident[idx_BB,:]>-1][1:].astype(int)
      adjacency_trial[int(neighBour_Ident[idx_BB,0]),non_zero] = 1

    if np.all(Adjacency[adjacency_trial==1]==1): # check if trial adjacency is allowed:
      
      correct_sol = True

    else:

      correct_sol = False

    # print("Single solution: " + str(single_sol) + ", Allowed solution: " + str(correct_sol)) 

  # Sil_Trial, final_Coord = simTrial(n_grid,use_blocks,form_BB_H, block_code) # get silhouette

  if idx_unique==0:
    final_Sil_random[idx_unique,:,:] = mkReduceGrid(Sil_Trial,n_grid_reduced)

    final_Coord = mkReduceGrid(final_Coord,n_grid_reduced)
    coord_Blocks_random[idx_unique,:,0:3], _, _ = mkCoords(final_Coord,form_BB)
    coord_Blocks_random[idx_unique,:,3] = idx_unique

    final_Coord_random[idx_unique,:,:] = final_Coord

    solutions_random[idx_unique,:]   = use_blocks
    # print(solutions_random[idx_unique,:])

    idx_unique += 1
  else:

    final_Sil_random[idx_unique,:,:] = mkReduceGrid(Sil_Trial,n_grid_reduced)

    overlap_prev = findOverlap_Sil(idx_unique,np.arange(idx_unique),final_Sil_random)
    if np.all(overlap_prev<1):
      final_Coord = mkReduceGrid(final_Coord,n_grid_reduced)
      coord_Blocks_random[idx_unique,:,0:3], _, _ = mkCoords(final_Coord,form_BB)
      coord_Blocks_random[idx_unique,:,3] = idx_unique

      final_Coord_random[idx_unique,:,:] = final_Coord

      solutions_random[idx_unique,:]   = use_blocks 
      # print(solutions_random[idx_unique,:])         

      idx_unique += 1 
      # print("Bingo, found another. Now have " + str(idx_unique))      

#  if idx_sim%10 == 0:
#    print('Trial ' + str(idx_sim) + ' of ' + str(n_sim) + ' done.')
#    print("Found " + str(idx_unique) + " so far.")    

final_Sil_random    = final_Sil_random[0:idx_unique,:,:].astype(int)
final_Coord_random  = final_Coord_random[0:idx_unique,:,:].astype(int)
solutions_random    = solutions_random[0:idx_unique,:].astype(int)
coord_Blocks_random = coord_Blocks_random[0:idx_unique,:,:].astype(int)

print("Found " + str(idx_unique) + " in total.")

###  Simulate actual experiment

n_sim = len(trial_type)

final_Sil    = np.zeros((n_sim,n_grid_reduced,n_grid_reduced))
solutions    = np.zeros((n_sim,4))
coord_Blocks = np.zeros((n_sim,4,4))

final_Sil    = final_Sil.astype(int)
solutions    = solutions.astype(int)
coord_Blocks = coord_Blocks.astype(int)

stims_use = np.zeros_like(trial_type)

# find random silhouettes
mainTask_sil_random = np.random.choice(np.size(final_Sil_random,0),n_stims_use,False)
# print("Random Stims:")
# mkPlot_subplots(final_Sil_random,mainTask_sil_random,10,'Stim ',5,5)
# mkPlot_subplots(final_Sil_random,np.arange(np.min([np.size(final_Sil_random,0),25])),10,'Stim ',5,5)

ChunkR_Sil_Use = final_Sil_random[mainTask_sil_random,:,:]

mainTask_sil_random = np.random.permutation(np.tile(mainTask_sil_random, 
                                            int(len(trial_type[trial_type==2])/n_stims_use)))
stims_use[trial_type==2] = mainTask_sil_random
# print(stims_use)

final_Sil[trial_type==2,:,:] = final_Sil_random[mainTask_sil_random,:,:].astype(int)
solutions[trial_type==2,:] = solutions_random[mainTask_sil_random,:].astype(int)
coord_Blocks[trial_type==2,:,:] = coord_Blocks_random[mainTask_sil_random,:,:].astype(int)

for idx in np.arange(n_sim):
  coord_Blocks[idx,:,3] = idx
# coord_Blocks[:,:,3] = np.tile(np.arange(n_sim),20)


### Saving the silhouettes

np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\final_Sil{tstp}.npy", final_Sil)
np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\solutions_{tstp}.npy", solutions)
np.save(f"C:\\Users\\kmedriano\\Documents\\Cactus\\tests\\silhouettes\\coord_Blocks_{tstp}.npy", coord_Blocks)