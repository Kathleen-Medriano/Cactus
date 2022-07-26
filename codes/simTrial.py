def simTrial(n_grid,blocks,form_block,block_code,maintain_order=False,verbose=False,):

  # Input:
  # n_grid: (scalar) size of grid
  # blocks: (array of int; each int corresponds to 1 BB and the sequence of ints in the array is randomized) identifies which blocks to use from the set of primitive BBs, expects array and then go through sequentially
  # form_block: (array of arrays; each of which specifies the form of each primitive building block) shape of blocks, called by 'blocks'
  # block code: (array of int; each int corresponds to BBs) used to identify which block is the certain portion of the silhouette from 
  # maintain_order: (boolean) used to "track completion of silhouette"
  # verbose: (boolean) used to display details
  # idea is to work with linear indices on grid


  # Output:
  # final_Form: (matrix of n_grid x n_grid) matrix of the silhouette, instead of using different BB_codes, 1 is used
  # final_Coord: (matrix of n_grid x n_grid) matrix with the BB_codes, to identify BBs used and the orientation of the BBs

  # number of blocks to use
  n_blocks = np.isize(blocks,0)

  # initialise form
  final_Form = np.zeros((n_grid,n_grid))

  # initialise coordinates
  final_Coord = np.zeros((n_grid,n_grid))
 
  #start with first building block
  current_form = np.array(form_block[blocks[0]])  

  # obtain coordinate information in reduced grid
  final_Coord[np.unravel_index(current_form, (n_grid,n_grid), order='F')] = block_code[0] # move from linear index into grid (matrix)

  # specify bounds on grid - to control we are not moving outside of grid
  up_bound    = np.arange(n_grid**2-(n_grid-1),n_grid)
  low_bound   = np.arange((n_grid-1),n_grid**2,n_grid)
  left_bound  = np.arange(n_grid)
  right_bound = np.arange(n_grid**2-n_grid,n_grid**2)


for idx_BB in np.arange(1,np.size(blocks,0)): # start with the next BB

    # find possible adjacent starting points for next BB
    if (maintain_order and idx_BB>1): # end state
      adj_points = np.unique(np.array([next_block-n_grid, next_block-1, next_block+n_grid, next_block+1])) # all adjacent pixels left, ontop, right, or below
    else:
      adj_points = np.unique(np.array([current_form-n_grid, current_form-1, current_form+n_grid, current_form+1])) # all adjacent pixels left, ontop, right, or below; note that the adjacent pixels here refer to the adjacent points in the silhouette we're building
    adj_points = adj_points[~np.isin(adj_points,current_form)] # can't move 'into' silhouette   
    adj_points = adj_points[adj_points>=0] # can't move out of linear grid  
    adj_points = adj_points[adj_points<=n_grid**2] # can't move out of linear grid    
    
    # now try them in random order as connection points for next building block:
    adj_points = np.random.permutation(adj_points)

    built = False # indicator of silhouette completion
    idx_adj = 0 # idx for the adjacent points
    
    while built == False:

        # necessary because it's all randomised, ideally should go through all possible combinations
        if idx_adj==len(adj_points):
          idx_adj = 0

        # put left bottom part (random choice) of next BB onto chosen adjacent point
        # next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - form_block[blocks[idx_BB]][0]
        # conn_point_block = np.random.choice(len(form_block[blocks[idx_BB]]),1)
        # conn_point_block = np.random.choice(np.size(form_block[blocks[idx_BB]],0),1)
        conn_point_block = np.random.choice(form_block[blocks[idx_BB]],1) # choosing a connecting point in the current block indicated by idx_BB
        # next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - form_block[blocks[idx_BB]][int(conn_point_block)]
        next_block = np.array(form_block[blocks[idx_BB]]) + adj_points[idx_adj] - int(conn_point_block) # i don't get the logic yet but this is concatenated with the already chosen silhouette

        # check if we didn't move 'around' grid
        if (
            all(np.isin(next_block,np.arange(0,n_grid**2))) and # did we move outside of grid?
            all(~np.isin(next_block,current_form)) and # is new BB overlapping with prev shape (can happen due to weird shapes of BBs)
            ~(any(np.isin(low_bound,next_block)) and any(np.isin(up_bound,next_block))) and # did we accidentally move from bottom to top of box (linear idx!)
            ~(any(np.isin(left_bound,next_block)) and any(np.isin(right_bound,next_block))) # did we accidentally move from left to right of box (linear idx!)
           ):
           
           current_form = np.concatenate((current_form, next_block), axis=0) # concatenate new block

           # obtain coordinate information in reduced grid
           final_Coord[np.unravel_index(next_block, (n_grid,n_grid), order='F')] = block_code[idx_BB] # move from linear index into grid (matrix)           

           if verbose:
             print('Done, it took ' + str(idx_adj+1) + ' attempts.')

           built = True
           
        else: # if any ofthe conditions starting in line 71 is violated then we look for a different adjacent pixel (next sequence in the adj_points)
            
            idx_adj += 1

  final_Form[np.unravel_index(current_form, (n_grid,n_grid), order='F')] = 1 # move from linear index into grid (matrix)

return final_Form, final_Coord

# 1. the set BB of primitive building blocks is defined
# 2. a subset UB from BB is taken, this consists of the primitive building blocks used for the silhouette; in the current tangram task |UB| = 4
# 3. one BB_i from the set UB is chosen as the first block to use for the silhoette
# 4. adjacent pixels of BB_i are identified to know which sides could the next BB be attached
# 5. one of the adjacent cells of BB_i is chosen 
# 6. then a next building block is chosen, BB_i+1
# 7. a connecting point in BB_i+1 is identified
# 8. then BB_i and BB_i+1 are connected through the chosen adjacent pixel and the connecting point, if no rules are violated then this is accepted and steps 4 to 8 are repeated for the remaining BB_i+2, if not
# then a new adjacent pixel is chosen