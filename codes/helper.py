# Contains helper functions

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 5]

#@title Helper functions for plotting

def mkPlot_subplots_LinIdx(form_linIdx,n_grid=20,n_grid_reduced=10,title='Stim ',cols=3,rows=3):

  # uses linear indexing

  fig = plt.figure() 

  for idx_BB in np.arange(np.size(form_linIdx,0)):

    BB = np.zeros((n_grid,n_grid))
    BB[np.unravel_index(form_linIdx[idx_BB], (n_grid,n_grid), order='F')] = 1 

    fig.add_subplot(cols, rows, idx_BB+1)

    plt.imshow(mkReduceGrid(BB,n_grid_reduced), cmap='Greys')
    plt.axis('off')
    plt.title(title + str(idx_BB))

  plt.show()

def mkPlot_subplots(form_matrix,plot_idx,n_grid_reduced=10,title='Stim ',cols=5,rows=5):
  
  # uses matrix code

  fig = plt.figure() 

  fig_count = 1;
  
  for idx_BB in plot_idx:

    BB = form_matrix[idx_BB,:,:]

    fig.add_subplot(rows, cols, fig_count)

    plt.imshow(mkReduceGrid(BB,n_grid_reduced), cmap='Greys')
    plt.axis('off')
    plt.title(title + str(idx_BB))

    fig_count += 1

    # print(coord_Blocks[:,:,idx_BB])

  plt.show()


def mkPlot_stim(form_matrix,n_grid_reduced=10,title='Stim'):

  # uses matrix code

  plt.imshow(mkReduceGrid(form_matrix,n_grid_reduced), cmap='Greys')
  plt.axis('off')
  plt.title(title)
  plt.show()

#@title Helper function crop silhouette mkCrop()
# returns form without additional padding, also size info
def mkCrop(FORM,output='reduced'):

  min_x = np.min(np.where(np.sum(FORM,axis=0)!=0))
  max_x = np.max(np.where(np.sum(FORM,axis=0)!=0))

  min_y = np.min(np.where(np.sum(FORM,axis=1)!=0))
  max_y = np.max(np.where(np.sum(FORM,axis=1)!=0))

  FORM_crop = FORM[min_y:max_y+1,min_x:max_x+1]

  range_x = max_x - min_x + 1;
  range_y = max_y - min_y + 1;

  if output == 'full':
    return FORM_crop, range_x, range_y, min_x, max_x, min_y, max_y
  elif output == 'reduced':
    return FORM_crop

#@title Helper function reduce grid silhouette mkReduceGrid()
# Function to reduce grid
def mkReduceGrid(FORM,n_grid_reduced):

  FORM_crop, range_x, range_y, min_x, max_x, min_y, max_y = mkCrop(FORM,'full')

  # Form_Reduced  = FORM[min_y:max_y+1+n_grid_reduced-range_y,min_x:max_x+1+n_grid_reduced-range_x]

  x_pad = (n_grid_reduced-range_x)/2
  y_pad = (n_grid_reduced-range_y)/2

  x_pad1 = int(np.floor(x_pad))
  x_pad2 = int(np.ceil(x_pad))

  y_pad1 = int(np.floor(y_pad))
  y_pad2 = int(np.ceil(y_pad))

  Form_Reduced = np.pad(FORM_crop, ((y_pad1, y_pad2), (x_pad1, x_pad2)), 'constant', constant_values=0)

  return Form_Reduced

#@title Helper function overlap, size_diff, overlap, size_1, size_2 = mk_vis_overlap()

# for checking the difference in size of 2 size reduced BBs???


def mk_vis_overlap(Form_1,Form_2):
  # expects cropped forms

  size_1 = np.shape(Form_1)
  size_2 = np.shape(Form_2)

  if size_1[0]==size_2[0] and size_1[1]==size_2[1]:

    y_diff = 0
    x_diff = 0
    Form_xy = np.add(Form_1,Form_2)
    overlap = np.sum(Form_xy[Form_xy==2])/np.sum(Form_xy[Form_xy>0]) 

  else:

    y_diff = np.max([size_1[0],size_2[0]])-np.min([size_1[0],size_2[0]])
    y_pad = np.arange(0,y_diff+1)

    x_diff = np.max([size_1[1],size_2[1]])-np.min([size_1[1],size_2[1]])
    x_pad = np.arange(0,x_diff+1)

    overlap = []

    for y_move in y_pad:
      for x_move in x_pad:
        if size_1[0]<=size_2[0] and size_1[1]<=size_2[1]:
          Form_1_pad = np.pad(Form_1, ((y_move, y_diff-y_move), (0, 0)), 'constant', constant_values=0)
          Form_1_pad = np.pad(Form_1_pad, ((0, 0), (x_move, x_diff-x_move)), 'constant', constant_values=0)
          Form_2_pad = Form_2
        elif size_1[0]<=size_2[0] and size_1[1]>size_2[1]:
          Form_1_pad = np.pad(Form_1, ((y_move, y_diff-y_move), (0, 0)), 'constant', constant_values=0)
          Form_2_pad = np.pad(Form_2, ((0, 0), (x_move, x_diff-x_move)), 'constant', constant_values=0)
        elif size_1[0]>size_2[0] and size_1[1]<=size_2[1]:
          Form_1_pad = np.pad(Form_1, ((0, 0), (x_move, x_diff-x_move)), 'constant', constant_values=0)
          Form_2_pad = np.pad(Form_2, ((y_move, y_diff-y_move), (0, 0)), 'constant', constant_values=0)
        elif size_1[0]>size_2[0] and size_1[1]>size_2[1]:
          Form_1_pad = Form_1
          Form_2_pad = np.pad(Form_2, ((y_move, y_diff-y_move), (0, 0)), 'constant', constant_values=0)
          Form_2_pad = np.pad(Form_2_pad, ((0, 0), (x_move, x_diff-x_move)), 'constant', constant_values=0)

        Form_xy = np.add(Form_1_pad,Form_2_pad)

        overlap = np.append(overlap,np.sum(Form_xy[Form_xy==2])/np.sum(Form_xy[Form_xy>0]))

  overlap = np.round(overlap,2)

  # print(np.max(overlap),np.round((y_diff+x_diff)/2,2))

  # visual overlap is defined as the max overlap under all possible translations (values between 0 and 1 = identical)
  # size overlap is average size difference (0 = same size to +inf)
  return np.max(overlap), np.round((y_diff+x_diff)/2,2), overlap, size_1, size_2

#@title Helper function coord_Blocks, neighBours = mkCoords()

# 


# # important: y is dominant, find minimum y coordinate (row), then minimum x (column) therein
# def mkCoords(final_Coord,block_code_HBB,n_blocks=4,verbose=False):

#   coord_Blocks = np.zeros((n_blocks, 3))

#   neighBours = np.zeros((n_blocks, 6)) # number blocks x number of neighbours to the (left, ontop, right, below, total number of unique neighbours)

#   neighBour_Ident = np.multiply(np.ones((n_blocks, 5)),-1) # number blocks x neighbours to the (left, ontop, right, below, total number of unique neighbours)


#   if np.size(np.where(final_Coord==10))!=0:
#     final_Coord[np.where(final_Coord==10)] = [x+1 for x in block_code_HBB[0]] # +1 because 0 = background
#   if np.size(np.where(final_Coord==20))!=0:
#     final_Coord[np.where(final_Coord==20)] = [x+1 for x in block_code_HBB[1]] # +1 because 0 = background

#   unique_vals = np.unique(final_Coord)
#   unique_vals = unique_vals[1:] # get rid of zero
#   unique_vals = unique_vals.astype(int)

#   block_count = 0
#   for idx_blocks in unique_vals:

#     coords_Block = np.where(final_Coord==idx_blocks)

#     # print(coords_Block)

#     coord_Blocks[block_count,1] = min(coords_Block[0])

#     coord_Blocks[block_count,0] = min(coords_Block[1][np.where(coords_Block[0]==min(coords_Block[0]))])

#     # for now, start counting coordinates at 1 - change to 0 later
#     # coord_Blocks[block_count,1] = min(coords_Block[0])+1

#     # coord_Blocks[block_count,0] = min(coords_Block[1][np.where(coords_Block[0]==min(coords_Block[0]))])+1

#     coord_Blocks[block_count,2] = idx_blocks-1 # -1 because we had to add one to differentiate from background (annoyed smiley)

#     # find neighbours - this is inefficient coding but hopefully easier to read that way
#     # find neighbouring grid elements (this doesn't work if we are at the border somewhere):
#     coords_Block_x = coords_Block[1]
#     coords_Block_y = coords_Block[0]
#     coords_Block_left  = [x-1 for x in coords_Block_x]
#     coords_Block_ontop = [y-1 for y in coords_Block_y]
#     coords_Block_right = [x+1 for x in coords_Block_x]
#     coords_Block_below = [y+1 for y in coords_Block_y]
    
#     # now find left, ontop, right, below neighbours, also keep track of overall (unique) neighbouers
#     all_neighbours = []

#     neighbours_left = (final_Coord[tuple([coords_Block_y,coords_Block_left])])
#     neighbours_left = neighbours_left[neighbours_left!=0]
#     neighbours_left = neighbours_left[neighbours_left!=idx_blocks]
#     neighbours_left = np.unique(neighbours_left)
#     if verbose:
#       print("Building Block " + str(idx_blocks-1) + " neighbour to left: " + str(neighbours_left-1))    

#     neighbours_ontop = (final_Coord[tuple([coords_Block_ontop,coords_Block_x])])
#     neighbours_ontop = neighbours_ontop[neighbours_ontop!=0]
#     neighbours_ontop = neighbours_ontop[neighbours_ontop!=idx_blocks]
#     neighbours_ontop = np.unique(neighbours_ontop)
#     if verbose:
#       print("Building Block " + str(idx_blocks-1) + " neighbour ontop: " + str(neighbours_ontop-1))

#     neighbours_right = (final_Coord[tuple([coords_Block_y,coords_Block_right])])
#     neighbours_right = neighbours_right[neighbours_right!=0]
#     neighbours_right = neighbours_right[neighbours_right!=idx_blocks]
#     neighbours_right = np.unique(neighbours_right)
#     if verbose:
#       print("Building Block " + str(idx_blocks-1) + " neighbour to right: " + str(neighbours_right-1))

#     neighbours_below = (final_Coord[tuple([coords_Block_below,coords_Block_x])])
#     neighbours_below = neighbours_below[neighbours_below!=0]
#     neighbours_below = neighbours_below[neighbours_below!=idx_blocks]
#     neighbours_below = np.unique(neighbours_below)
#     if verbose:
#       print("Building Block " + str(idx_blocks-1) + " neighbour below: " + str(neighbours_below-1))

#     all_neighbours = np.append(all_neighbours,neighbours_left-1)
#     all_neighbours = np.append(all_neighbours,neighbours_ontop-1)
#     all_neighbours = np.append(all_neighbours,neighbours_right-1)
#     all_neighbours = np.append(all_neighbours,neighbours_below-1)

#     all_neighbours = np.unique(all_neighbours)

#     if verbose:
#       print("Building Block " + str(idx_blocks-1) + " neighbour in total (sum): " + str(all_neighbours) + "(" + str(len(all_neighbours)) + ")")

#     neighBours[block_count,0] = idx_blocks-1
#     neighBours[block_count,1] = len(neighbours_left)
#     neighBours[block_count,2] = len(neighbours_ontop)
#     neighBours[block_count,3] = len(neighbours_right)
#     neighBours[block_count,4] = len(neighbours_below)
#     neighBours[block_count,5] = len(all_neighbours)

#     # find neighbour identity ONLY if 1 neighbour
#     neighBour_Ident[block_count,0] = idx_blocks-1
#     if len(neighbours_left)==1:
#       neighBour_Ident[block_count,1] = neighbours_left-1
#     if len(neighbours_ontop)==1:
#       neighBour_Ident[block_count,2] = neighbours_ontop-1
#     if len(neighbours_right)==1:
#       neighBour_Ident[block_count,3] = neighbours_right-1
#     if len(neighbours_below)==1:
#       neighBour_Ident[block_count,4] = neighbours_below-1

#     block_count += 1

#   return coord_Blocks, neighBours, neighBour_Ident

#@title Helper function overlap_prev = findOverlap_Sil()
# 
# get overlap info one silhouette to some other/s
def findOverlap_Sil(idx_sil,idx_otherSil,Sil_Shape):

  overlap_prev = []
  for idx_other in idx_otherSil:
      overlap, _, _, _, _ = mk_vis_overlap(mkCrop(Sil_Shape[idx_sil,:,:]),mkCrop(Sil_Shape[idx_other,:,:]))
      overlap_prev = np.append(overlap_prev,overlap)

  return overlap_prev

#@title Helper function unique_sil = findUnique_Sil()
# 
# find unique silhouettes
def findUnique_Sil(idx_AllSil,final_Sil):

  unique_sil = []

  for idx_sil in idx_AllSil:
    
    if idx_sil>idx_AllSil[0]:
      overlap_prev = findOverlap_Sil(idx_sil,idx_AllSil[0:np.where(idx_sil==idx_AllSil)[0][0]],final_Sil)
    
    if idx_sil==idx_AllSil[0] or np.all(overlap_prev<1):
      unique_sil = np.append(unique_sil,idx_sil)

  unique_sil = unique_sil.astype(int)

  return unique_sil

# Helper function plaCement,remAinder,is_part_count = is_part()


# find out if building blocks are part of a silhouette and where

from pylab import *
from scipy.ndimage import measurements

def is_part(form_use,sil,min_block_size=3,n_grid=20,verbose=False):

  plaCement = np.zeros((100,np.size(sil,0),np.size(sil,1)))
  remAinder = np.zeros((100,np.size(sil,0),np.size(sil,1)))

  remAinder_justOverlap = np.zeros((100,np.size(sil,0),np.size(sil,1)))

  # find shape of given building block
  BB = np.zeros((n_grid,n_grid))
  BB[np.unravel_index(form_use, (n_grid,n_grid), order='F')] = 1
  BB = mkCrop(BB).astype(int)

  if verbose:
    print(sil)
    print(BB)

  # find discrepancy in size between building block and silhouette
  row_idx = np.arange(0,np.size(sil,0)-np.size(BB,0)+1)
  col_idx = np.arange(0,np.size(sil,1)-np.size(BB,1)+1)

  is_part_count = 0 # this will count if it's a part and remaining parts can be built
  is_part_count_justOverlap = 0 # this will just count if BB fits somehow

  for r_idx in row_idx:
    for c_idx in col_idx:
      temp = np.array(sil).astype(int)
      temp[r_idx:r_idx+np.size(BB,0),c_idx:c_idx+np.size(BB,1)] -= BB.astype(int) # subtract the building block shape from the silhouette

      # print(temp)

      if np.all(temp!=-1): # if the subtraction worked (i.e. we only removed part of a silhouette)

        remAinder_justOverlap[is_part_count_justOverlap,:,:] = temp

        is_part_count_justOverlap += 1

        if verbose:
          print(temp)

        lw, num = measurements.label(temp) # gather info about remaining shape, lw = remaining pieces
        area = measurements.sum(temp, lw, index=arange(lw.max() + 1))

        # check if split in two clusters at most, and those cluster can be built with building blocks 
        # (i.e. their size must either be 3, 6 or 9 given all building blocks have size 3):
        if np.all(lw<3) and np.all(np.isin(area[1:],[min_block_size,min_block_size*2,min_block_size*3])):

          temp_placement = np.zeros_like(sil)
          temp_placement[r_idx:r_idx+np.size(BB,0),c_idx:c_idx+np.size(BB,1)] += BB.astype(int)   # now let's assume we've placed this building block       

          plaCement[is_part_count,:,:] = temp_placement
          remAinder[is_part_count,:,:] = temp

          is_part_count += 1

          # if verbose:                        
          #   print(temp_placement)
          #   print(temp)
          #   print("Found " + str(is_part_count) + " part solutions.")
  
  plaCement = plaCement[0:is_part_count,:,:]
  remAinder = remAinder[0:is_part_count,:,:]
  remAinder_justOverlap = remAinder_justOverlap[0:is_part_count_justOverlap,:,:]

  return plaCement.astype(int),remAinder.astype(int),is_part_count,is_part_count_justOverlap,remAinder_justOverlap

# Helper function single_sol,is_part_count = find_solution()

# idea: find all possible locations of building blocks, see if they would work in principle
# see if those combinations work
def find_solution(sil,form_BB,verbose=False):

  n_blocks = np.size(form_BB,0)

  is_part_count = np.zeros(n_blocks)

  for idx_block in np.arange(n_blocks):

    _,_,is_part_count[idx_block],_,_ = is_part(form_BB[idx_block],sil)
    # _,_,is_part_count[idx_block] = is_part(form_BB[idx_block],sil,verbose=True)

  is_part_count = is_part_count.astype(int)

  # single_sol = np.sum(is_part_count)==4
  single_sol = np.sum(is_part_count!=0)==4

  if verbose:
    print(is_part_count)
    print(single_sol)  
    # print(np.sum(is_part_count!=0))

  return single_sol,is_part_count

# generate block sequence based on a adjacency matrix

def block_sequence(Adjacency,starting_set=[],seq_length=4,verbose=False):

# Input:
  # n_grid: (scalar) size of grid
  # blocks: (array of int; each int corresponds to 1 BB and the sequence of ints in the array is randomized) identifies which blocks to use from the set of primitive BBs, expects array and then go through sequentially
  # form_block: (array of arrays; each of which specifies the form of each primitive building block) shape of blocks, called by 'blocks'

# Input:
# Adjacency: (n_block x n_block matrix) describes the adjacency of building blocks to each other
# starting set
# seq_length: (scalar) indicates the number of the blocks to use for silhouette generation
# verbose: (boolean) used to display details

# Output:
# use_blocks: (array of seq_length size) indicates the blocks to use for silhouette generation when an adjacency matrix is considered

  sim_done = False

  while sim_done==False:
  
    blocks = np.arange(np.size(Adjacency,0))
    use_blocks = []

    # Pick a first block
    if len(starting_set)==0:
      use_blocks.append(np.random.choice(blocks[np.sum(Adjacency,1)!=0],1)[0]) # find first block // checks if the sum along each column is not equal to zero, then that block is allowed for usage
    else:
      use_blocks.append(np.random.choice(starting_set,1)[0]) # find first block # // i don't understand what this is for, maybe accounts for the case when starting_set isn't initially empty

    if verbose:
      print(use_blocks)

    for idx_next in np.arange(1,seq_length):
      #next_block = blocks[Adjacency[use_blocks[-1],:]==1] # find allowed next blocks // sequential adding of blocks
      next_block = blocks[Adjacency[use_blocks[-1],:]>0] # find allowed next blocks // sequential adding of blocks // not just discrete 1 (yes, connected) or 0 (no, not connected) // incorporate probability in this case
      next_block = np.setdiff1d(next_block,use_blocks) # of those, remove those that are already used // just checks that already used blocks are filtered

      if len(next_block)>0: 
        use_blocks.append(np.random.choice(next_block,1)[0]) # add them // add randomly from all the the possible blocks

      if verbose:
        print(use_blocks)

    if verbose:
      print(use_blocks)
      print(Adjacency)

    if len(use_blocks)==4:
      sim_done = True

  return use_blocks


######## Helper functions for main.py

def tuple_counter(row):

  """Counts the number of times a pair of consecutive elements appeared in an array. """

  # Input:
  # row (a one dimensional array)

  # Output:
  # tuple_dict (dictionary) the keys are tuples with 2 consecutive elements, values are the number of times the tuple (consecutive elements) appeared in the array

  tuple_dict = {}

  for i in np.arange(len(row)-1): # not including the last element in the array
    tuple_ = (row[i],row[i+1]) # collect the consecutive elements in a tuple
    if tuple_ in tuple_dict.keys():
      tuple_dict[tuple_] += 1
    else:
      tuple_dict[tuple_] = 1

  return tuple_dict


def ignore_operation_diag(matrix, operation):

  """Don't do the operation on the diagonal of a matrix"""
  # Input:

  # matrix
  # operation: (scalar operation)

  (x,y) = matrix.shape

  for i in np.arange(x):
    for j in np.arange(y):
      if i!=j: 
        matrix[i,j] = operation(matrix[i,j])

  return matrix



def retreive_bbs(trial_Shape):

  BBs = []
  solution_str = trial_Shape # the entry in the solution column
  digits = re.findall('\d',solution_str) # get all digits
  bb_digits = digits[2:12:3]
  BBs.append(bb_digits)

  return BBs


# stupid readout function of text-based silhouette description of coordinate form
def make_ShapeCoord(trial_Shape):

  BBs = []
  solution_str = trial_Shape # the entry in the solution column
  digits = re.findall('\d',solution_str) # get all digits
  for pos in [0,3,6,9]:
    BBs.append([int(i) for i in digits[pos:pos+3]])

  return BBs


# draw a silhoutte based on coordinate descriptions
# Input
# trial_ShapeCoord: coordinates of to-be-drawn buiding blocks (format: [x,y,building block id])
# n_grid_reduced: size of grid
# BB_shape: if there's already a shape, empty grid otherwise
# add_val: which value to give to the grid elements
# 
# OUTPUT
# adjacency: adjacency (i.e. connection) between building blocks in present silhouette
# BB_built: shape of silhouette
# BB_built_vals: shape of silhouette with building block info
def draw_silh(trial_ShapeCoord,n_grid_reduced=10,BB_shape=[],add_val=1):

  # those are the translations from a given coordinate of a building block
  translations_BB = [
                    [[0,0], [1,0], [0,1]],
                    [[0,0], [1,0], [1,1]],
                    [[0,0], [0,1], [-1,1]],
                    [[0,0], [0,1], [1,1]],
                    [[0,0], [1,0], [2,0]],
                    [[0,0], [0,1], [0,2]],
                    ]

  if len(BB_shape)==0:
    BB_built = np.zeros((n_grid_reduced,n_grid_reduced)) # initialise empty grid
    BB_built_vals = np.zeros_like(BB_built)
  else:
    BB_built = np.array(BB_shape)
    BB_built_vals = np.array(BB_shape)

  n_blocks = len(trial_ShapeCoord)

  for idx_block in np.arange(n_blocks):

    idx_block_translation = translations_BB[trial_ShapeCoord[idx_block][2]]

    for idx_translation in np.arange(len(idx_block_translation)):

      idx_draw = np.add(trial_ShapeCoord[idx_block][:2],idx_block_translation[idx_translation])

      BB_built[idx_draw[1],idx_draw[0]] = add_val

      BB_built_vals[idx_draw[1],idx_draw[0]] = trial_ShapeCoord[idx_block][2] +1 # +1 to differentiate from background

  coord_Blocks, neighBours, neighBour_Ident = mkCoords(BB_built_vals)

  adjacency = np.zeros((6,6))

  for idx_BB in np.arange(np.size(neighBour_Ident,0)):
    idx_connected = neighBour_Ident[idx_BB][neighBour_Ident[idx_BB]>-1].astype(int)
    adjacency[idx_connected[0],idx_connected[1:]] = 1

  return adjacency, BB_built.astype(int), BB_built_vals.astype(int)


# Obtain coordinate and neighbour info for a given silhouette
# Expects shape of silhouette with building block info as input (see draw_silh() function below)
# important: y is dominant, find minimum y coordinate (row), then minimum x (column) therein
def mkCoords(final_Coord,n_blocks=4,verbose=False):

  coord_Blocks = np.zeros((n_blocks, 3)) ## why 4x3? what is this for???

  neighBours = np.zeros((n_blocks, 6)) # number blocks x number of neighbours to the (left, ontop, right, below, total number of unique neighbours)

  neighBour_Ident = np.multiply(np.ones((n_blocks, 5)),-1) # number blocks x neighbours to the (left, ontop, right, below, total number of unique neighbours)

  unique_vals = np.unique(final_Coord) ## get all the building blocks present
  unique_vals = unique_vals[1:] # get rid of zero ## why get rid of zero???? # ahh since 0 is the background, then bb number starts at 1?
  unique_vals = unique_vals.astype(int)

  block_count = 0 # counter 
  for idx_blocks in unique_vals: # you can just enumerate instead of using a counter 

    coords_Block = np.where(final_Coord==idx_blocks) ## this is whatever the size of final_Coord is, logical 1s and 0s

    coord_Blocks[block_count,1] = min(coords_Block[0]) ## minimum of the 1st row??

    coord_Blocks[block_count,0] = min(coords_Block[1][np.where(coords_Block[0]==min(coords_Block[0]))])

    coord_Blocks[block_count,2] = idx_blocks-1 # -1 because we had to add one to differentiate from background (annoyed smiley)

    # find neighbours - this is inefficient coding but hopefully easier to read that way
    # find neighbouring grid elements (this doesn't work if we are at the border somewhere):
    coords_Block_x = coords_Block[1]
    coords_Block_y = coords_Block[0]
    coords_Block_left  = [x-1 for x in coords_Block_x]
    coords_Block_ontop = [y-1 for y in coords_Block_y]
    coords_Block_right = [x+1 for x in coords_Block_x]
    coords_Block_below = [y+1 for y in coords_Block_y]
    
    # now find left, ontop, right, below neighbours, also keep track of overall (unique) neighbouers
    all_neighbours = []

    neighbours_left = (final_Coord[tuple([coords_Block_y,coords_Block_left])])
    neighbours_left = neighbours_left[neighbours_left!=0]
    neighbours_left = neighbours_left[neighbours_left!=idx_blocks]
    neighbours_left = np.unique(neighbours_left)
    if verbose:
      print("Building Block " + str(idx_blocks-1) + " neighbour to left: " + str(neighbours_left-1))    

    neighbours_ontop = (final_Coord[tuple([coords_Block_ontop,coords_Block_x])])
    neighbours_ontop = neighbours_ontop[neighbours_ontop!=0]
    neighbours_ontop = neighbours_ontop[neighbours_ontop!=idx_blocks]
    neighbours_ontop = np.unique(neighbours_ontop)
    if verbose:
      print("Building Block " + str(idx_blocks-1) + " neighbour ontop: " + str(neighbours_ontop-1))

    neighbours_right = (final_Coord[tuple([coords_Block_y,coords_Block_right])])
    neighbours_right = neighbours_right[neighbours_right!=0]
    neighbours_right = neighbours_right[neighbours_right!=idx_blocks]
    neighbours_right = np.unique(neighbours_right)
    if verbose:
      print("Building Block " + str(idx_blocks-1) + " neighbour to right: " + str(neighbours_right-1))

    neighbours_below = (final_Coord[tuple([coords_Block_below,coords_Block_x])])
    neighbours_below = neighbours_below[neighbours_below!=0]
    neighbours_below = neighbours_below[neighbours_below!=idx_blocks]
    neighbours_below = np.unique(neighbours_below)
    if verbose:
      print("Building Block " + str(idx_blocks-1) + " neighbour below: " + str(neighbours_below-1))

    all_neighbours = np.append(all_neighbours,neighbours_left-1)
    all_neighbours = np.append(all_neighbours,neighbours_ontop-1)
    all_neighbours = np.append(all_neighbours,neighbours_right-1)
    all_neighbours = np.append(all_neighbours,neighbours_below-1)

    all_neighbours = np.unique(all_neighbours)

    if verbose:
      print("Building Block " + str(idx_blocks-1) + " neighbour in total (sum): " + str(all_neighbours) + "(" + str(len(all_neighbours)) + ")")

    neighBours[block_count,0] = idx_blocks-1
    neighBours[block_count,1] = len(neighbours_left)
    neighBours[block_count,2] = len(neighbours_ontop)
    neighBours[block_count,3] = len(neighbours_right)
    neighBours[block_count,4] = len(neighbours_below)
    neighBours[block_count,5] = len(all_neighbours)

    # find neighbour identity ONLY if 1 neighbour
    neighBour_Ident[block_count,0] = idx_blocks-1
    if len(neighbours_left)==1:
      neighBour_Ident[block_count,1] = neighbours_left-1
    if len(neighbours_ontop)==1:
      neighBour_Ident[block_count,2] = neighbours_ontop-1
    if len(neighbours_right)==1:
      neighBour_Ident[block_count,3] = neighbours_right-1
    if len(neighbours_below)==1:
      neighBour_Ident[block_count,4] = neighbours_below-1

    block_count += 1

  return coord_Blocks, neighBours, neighBour_Ident

# coord_Blocks -
# neighBours -
# neighBour_Ident - 