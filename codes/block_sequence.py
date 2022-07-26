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
      use_blocks.append(np.random.choice(blocks[np.sum(Adjacency,1)!=0],1)[0]) # find first block
    else:
      use_blocks.append(np.random.choice(starting_set,1)[0]) # find first block # // i don't understand what this is for, maybe accounts for the case when starting_set isn't initially empty

    if verbose:
      print(use_blocks)

    for idx_next in np.arange(1,seq_length):
      next_block = blocks[Adjacency[use_blocks[-1],:]==1] # find allowed next blocks
      next_block = np.setdiff1d(next_block,use_blocks) # of those, remove those that are already used   

      if len(next_block)>0: 
        use_blocks.append(np.random.choice(next_block,1)[0]) # add them

      if verbose:
        print(use_blocks)

    if verbose:
      print(use_blocks)
      print(Adjacency)

    if len(use_blocks)==4:
      sim_done = True

  return use_blocks