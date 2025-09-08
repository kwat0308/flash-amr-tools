'''
Supplementary module to contain low-level routines for block determination
'''
import numpy as np
from .zorder import zenumerate

# Find all blocks at lowest refinement level which are 
# correspoding to our initial guess
def find_blocks(
    block_list, 
    min_ref_lvl, max_ref_lvl, 
    brlvl, coords, block_size, bsmin,
    refine_level, gid, center, 
    is_cuboid=False,
    verbose=False
):
    # For all block which are higher than our set refinement 
    # level find the lowest parent which is fulfills our
    # requested refinement level.
    
    # Faster routine
    tmp_blist = block_list.copy()
    blist_ref = []
    for i in range(max_ref_lvl - min_ref_lvl):
        tmp_sel = refine_level[tmp_blist] > min_ref_lvl
        tmp_gid = gid[tmp_blist]
        tmp_blist = np.unique(tmp_gid[tmp_sel, 6] - 1)
        blist_ref += tmp_blist[refine_level[tmp_blist] == min_ref_lvl].tolist()

    #blist_ref = []
    #for block in block_list[brlvl > min_ref_lvl]:
    #    b_tmp = block
    #    while refine_level[b_tmp] > min_ref_lvl:
    #        b_tmp = gid[b_tmp][6] - 1
    #    blist_ref.append(b_tmp)

    # As there will be many overlaps
    # we create a unique list of block ids
    blist_ref = np.unique(blist_ref)
    
    # And add them to all the cells which already
    # were at the lowest refinement level.
    minref_blist = np.unique(
        np.concatenate(
            (block_list[brlvl == min_ref_lvl], blist_ref)
        )
    ).astype(int)

    # Check how many blocks fit into 
    # the current range in x, y and z 
    # at the lowest refinement level.
    up_cor = (coords[minref_blist, :] + 0.5 * block_size[minref_blist, :]).max(axis=0)
    low_cor = (coords[minref_blist, :] - 0.5 * block_size[minref_blist, :]).min(axis=0)
    
    bns = (up_cor - low_cor) / (bsmin * 2**(max_ref_lvl - min_ref_lvl))
    bnx, bny, bnz = np.round(bns).astype(int)

    # Calculate the max number of blocks at
    # lowest refinement level needed to be 
    # able to create an amr tree.
    bn = np.max([bnx, bny, bnz])
    bnmax = 2**np.ceil(np.log2(bn))

    # Check if we have all our blocks fill the
    # cuboid at least in numbers.
    # Let's hope this is never the case.
    # Currently no idea how to fix this.
    if not bnx * bny * bnz == minref_blist.shape[0]:
        if verbose:
            print('Blocks in x: ', bnx)
            print('Blocks in y: ', bny)
            print('Blocks in z: ', bnz)
            print(minref_blist.shape)
        return 1, 1, 1, 1, 2
        # sys.exit('Could not find enough blocks to fill cuboid.')
    elif is_cuboid:
        return minref_blist, bnx, bny, bnz, bnmax

    # For each round add blocks add at least
    # one side to reach the maximum number
    # of blocks for amr tree.
    for i in range(int(bnmax - np.min([bnx, bny, bnz]) + 2)):

        minref_blist, bnx, bny, bnz = add_blocks(
            minref_blist=minref_blist, coords=coords,
            block_size=block_size, gid=gid,
            bnx=bnx, bny=bny, bnz=bnz, bnmax=bnmax,
            center=center, axis=0
        )

        minref_blist, bnx, bny, bnz = add_blocks(
            minref_blist=minref_blist, coords=coords,
            block_size=block_size, gid=gid,
            bnx=bnx, bny=bny, bnz=bnz, bnmax=bnmax,
            center=center, axis=1
        )

        minref_blist, bnx, bny, bnz = add_blocks(
            minref_blist=minref_blist, coords=coords,
            block_size=block_size, gid=gid,
            bnx=bnx, bny=bny, bnz=bnz, bnmax=bnmax,
            center=center, axis=2
        )

    return minref_blist, bnx, bny, bnz, bnmax


def add_blocks(
    minref_blist,
    coords, block_size, gid,
    bnx, bny, bnz, bnmax,
    center, axis
):
    # Number of blocks in x, y and z direction
    bn = np.asarray([bnx, bny, bnz])
    gid_ind = 2 * axis

    # Check if there are neighbours availabe
    # for all blocks in + and - direction of the given axis.
    # We should use the GID here as it contains all neighbours
    # Select all blocks at the outer most coordinate in given axis
    sel_blocks_p = coords[minref_blist, axis] == coords[minref_blist, axis].max()
    sel_blocks_m = coords[minref_blist, axis] == coords[minref_blist, axis].min()

    # Index of neighbour blocks in +/- direction
    gid_neigh_p = gid[minref_blist[sel_blocks_p], gid_ind + 1] - 1
    gid_neigh_m = gid[minref_blist[sel_blocks_m], gid_ind] - 1
    
    # Check that all blocks have a neighbour
    nb_avail_p = np.all(gid_neigh_p >= 0)
    nb_avail_m = np.all(gid_neigh_m >= 0)

    # List of block id in +/- direction
    new_b_p = []
    new_b_m = []
    
    # Check if we reached the required number of blocks (bnmax)
    # at lowest ref. level to fill the cube.
    # If the difference is greater than 1 we add blocks on both sides
    # Else we only add blocks on one side depending which side
    # keeps center closer to the given center.
    if bnmax - bn[axis] >= 2:
        # Add blocks on both sides if neighbours are available
        if nb_avail_p:
            # Get list of neighbours in + direction
            new_b_p = gid_neigh_p
            
            # Check that it has the correct number of entries
            # One for each input block
            if new_b_p.size == int(np.prod(bn) / bn[axis]):
                # If so, increase counter
                bn[axis] += 1
            else:
                # Else reset the list
                new_b_p = []
        
        if nb_avail_m:
            # Get list of neighbours in - direction
            new_b_m = gid_neigh_m
            
            # Check that it has the correct number of entries
            # One for each input block
            if new_b_m.size == int(np.prod(bn) / bn[axis]):
                # If so, increase counter
                bn[axis] += 1
            else:
                # # Else reset the list
                new_b_m = []

    elif bn[axis] != bnmax:
        # If both list are available we check which side is closer

        if nb_avail_p and nb_avail_m:
            trig = False
            # Get list of neighbours in + / -  direction
            new_b_p = gid_neigh_p
            new_b_m = gid_neigh_m
            
            # Determine the position of outer edge 
            # when each of the neighbours would be included
            boundariesl = (
                coords[new_b_m, axis] - 0.5 * block_size[new_b_m, axis]
            ).min()
            boundariesr = (
                coords[new_b_p, axis] + 0.5 * block_size[new_b_p, axis]
            ).max()
            
            # Check if the - blocks are keeping the 
            # center closer to the original center
            if np.abs(boundariesl - center[axis]) < np.abs(boundariesr - center[axis]):
                # Check that the size is correct
                # If not trigger that the + blocks should be included
                if new_b_m.size == int(np.prod(bn) / bn[axis]):
                    new_b_p = []
                    bn[axis] += 1
                else:
                    new_b_m = []
                    trig = True

            # Check if the + blocks are keeping the
            # center closer to the original center
            if np.abs(boundariesl - center[axis]) >= np.abs(boundariesr - center[axis]) or trig:
                if new_b_p.size == int(np.prod(bn) / bn[axis]):
                    new_b_m = []
                    bn[axis] += 1
                else:
                    new_b_p = []
                    new_b_m = []

        # Same as above
        elif nb_avail_p:
            new_b_p = gid_neigh_p

            if new_b_p.size == int(np.prod(bn) / bn[axis]):
                bn[axis] += 1
            else:
                new_b_p = []

        elif nb_avail_m:
            new_b_m = gid_neigh_m

            if new_b_m.size == int(np.prod(bn) / bn[axis]):
                bn[axis] += 1
            else:
                new_b_m = []

    # Add new blocks to the current list of blocks at lowest refinement level.
    minref_blist = np.unique(np.concatenate((minref_blist, new_b_p, new_b_m))).astype(int)

    return minref_blist, bn[0], bn[1], bn[2]


# Sort all blocks at minimum refinement level
# and replace block with their higher refinement
# level counterparts.
def create_blists(
    minref_blist, max_ref_lvl,
    block_level, gid, coords, 
    bnx=0, bny=0, bnz=0,
    is_cuboid=False, verbose=False,
    is_radmc=False
):

    # Put the block of the minimum refinement level
    # on a grid correspoding to their coordinates.
    # Change axes to be in order of x, y and z.
    blist_minsort_tmp = blocks_on_grid(
        b_ids=minref_blist, coords=coords,
        bnx=bnx, bny=bny, bnz=bnz
    ).swapaxes(0, 2)

    if verbose:
        print('bnx, bny, bnz: ', bnx, bny, bnz)
        print('Block shape: ', blist_minsort_tmp.shape)
    
    blist_minsort = []
    
    # Add blocks in amr order to list.
    if is_cuboid:
        blist_minsort = blist_minsort_tmp.swapaxes(0, 2).flatten()
    
    else:
        for pos in zenumerate(blist_minsort_tmp.shape):
            blist_minsort.append(blist_minsort_tmp[pos])

    # Check all blocks if they have children (!-1)
    # using the gid and replace the corresponding block with them.
    maxref_blist = np.asarray(blist_minsort)
    gid_tmp = gid[maxref_blist, 7:]
    tot_nr_blocks = maxref_blist.size

    # For the FLASH-PP-Pipeline we still require this mode.
    # As we have to preserve the correct order of blocks
    # in the cuboid mode.
    # Possibly one can speed up this part but this is not
    # necessary for the pipeline.
    if is_radmc:
        # Goes through all refinement levels one by one.
        for j in range(max_ref_lvl - block_level):
            # Adds from back to front to circumvent
            # changing the position every time blocks are added.
            for k in reversed(range(len(gid_tmp))):
                if gid_tmp[k, 0] != -1:
                    maxref_blist = np.delete(maxref_blist, k)
                    maxref_blist = np.insert(maxref_blist, k, gid_tmp[k]-1)
                    tot_nr_blocks += 8
            gid_tmp = gid[maxref_blist, 7:]

    else:
        # This is much faster than the old routine
        # Especially for larger simulations (> 100k blocks)
        for j in range(max_ref_lvl - block_level):
            tmp_gid = gid[maxref_blist]
            sel_gid = tmp_gid[:, -1] > 0
            new_children = tmp_gid[sel_gid, 7:] - 1
            tmp_blist = maxref_blist[np.logical_not(sel_gid)].tolist()
            tmp_blist += new_children.flatten().tolist()
            tot_nr_blocks += new_children.size
            maxref_blist = np.sort(tmp_blist)

    return maxref_blist, tot_nr_blocks
    

# Arrage the block list onto a cuboid grid
def blocks_on_grid(b_ids, coords, bnx=0, bny=0, bnz=0):
    
    # Check if an entry is missing
    if bnx == 0 or bny == 0 or bnz == 0:
        # Determine next higher power which can
        # fit the blocks
        bn = 2**(int(np.log2(len(b_ids))) / 3)
        bnx = bn
        bny = bn
        bnz = bn

    # We sort the blocks by z-coordinate
    # and reshape the blocks to 2D list
    bgrid = np.reshape(b_ids[np.argsort(a=coords[b_ids, 2])], (int(bnz), int(bnx * bny)))

    # Loop over all z positions
    for i in range(int(bnz)):
        # Sort blocks by y-coordinate 
        # for each z-coordinate
        bgrid[i] = bgrid[i, np.argsort(coords[bgrid[i]][:, 1])]
    
    # Reshape to 3D structure
    bgrid = bgrid.reshape((int(bnz), int(bny), int(bnx)))

    # Loop over all z- and y-coordinates
    for i in range(int(bnz)):
        for j in range(int(bny)):
            # Sort each line by x-coordinate
            bgrid[i, j] = bgrid[i, j, np.argsort(coords[bgrid[i, j]][:, 0])]

    return bgrid
