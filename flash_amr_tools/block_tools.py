'''
Supplementary module to contain low-level routines for block determination
'''
import numpy as np
from .zorder import zenumerate

# Find all blocks at lowest refinement level which are correspoding to our initial guess
def find_blocks(
        block_list, min_ref_lvl, max_ref_lvl, brlvl, coords, block_size, bsmin, refine_level, gid, center, is_cuboid=False):

    # For all block which are higher than our set refinement level find the lowest parent which is fulfills our
    # requested refinement level.
    blist_ref = []
    for block in block_list[brlvl > min_ref_lvl]:
        b_tmp = block
        while refine_level[b_tmp] > min_ref_lvl:
            b_tmp = gid[b_tmp][6] - 1
        blist_ref.append(b_tmp)

    # As there will be many overlaps we create a unique list of block ids
    blist_ref = np.unique(blist_ref)
    # And add them to all the cells which already were at the lowest refinement level.
    minref_blist = np.unique(np.concatenate((block_list[brlvl == min_ref_lvl], blist_ref))).astype(int)

    # Check how many blocks fit into the current range in x, y and z at the lowest refinement level.
    bnx = np.round((np.amax(
        coords[minref_blist, 0] + 0.5 * block_size[minref_blist, 0])
        - np.amin(coords[minref_blist, 0] - 0.5 * block_size[minref_blist, 0])
        )/(bsmin * 2**(max_ref_lvl-min_ref_lvl)))

    bny = np.round((np.amax(
        coords[minref_blist, 1] + 0.5 * block_size[minref_blist, 1])
        - np.amin(coords[minref_blist, 1] - 0.5 * block_size[minref_blist, 1])
        )/(bsmin * 2**(max_ref_lvl-min_ref_lvl)))

    bnz = np.round((np.amax(
        coords[minref_blist, 2] + 0.5 * block_size[minref_blist, 2])
        - np.amin(coords[minref_blist, 2] - 0.5 * block_size[minref_blist, 2])
        )/(bsmin * 2**(max_ref_lvl-min_ref_lvl)))

    # Calculate the max number of blocks at lowest refinement level needed to be able to create an amr tree.
    bn = np.max([bnx, bny, bnz])
    bnmax = 2**np.ceil(np.log2(bn))

    # Check if we have all our blocks fill the cuboid at least in numbers. Let's hope this is never the case.
    # Currently no idea how to fix this.
    if not bnx * bny * bnz == minref_blist.shape[0]:
        print('Blocks in x: ', bnx)
        print('Blocks in y: ', bny)
        print('Blocks in z: ', bnz)
        print(minref_blist.shape)
        return 1, 1, 1, 1, 2
        # sys.exit('Could not find enough blocks to fill cuboid.')
    elif is_cuboid:
        return minref_blist, bnx, bny, bnz, bnmax

    # For each round add blocks at at least one side to reach the maximum number of blocks for amr tree.
    for i in range(int(bnmax - np.min([bnx, bny, bnz]) + 2)):

        minref_blist, bnx, bny, bnz = add_blocks(
            minref_blist=minref_blist, coords=coords, block_size=block_size, gid=gid, bnx=bnx, bny=bny, bnz=bnz,
            bnmax=bnmax, center=center, axis=0)

        minref_blist, bnx, bny, bnz = add_blocks(
            minref_blist=minref_blist, coords=coords, block_size=block_size, gid=gid, bnx=bnx, bny=bny, bnz=bnz,
            bnmax=bnmax, center=center, axis=1)

        minref_blist, bnx, bny, bnz = add_blocks(
            minref_blist=minref_blist, coords=coords, block_size=block_size, gid=gid, bnx=bnx, bny=bny, bnz=bnz,
            bnmax=bnmax, center=center, axis=2)

    return minref_blist, bnx, bny, bnz, bnmax


def add_blocks(minref_blist, coords, block_size, gid, bnx, bny, bnz, bnmax, center, axis):

    bn = np.asarray([bnx, bny, bnz])
    gid_ind = 2 * axis

    # Check if there are neghbours availabe for all blocks in + and - direction of the given axis.
    nb_avail_p = np.all(
        coords[
            gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].max()], gid_ind + 1] - 1, axis
        ] > coords[minref_blist, axis].max()
    )

    if nb_avail_p:
        if np.all(gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].max()], gid_ind + 1] - 1 < 0):
            nb_avail_p = False

    nb_avail_m = np.all(
        coords[
            gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].min()], gid_ind] - 1, axis
        ] < coords[minref_blist, axis].min()
    )

    if nb_avail_m:
        if np.all(gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].min()], gid_ind] - 1 < 0):
            nb_avail_m = False

    new_b_p = []
    new_b_m = []
    # Check if the maximum number of blocks at lowest ref. level are reached and if the difference greater than 1.
    # In that case we add blocks at either side, if available, otherwise we only add blocks at one side and check
    # which keeps the box center the closest to the given center.
    if bn[axis] != bnmax and bnmax - bn[axis] >= 2:
        if nb_avail_p:
            new_b_p = gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].max()], gid_ind + 1] - 1
            new_b_p = new_b_p[new_b_p > 0]
            if new_b_p.size == int(np.prod(bn)/bn[axis]):
                bn[axis] += 1
            else:
                new_b_p = []
        if nb_avail_m:
            new_b_m = gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].min()], gid_ind] - 1
            new_b_m = new_b_m[new_b_m > 0]
            if new_b_m.size == int(np.prod(bn)/bn[axis]):
                bn[axis] += 1
            else:
                new_b_m = []
    elif bn[axis] != bnmax:
        if nb_avail_p and nb_avail_m:
            trig = False
            new_b_p = gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].max()], gid_ind + 1] - 1
            new_b_p = new_b_p[new_b_p > 0]
            new_b_m = gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].min()], gid_ind] - 1
            new_b_m = new_b_m[new_b_m > 0]
            boundariesl = np.min(coords[new_b_m] - 0.5 * block_size[new_b_m], axis=axis)[axis]
            boundariesr = np.max(coords[new_b_p] + 0.5 * block_size[new_b_p], axis=axis)[axis]
            if np.abs(boundariesl - center[axis]) < np.abs(boundariesr - center[axis]):
                if new_b_m.size == int(np.prod(bn)/bn[axis]):
                    new_b_p = []
                    bn[axis] += 1
                else:
                    new_b_m = []
                    trig = True
            if np.abs(boundariesl - center[axis]) >= np.abs(boundariesr - center[axis]) or trig:
                if new_b_p.shape[axis] == int(np.prod(bn)/bn[axis]):
                    new_b_m = []
                    bn[axis] += 1
                else:
                    new_b_p = []
                    new_b_m = []
        elif nb_avail_p:
            new_b_p = gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].max()], gid_ind + 1] - 1
            new_b_p = new_b_p[new_b_p > 0]
            if new_b_p.size == int(np.prod(bn)/bn[axis]):
                bn[axis] += 1
            else:
                new_b_p = []
        elif nb_avail_m:
            new_b_m = gid[minref_blist[coords[minref_blist, axis] == coords[minref_blist, axis].min()], gid_ind] - 1
            new_b_m = new_b_m[new_b_m > 0]
            if new_b_m.size == int(np.prod(bn)/bn[axis]):
                bn[axis] += 1
            else:
                new_b_m = []

    # Add new blocks to the current list of blocks at lowest refinement level.
    minref_blist = np.unique(np.concatenate((minref_blist, new_b_p, new_b_m))).astype(int)

    return minref_blist, bn[0], bn[1], bn[2]


# Sort all blocks at minimum refinement level and replace block with their higher refinement level counterparts.
def create_blists(minref_blist, max_ref_lvl, block_level, gid, coords, bnx=0, bny=0, bnz=0, is_cuboid=False):

    # Put the block of the minimum refinement level on a grid correspoding to their coordinates.
    # Change axes to be in order of x, y and z.
    blist_minsort_tmp = blocks_on_grid(b_ids=minref_blist, coords=coords, bnx=bnx, bny=bny, bnz=bnz).swapaxes(0, 2)
    print('bnx, bny, bnz: ', bnx, bny, bnz)
    print('Block shape: ', blist_minsort_tmp.shape)
    blist_minsort = []
    # Add blocks in amr order to list.
    if is_cuboid:
        blist_minsort = blist_minsort_tmp.swapaxes(0, 2).flatten()
    else:
        for pos in zenumerate(blist_minsort_tmp.shape):
            blist_minsort.append(blist_minsort_tmp[pos])

    # Check all blocks if they have children (!-1) using the gid and replace the corresponding block with them.
    maxref_blist = np.asarray(blist_minsort)
    gid_tmp = gid[maxref_blist, 7:]
    tot_nr_blocks = maxref_blist.size
    # Goes through all refinement levels one by one.
    for j in range(max_ref_lvl - block_level):
        # Adds from back to front to circumvent changing the position every time blocks are added.
        for k in reversed(range(len(gid_tmp))):
            if gid_tmp[k, 0] != -1:
                maxref_blist = np.delete(maxref_blist, k)
                maxref_blist = np.insert(maxref_blist, k, gid_tmp[k]-1)
                tot_nr_blocks += 8
        gid_tmp = gid[maxref_blist, 7:]
    return maxref_blist, tot_nr_blocks
    

def blocks_on_grid(b_ids, coords, bnx=0, bny=0, bnz=0):

    if bnx == 0 or bny == 0 or bnz == 0:
        bn = 2**(int(np.log2(len(b_ids)))/3)
        bnx = bn
        bny = bn
        bnz = bn

    bgrid = np.reshape(b_ids[np.argsort(a=coords[b_ids, 2])], (int(bnz), int(bnx * bny)))

    for i in range(int(bnz)):
        bgrid[i] = bgrid[i, np.argsort(coords[bgrid[i]][:, 1])]
    bgrid = bgrid.reshape((int(bnz), int(bny), int(bnx)))

    for i in range(int(bnz)):
        for j in range(int(bny)):
            bgrid[i, j] = bgrid[i, j, np.argsort(coords[bgrid[i, j]][:, 0])]

    return bgrid