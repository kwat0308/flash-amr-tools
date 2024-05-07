'''
Toolkit to convert from AMR data in FLASH -> Uniform data
'''

import numpy as np
import os, sys
import h5py

# relative imports for finding blocks & creating block lists
from .block_tools import find_blocks, create_blists


def get_true_blocks(
        fname: str,
        xmin=np.array([], dtype=np.float32),
        xmax=np.array([], dtype=np.float32),
        is_cuboid=True,
        max_ref_given=None,
        min_ref_given=None,
    ):
    '''
    Extract the complete list of blocks which are in the box chosen within xmin and xmax. In particular, it sets the minimum & maximum refinement within the block list, and the number of blocks in x, y, z on the lowest refinement level.

    - fname (str): the filename of the FLASH output file. needs to be a .h5 file
    - xmin (3-D array): the lower corner of the region of interest. Defaults to empty array, which uses the whole domain
    - xmax (3-D array): the upper corner of the region of interest. Defaults to empty array, which uses the whole domain
    - is_cuboid (bool): set to True if the region is not a cube. Defaults to True.
    - max_ref_given (int): force the maximum refinement level of the blocks. Defaults to None, which uses the maximum refinement level in the simulation.
    - min_ref_given (int): force the minimum refinement level of the blocks. Defaults to None, which uses the minimum refinement level in the simulation.


    Returns:
    - blist (list) : the list of blocks that cover the region of interest 
    - brefs (list) : the maximum & minimum refinement level within the region of interest
    - bns (list) : returns the number of blocks in (x, y, z) direction at the lowest refinement level
    '''

    # makes sure that the filename exists
    if not os.path.exists(fname):
        raise FileNotFoundError("File %s is not found!" % fname)

    # Read in the FLASH file and create the 3d grid of the blocks.
    pf = h5py.File(name=fname, mode='r')

    # Define some pointers for regular used data tables.
    coords = pf['coordinates'][()]
    gid = pf['gid'][()]
    refine_level = pf['refine level'][()]
    block_size = pf['block size'][()]

    # Reading the border of the simulation box.
    sim_dict = dict(pf['real runtime parameters'][()])

    simxl = sim_dict[('%-80s' % 'xmin').encode('UTF-8')]
    simxr = sim_dict[('%-80s' % 'xmax').encode('UTF-8')]
    simyl = sim_dict[('%-80s' % 'ymin').encode('UTF-8')]
    simyr = sim_dict[('%-80s' % 'ymax').encode('UTF-8')]
    simzl = sim_dict[('%-80s' % 'zmin').encode('UTF-8')]
    simzr = sim_dict[('%-80s' % 'zmax').encode('UTF-8')]

    xmin = np.asarray(xmin)
    xmax = np.asarray(xmax)

    # Set region to whole simulation box if region is not given.
    if xmin.size == 0 or xmax.size == 0:
        xmin = np.asarray([simxl, simyl, simzl])
        xmax = np.asarray([simxr, simyr, simzr])

    # Checking if the selected box is located within the simulation.
    if np.any(xmax <= np.asarray([simxl, simyl, simzl])) or np.any(xmin >= np.asarray([simxr, simyr, simzr])):
        sys.exit(
            'Selected box is not located within simulation. Please check selected coordinates.\n'
            'Simulation box:\tx\t%1.8e\t%1.8e\n' % (simxl, simxr) +
            '\t\ty\t%1.8e\t%1.8e\n' % (simyl, simyr) +
            '\t\tz\t%1.8e\t%1.8e\n\n' % (simzl, simzr) +
            'Selected box:\tx\t%1.8e\t%1.8e\n' % (xmin[0], xmax[0]) +
            '\t\ty\t%1.8e\t%1.8e\n' % (xmin[1], xmax[1]) +
            '\t\tz\t%1.8e\t%1.8e\n' % (xmin[2], xmax[2])
        )

    # Sanity check coordinates
    if np.any(xmax < xmin):
        sys.exit(
            'Check coordinates. Some value in xmin is larger than xmax.\n' +
            'xmin : %s\t%s\t%s\n' % tuple(xmin) +
            'xmax : %s\t%s\t%s\n' % tuple(xmax)
        )

    print("minimum position of the region of interest: ", xmin)
    print("maximum position of the region of interest: ", xmax)

    # Find all leaf blocks for inner and outer borders
    # Inner borders only includes blocks which center coordinate is included in selected region.
    # Outer borders also includes blocks which intersect with the selected region. Those variables are denoted with a 2.
    print('Finding all leaf blocks in given region')
    ind = np.argwhere(np.all(gid[:, 7:] == -1, axis=1)).T[0]
    tmpc = coords[ind]
    tmpbs = block_size[ind] * 0.5

    c = np.all(np.concatenate((tmpc >= xmin, tmpc <= xmax), axis=1), axis=1)
    c2 = np.all(np.concatenate((tmpc + tmpbs >= xmin, tmpc - tmpbs <= xmax), axis=1), axis=1)

    blist_raw = ind[c]
    blist_raw2 = ind[c2]

    if len(blist_raw) == 0:
        sys.exit(
            'No block center is availabe in the selected region.'
        )

    print(blist_raw.size, blist_raw2.size)
    bsmin = block_size[ind].min()

    brlvl = refine_level[blist_raw]
    brlvl2 = refine_level[blist_raw2]

    min_ref = brlvl.min()
    min_ref2 = brlvl2.min()
    
    if max_ref_given != None:
        print('Force maximum refinement level: %s' % max_ref_given)
        if max_ref_given < min_ref:
            min_ref = max_ref_given
        if max_ref_given < min_ref2:
            min_ref2 = max_ref_given

    if min_ref_given != None:
        print('Force minimum common refinement level: %s' % min_ref_given)
        if min_ref_given < min_ref:
            min_ref = min_ref_given
        if min_ref_given < min_ref2:
            min_ref2 = min_ref_given

    max_ref_reg = brlvl.max()
    max_ref_reg2 = brlvl2.max()

    max_ref = refine_level[ind].max()
    print('Biggest block has refinement level: %s' % min_ref)
    print('Highest refinement level in simulation: %s' % max_ref)
    print('Highest refinement level in selected box: %s' % max_ref_reg)

    print('Extending region to fit amr structure at level %s.' % min_ref)
    blist_minref, bx, by, bz, bmax = find_blocks(
        block_list=blist_raw, min_ref_lvl=min_ref, max_ref_lvl=max_ref, block_size=block_size, brlvl=brlvl, bsmin=bsmin,
        coords=coords, gid=gid, refine_level=refine_level, center=(xmin+xmax)/2., is_cuboid=is_cuboid
    )

    while np.any([bx != bmax, by != bmax, bz != bmax]):
        print(
            'Trying extended region given by including blocks whose center is not in predefined region but part '
            'of their block volume is.')

        blist_minref, bx, by, bz, bmax = find_blocks(
            block_list=blist_raw2, min_ref_lvl=min_ref2, max_ref_lvl=max_ref, block_size=block_size, brlvl=brlvl2,
            bsmin=bsmin, coords=coords, gid=gid, refine_level=refine_level, center=(xmin + xmax) / 2., is_cuboid=is_cuboid
        )

        if is_cuboid and type(blist_minref) != int:
            min_ref = min_ref2
            brlvl = brlvl2
            blist_raw = blist_raw2
            max_ref_reg = max_ref_reg2
            break

        if bx == bmax and by == bmax and bz == bmax:
            min_ref = min_ref2
            brlvl = brlvl2
            blist_raw = blist_raw2
            max_ref_reg = max_ref_reg2
            print('New biggest block has refinement level: %s' % min_ref)
            print('New highest refinement level in simulation: %s' % max_ref)
            print('New highest refinement level in selected box: %s' % max_ref_reg)

        else:
            min_ref -= 1
            min_ref2 -= 1
            if min_ref == 0:
                sys.exit('Region could not be extended to a cubic block shape.')

            print('Could not extend region far enough. Extending again at level %s' % min_ref)
            blist_minref, bx, by, bz, bmax = find_blocks(
                block_list=blist_raw, min_ref_lvl=min_ref, max_ref_lvl=max_ref, block_size=block_size, brlvl=brlvl,
                bsmin=bsmin, coords=coords, gid=gid, refine_level=refine_level, center=(xmin+xmax)/2., is_cuboid=is_cuboid
            )

            if is_cuboid and type(blist_minref) != int:
                break

    if is_cuboid:
        blvl = min_ref
    else:
        blvl = min_ref - np.int32(np.round(np.log2(bmax)))
        
    if max_ref_given != None:
        max_ref = max_ref_given

    blist_maxref, b_tot_nr = create_blists(
        minref_blist=blist_minref, block_level=blvl, gid=gid, coords=coords, max_ref_lvl=max_ref,
        bnx=bx, bny=by, bnz=bz, is_cuboid=is_cuboid
    )

    if not is_cuboid:
        b_tot_nr += np.sum(np.logspace(start=0., stop=np.log2(bmax)-1, base=2, num=int(np.log2(bmax)))**3).astype(int)

    print('Total number of blocks including parent blocks: %s' % b_tot_nr)

    boundariesl = np.min(coords[blist_maxref] - 0.5 * block_size[blist_maxref], axis=0)
    boundariesr = np.max(coords[blist_maxref] + 0.5 * block_size[blist_maxref], axis=0)
    print('Region extended to:\tlower\t\tupper')
    print('\tx\t%+1.8e\t\t%+1.8e' % (boundariesl[0], boundariesr[0]))
    print('\ty\t%+1.8e\t\t%+1.8e' % (boundariesl[1], boundariesr[1]))
    print('\tz\t%+1.8e\t\t%+1.8e\n' % (boundariesl[2], boundariesr[2]))

    print('Region center at:\t%+1.8e\t%+1.8e\t%+1.8e' % tuple((boundariesl+boundariesr)/2))
    print('Given center:\t\t%+1.8e\t%+1.8e\t%+1.8e' % tuple((xmin+xmax)/2))

    # return min & maximum refinement levels in the block, block list, and number of blocks in each direction
    bns = [bx, by, bz]
    brefs = [max_ref, min_ref]

    return blist_maxref, brefs, bns


def get_cube(data, ref_lvl, bbox, bsize, brefs, bns):
    '''
    Transforms the data as a 3-D uniform cube.

    - data: data extracted by h5py filtered by the block list
    - ref_lvl : refinement level for each block, filtered by the block list
    - bbox : the bounding box 
    - bsize : the block size, used to calculate the coordinates
    - brefs : the maximum and minimum refinement level in the region of interest
    - bns : number of blocks in each direction at lowest refinement level
    '''
    # unpack the list
    bmax, bmin = brefs
    bnx, bny, bnz = bns
    
    # maximum level, used to create the uniform grid
    max_lvl = bmax - bmin
    
    # computing the coordiante using the lower & upper corners divided by the smallest block size
    bsize = bsize.min(axis=0)
    coords = np.round((bbox[:, :, 0] - bbox[0, :, 0]) / bsize)
    coords = coords.astype(int) * 8
    
    # shift refinement level by the maximum within the region of interest
    ref_lvl = bmax - ref_lvl
    
    cube = np.zeros((int(bnx) * 2**(max_lvl+3), int(bny) * 2**(max_lvl+3), int(bnz) * 2**(max_lvl+3)), dtype=data.dtype)

    for i in range(len(data)):
        x, y, z = coords[i]
        
        diff = 2**(ref_lvl[i])
        
        data_reshape = np.repeat(data[i], diff, axis=0)
        data_reshape = np.repeat(data_reshape, diff, axis=1)
        data_reshape = np.repeat(data_reshape, diff, axis=2)
        
        sub_cube_size = int(2**(ref_lvl[i] + 3))
        cube[x:x+sub_cube_size, y:y+sub_cube_size, z:z+sub_cube_size] = data_reshape.T
        
    return cube.T


def get_reflvl_cube(ref_lvl, bbox, bsize, brefs, bns):
    '''
    Returns the refinement level as a 3-D uniform cube.

    - ref_lvl : refinement level for each block, filtered by the block list
    - bbox : the bounding box 
    - bsize : the block size, used to calculate the coordinates
    - brefs : the maximum and minimum refinement level in the region of interest
    - bns : number of blocks in each direction at lowest refinement level
    '''
    # unpack the list
    bmax, bmin = brefs
    bnx, bny, bnz = bns
    
    # maximum level, used to create the uniform grid
    max_lvl = bmax - bmin

    # computing the coordiante using the lower & upper corners divided by the smallest block size
    bsize = bsize.min(axis=0)
    coords = np.round((bbox[:, :, 0] - bbox[0, :, 0]) / bsize)
    coords = coords.astype(int) * 8
    
    # shift refinement level by the maximum within the region of interest
    ref_lvl = bmax - ref_lvl

    cube = np.zeros((int(bnx) * 2 ** (max_lvl + 3), int(bny) * 2 ** (max_lvl + 3), int(bnz) * 2 ** (max_lvl + 3)), dtype=ref_lvl.dtype)

    for i in range(len(ref_lvl)):
        x, y, z = coords[i]

        diff = 2 ** (ref_lvl[i]+3)

        data_reshape = np.repeat(ref_lvl[i], diff, axis=0)[..., None]
        data_reshape = np.repeat(data_reshape, diff, axis=1)[..., None]
        data_reshape = np.repeat(data_reshape, diff, axis=2)

        sub_cube_size = int(2 ** (ref_lvl[i]+3))
        cube[x:x + sub_cube_size, y:y + sub_cube_size, z:z + sub_cube_size] = data_reshape.T

    return cube.T


def get_vector_cube(data_vec, ref_lvl, bbox, bsize, brefs, bns):
    '''
    Wrapper for returning vectorial data as a uniform grid, where the last axis appends the data in each direction.

    - data_vec (list): list of length 3, containing each data in each direction.  extracted by h5py & filtered by the block list
    - bbox : the bounding box 
    - bsize : the block size, used to calculate the coordinates
    - brefs : the maximum and minimum refinement level in the region of interest
    - bns : number of blocks in each direction at lowest refinement level
    '''
    # unpack the list
    bmax, bmin = brefs
    bnx, bny, bnz = bns
    
    # maximum level, used to create the uniform grid
    max_lvl = bmax - bmin

    vec_cube = np.zeros((
        int(bnx) * 2 ** (max_lvl + 3),
        int(bny) * 2 ** (max_lvl + 3),
        int(bnz) * 2 ** (max_lvl + 3),
        3
    ), dtype=data_vec[0].dtype)

    for i in range(3):
        vec_cube[..., i] = get_cube(data_vec[i], ref_lvl, bbox, bsize, brefs, bns)

    return vec_cube


def calc_cube_size(brefs, bns, dtype=np.float32):
    '''
    Calculate the size of the cube in cell units
    
    - brefs : the minimum and maximum refinement level in the region of interest
    - bns : number of blocks in each direction at lowest refinement level
    - dtype: datatype of the cube size. Defaults to float32
    '''
    # unpack the list
    bmax, bmin = brefs
    bnx, bny, bnz = bns
    
    # maximum level, used to create the uniform grid
    max_lvl = bmax - bmin

    # number of cells
    cells = int(bnx) * 2**(max_lvl+3) * int(bny) * 2**(max_lvl+3) * int(bnz) * 2**(max_lvl+3)
    return cells * np.dtype(dtype).itemsize


def get_cdens(data, axis, ref_lvl, bbox, bsize, brefs, bns, weights=None):
    '''
    Get the column density of the data along the axis.

    NB: Only on-axis projections are currently supported.

    - data: data extracted by h5py filtered by the block list
    - axis : the axis in which we want to project [0, 1, or 2]
    - ref_lvl : refinement level for each block, filtered by the block list
    - bbox : the bounding box 
    - bsize : the block size, used to calculate the coordinates
    - brefs : the maximum and minimum refinement level in the region of interest
    - bns : number of blocks in each direction at lowest refinement level
    - weights : optional argument to pass weights for weighted projections. must be the same shape as data. Defaults to None, which performs unweighted projections.
    '''
    # unpack the list
    bmax, bmin = brefs
    
    # maximum level, used to create the uniform grid
    max_lvl = bmax - bmin

    # safety features
    assert axis in [0,1,2], f"Axis {axis} is not 0, 1, or 2 (x, y or z)."

    # remove axis in which projection occrurs
    bn_ax = bns.pop(axis)
    ax = [0, 1, 2]
    ax.pop(axis)

    coords = np.round((bbox[:, :, 0] - bbox[0, :, 0]) / bsize.min())
    coords = coords.astype(int) * 8

    ref_lvl = bmax - ref_lvl
    sh = data.shape

    use_weights = False
    if weights is not None:
        norm = np.zeros((int(bns[0]) * 2**(max_lvl+3), int(bns[1]) * 2**(max_lvl+3)), dtype=data.dtype)

        # make sure shape of weights is same as data
        assert data.shape == weights.shape, "shape of weights " + weights.shape + " != shape of data " + data.shape + " ."

        use_weights = True

    cdens = np.zeros((int(bns[0]) * 2**(max_lvl+3), int(bns[1]) * 2**(max_lvl+3)), dtype=data.dtype)

    for i in range(len(data)):
        xyz = coords[i].tolist()

        xyz.pop(axis)

        diff = 2**(ref_lvl[i])
        sub_cube_size = int(2**(ref_lvl[i] + 3))

        if use_weights:
            tmp_data = (data[i] * weights[i]).T.sum(axis=axis)
            weights_reshape = np.repeat(weights[i].T.sum(axis=axis), diff, axis=0)
            weights_reshape = np.repeat(weights_reshape, diff, axis=1)
            norm[xyz[0]:xyz[0]+sub_cube_size, xyz[1]:xyz[1]+sub_cube_size] += weights_reshape
        else:
            tmp_data = data[i].T.sum(axis=axis) * bsize[i, axis] / sh[axis]

        data_reshape = np.repeat(tmp_data, diff, axis=0)
        data_reshape = np.repeat(data_reshape, diff, axis=1)

        cdens[xyz[0]:xyz[0]+sub_cube_size, xyz[1]:xyz[1]+sub_cube_size] += data_reshape
    if use_weights:
        cdens /= norm
    return cdens.T 


def get_slice(data, pos, axis, ref_lvl, bbox, bsize, brefs, bns):
    '''
    Get the slice of the data at the current position.

    - data: data extracted by h5py filtered by the block list
    - pos: 3-D vector of the position we want to slice at
    - axis: axis in which we want to take the slice in
    - ref_lvl : refinement level for each block, filtered by the block list
    - bbox : the bounding box 
    - bsize : the block size, used to calculate the coordinates
    - brefs : the maximum and minimum refinement level in the region of interest
    - bns : number of blocks in each direction at lowest refinement level
    '''

    # unpack the list
    bmax, bmin = brefs
    
    # Get the maximum level difference between base blocks 
    # These don't have to be root blocks of the simulation
    max_lvl = bmax - bmin

    bshape = data.shape[1:4]

    # safety features
    assert axis in [0,1,2], f"Axis {axis} is not 0, 1, or 2 (x, y or z)."

    # As the data and coordinates are stored in different order
    # we need to derive the axis we have for the data insertion seperately
    ax = [0, 1, 2]
    ax_c = ax.pop(axis)
    pax = [2, 1, 0]
    # This is the axis which we slice over
    pax_c = pax.pop(axis)
    # Remaining axis are left for expansion if necessary

    # Derive coordinates from the lower corner of the bounding box
    coords = bbox[:, :, 0] - bbox[0, :, 0]
    # Shift the axis such that 0 in the selected axis is our required slice
    # Normalise to smallest block size
    coords /= bsize.min()
    # Round values to get next cell position
    coords = np.round(coords)
    # Convert to integer as these are now direct indicies
    coords = coords.astype(int) * bshape
    
    tmp_id = (pos - bbox[:, axis, 0]) / bsize[:, axis] * bshape[axis]
    tmp_id = tmp_id.astype(int)
    # Create array where we store our slice
    sl_ax = np.zeros((int(bns[ax[0]]) * 2**(max_lvl+3), int(bns[ax[1]]) * 2**(max_lvl+3)))
    # Shift the refinement level such that it represent the differen to the highest level
    ref_lvl = bmax - ref_lvl

    # Loop over all blocks
    for i in range(len(data)):
        # Get indicies for current lower corner of the block
        xyz = coords[i].tolist()
        # Get the lower index along the slice axis
        pi = xyz.pop(axis)

        # Get factor of block size in relation to smallest block
        diff = 2**(ref_lvl[i])
        
        # Check if our 0 index is within the range of the current block
        if tmp_id[i] < 0 or tmp_id[i] >= 8:
            continue
        
        # Create general slice array
        sel = [slice(None, None, None), slice(None, None, None), slice(None, None, None)]
        sel[ax_c] = slice(tmp_id[i], tmp_id[i]+1, None)

        # Increase block size to fit the refinement level
        data_reshape = np.repeat(data[i].T[sel[0], sel[1], sel[2]], diff, axis=ax[0])
        data_reshape = np.repeat(data_reshape, diff, axis=ax[1])
        # Get size of the slice patch
        sub_cube_size = int(2**(ref_lvl[i] + 3))
        # Store in slice array, needs to be transposed to convert from ZYX to XYZ
        sl_ax[xyz[0]:xyz[0]+sub_cube_size, xyz[1]:xyz[1]+sub_cube_size] = np.squeeze(data_reshape)

    return sl_ax.T
