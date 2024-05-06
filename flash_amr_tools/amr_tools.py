import numpy as np
import os, sys
import h5py
from .zorder import zenumerate

class AMRToolkit:
    '''
    Toolkit to convert from AMR data in FLASH -> Uniform data
    '''

    def __init__(
            self,
            fname: str,
            xmin=np.array([], dtype=np.float32),
            xmax=np.array([], dtype=np.float32),
            is_cuboid=True,
            max_ref_given=None,
            min_ref_given=None,
    ):
        '''
        Toolkit to convert from AMR data in FLASH -> Uniform data

        - fname (str): the filename of the FLASH output file. needs to be a .h5 file

        - xmin (3D vector): the lower corner of the region of interest. defaults to empty list, which uses the whole domain

        - xmax (3D vector): the upper corner of the region of interest. defaults to empty list, which uses the whole domain

        - is_cuboid (bool): set to True if the region is not a cube. Defaults to True.

        - max_ref_given (int): force the maximum refinement level of the blocks. Defaults to None, which uses the maximum refinement level in the simulation.

        - min_ref_given (int): force the minimum refinement level of the blocks. Defaults to None, which uses the minimum refinement level in the simulation.

        '''
        self.fname = fname  # filename of interest
        self.xmin = xmin    # lower corner of region of interest
        self.xmax = xmax    # upper corner of region of interest

        # makes sure that the filename exists
        if not os.path.exists(self.fname):
            raise FileNotFoundError("File %s is not found!" % self.fname)
        
        # gets the complete list of blocks which are in the box chosen with xmin and xmax 
        bmin, bmax, blist, bnx, bny, bnz = self._get_true_blocks(is_cuboid, max_ref_given, min_ref_given)

        # NB: typecasting here for safety
        self.min_ref = int(bmin)    # minimum refinement level in block list
        self.max_ref = int(bmax)    # maximum refinement level in block list

        self.blist = blist    # block list
        self.bnx = int(bnx)         # number of blocks in x direction
        self.bny = int(bny)         # number of blocks in y direction
        self.bnz = int(bnz)         # number of blocks in z direction

        # get the bounding box and block size from the obtained block list
        # also get the refinement level 
        with h5py.File(self.fname, "r") as pf:
            self.bbox = pf['bounding box'][()][blist]
            self.bsize = pf['block size'][()][blist]

        # store the lower and upper corner
        self.low_cor = self.bbox[0, :, 0]
        self.up_cor = self.bbox[-1, :, 1]

        self.max_lvl = self.max_ref - self.min_ref    # maximum level


    def calc_cube_size(self, dtype=np.float32):
        '''
        Calculate the size of the cube in cell units
        
        - dtype: datatype of the cube size. Defaults to float32
        '''
        cells = self.bnx * 2**(self.max_lvl+3) * self.bny * 2**(self.max_lvl+3) * self.bnz * 2**(self.max_lvl+3)
        return cells * np.dtype(dtype).itemsize
    
    def get_cube(self, field : str):
        '''
        Get the data as a 3-D uniform cube.

        - field: field variable of the data we want to extract. name must follow the convention from flash.par
        '''
        
        bs = self.bsize.min(axis=0)
        coords = np.round((self.bbox[:, :, 0] - self.bbox[0, :, 0]) / bs)
        coords = coords.astype(int) * 8

        # extract refinement level and data
        with h5py.File(self.fname, "r") as pf:
            ref_lvl = pf['refine level'][()][self.blist]

            if field not in list(pf.keys()):
                raise KeyError("Field %s not found in %s!" % (field, self.fname))
            
            data = pf[field][()][self.blist]
        
        ref_lvl = self.max_ref - ref_lvl
        
        cube = np.zeros((self.bnx * 2**(self.max_lvl+3), self.bny * 2**(self.max_lvl+3), self.bnz * 2**(self.max_lvl+3)), dtype=data.dtype)

        for i in range(len(data)):
            x, y, z = coords[i]
            
            diff = 2**(ref_lvl[i])
            
            data_reshape = np.repeat(data[i], diff, axis=0)
            data_reshape = np.repeat(data_reshape, diff, axis=1)
            data_reshape = np.repeat(data_reshape, diff, axis=2)
            
            sub_cube_size = int(2**(ref_lvl[i] + 3))
            cube[x:x+sub_cube_size, y:y+sub_cube_size, z:z+sub_cube_size] = data_reshape.T
            
        return cube
    
    def get_reflvl_cube(self):
        '''
        Get the refinement level as a 3-D uniform cube.
        '''
        
        bs = self.bsize.min(axis=0)
        coords = np.round((self.bbox[:, :, 0] - self.bbox[0, :, 0]) / bs)
        coords = coords.astype(int) * 8

        # extract refinement level locally
        with h5py.File(self.fname, "r") as pf:
            ref_lvl = pf['refine level'][()][self.blist]
        
        ref_lvl = self.max_ref - ref_lvl
        
        cube = np.zeros((self.bnx * 2**(self.max_lvl+3), self.bny * 2**(self.max_lvl+3), self.bnz * 2**(self.max_lvl+3)), dtype=ref_lvl.dtype)

        for i in range(len(ref_lvl)):
            x, y, z = coords[i]
            
            diff = 2**(ref_lvl[i]+3)
            
            data_reshape = np.repeat(ref_lvl[i], diff, axis=0)
            data_reshape = np.repeat(data_reshape, diff, axis=1)
            data_reshape = np.repeat(data_reshape, diff, axis=2)
            
            sub_cube_size = int(2**(ref_lvl[i] + 3))
            cube[x:x+sub_cube_size, y:y+sub_cube_size, z:z+sub_cube_size] = data_reshape.T
            
        return cube
    
    def get_vector_cube(self, field : str):
        '''
        Get the data as a 3-dimensional list of 3-D uniform cubes for vector data.

        - field: field variable of the data we want to extract.
        Note:
            - the field variable must be of length 3 (ex. "vel" or "mag")
            - the field variable name must be contained in flash.par, with suffixes of "x", "y", "z" as the fourth letter
        '''

        vector_suffixes = ["x", "y", "z"]
        field_veclabels = [field + vsf for vsf in vector_suffixes]

        # extract refinement level and data
        with h5py.File(self.fname, "r") as pf:
            for field_veclabel in field_veclabels:
                if field_veclabel not in list(pf.keys()):
                    raise KeyError("Field %s not found in %s!" % (field_veclabel, self.fname))

            # get the data type of one entry
            field_dtype = pf[field_veclabels[0]][()].dtype

        vec_cube = np.zeros((
            self.bnx * 2 ** (self.max_lvl + 3),
            self.bny * 2 ** (self.max_lvl + 3),
            self.bnz * 2 ** (self.max_lvl + 3),
            3
        ), dtype=field_dtype)

        for i in range(3):
            vec_cube[..., i] = self.get_cube(field_veclabels[i])

        return vec_cube
    
    def get_data(self, field : str):
        '''
        Get the field data from the h5 file
        
        - field: field variable of the data we want to extract. name must follow the convention from flash.par
        '''
        
        with h5py.File(self.fname, "r") as pf:
            if field not in list(pf.keys()):
                raise KeyError("Field %s not found in %s!" % (field, self.fname))
            
            field_arr = pf[field][()][self.blist]

        return field_arr
    

    def get_slice(self, field : str, pos : float = 0.0, axis : int = 2):
        '''
        Get the slice of the data at the current position.

        - field: field variable of the data we want to extract. name must follow the convention from flash.par

        - pos: 3-D vector of the position we want to slice at

        - axis: axis in which we want to take the slice in
        '''

        # safety features
        assert axis in [0,1,2], "Axis %d is not 0, 1, or 2 (x, y or z)." % axis

        # extract refinement level & field
        with h5py.File(self.fname, "r") as pf:
            ref_lvl = pf['refine level'][()][self.blist]

            if field not in list(pf.keys()):
                raise KeyError("Field %s not found in %s!" % (field, self.fname))
            
            data = pf[field][()][self.blist]

        bshape = data.shape[1:4]

        # Get the number of base blocks in remaining axis
        bn = [self.bnx, self.bny, self.bnz]

        # As the data and coordinates are stored in different order
        # we need to derive the axis we have for the data insertion seperately
        ax = [0, 1, 2]
        ax_c = ax.pop(axis)
        pax = [2, 1, 0]
        # This is the axis which we slice over
        pax_c = pax.pop(axis)
        # Remaining axis are left for expansion if necessary

        # Derive coordinates from the lower corner of the bounding box
        coords = self.bbox[:, :, 0] - self.bbox[0, :, 0]
        # Shift the axis such that 0 in the selected axis is our required slice
        # Normalise to smallest block size
        coords /= self.bsize.min()
        # Round values to get next cell position
        coords = np.round(coords)
        # Convert to integer as these are now direct indicies
        coords = coords.astype(int) * bshape
        
        tmp_id = (pos - self.bbox[:, axis, 0]) / self.bsize[:, axis] * bshape[axis]
        tmp_id = tmp_id.astype(int)
        # Create array where we store our slice
        sl_ax = np.zeros((int(bn[ax[0]]) * 2**(self.max_lvl+3), int(bn[ax[1]]) * 2**(self.max_lvl+3)))

        # Shift the refinement level such that it represent the differen to the highest level
        ref_lvl = self.max_ref - ref_lvl

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

        return sl_ax
    
    def get_cdens(self, field : str, axis : int = 2, weights_field : str = ""):
        '''
        Get the column density of the data along the axis.

        NB: Only on-axis projections are currently supported.

        - field: field variable of the data we want to extract. name must follow the convention from flash.par

        - axis: the axis in which we want to project onto.

        - weights_field: optional argument to add weights to the projection. name must follow the convention from flash.par
        '''
        # internal flag to check if we use weights or not
        use_weights = False

        # safety features
        assert axis in [0,1,2], f"Axis {axis} is not 0, 1, or 2 (x, y or z)."

        # extract refinement level & field
        with h5py.File(self.fname, "r") as pf:
            ref_lvl = pf['refine level'][()][self.blist]

            if field not in list(pf.keys()):
                raise KeyError("Field %s not found in %s!" % (field, self.fname))
            
            data = pf[field][()][self.blist]

            # extract the weights if an argument is given
            if weights_field != "":
                if weights_field not in list(pf.keys()):
                    raise KeyError("Field %s not found in %s!" % (weights_field, self.fname))
                
                weights = pf[weights_field][()][self.blist]
                use_weights = True


        if use_weights:
            assert data.shape == weights.shape, "shape of weights " + weights.shape + " != shape of data " + data.shape + " ."

        # Get the number of base blocks in remaining axis
        bn = [self.bnx, self.bny, self.bnz]
        _ = bn.pop(axis)

        ax = [0, 1, 2]
        ax.pop(axis)

        coords = np.round((self.bbox[:, :, 0] - self.bbox[0, :, 0]) / self.bsize.min())
        coords = coords.astype(int) * 8

        # Shift the refinement level such that it represent the differen to the highest level
        ref_lvl = self.max_ref - ref_lvl

        sh = data.shape

        
        if use_weights:
            norm = np.zeros((int(bn[0]) * 2**(self.max_lvl+3), int(bn[1]) * 2**(self.max_lvl+3)), dtype=data.dtype)
            use_weights = True

        cdens = np.zeros((int(bn[0]) * 2**(self.max_lvl+3), int(bn[1]) * 2**(self.max_lvl+3)), dtype=data.dtype)

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
                tmp_data = data[i].T.sum(axis=axis) * self.bsize[i, axis] / sh[axis]

            data_reshape = np.repeat(tmp_data, diff, axis=0)
            data_reshape = np.repeat(data_reshape, diff, axis=1)

            cdens[xyz[0]:xyz[0]+sub_cube_size, xyz[1]:xyz[1]+sub_cube_size] += data_reshape
        if use_weights:
            cdens /= norm
        return cdens 
    

    def _get_true_blocks(self, is_cuboid=True, max_ref_given=None, min_ref_given=None):
        '''
        Extract the complete list of blocks which are in the box chosen within xmin and xmax. In particular, it sets the minimum & maximum refinement within the block list, and the number of blocks in x, y, z on the lowest refinement level.
        
        - is_cuboid (boolean): set to True if the region is not a cube. Defaults to True.

        - max_ref_given (int): force the maximum refinement level of the blocks. Defaults to None, which uses the maximum refinement level in the simulation.

        - min_ref_given (int): force the minimum refinement level of the blocks. Defaults to None, which uses the minimum refinement level in the simulation.

        '''
        # Read in the FLASH file and create the 3d grid of the blocks.
        pf = h5py.File(name=self.fname, mode='r')

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

        xmin = np.asarray(self.xmin)
        xmax = np.asarray(self.xmax)

        print(f"xmin, xmax: {xmin}, {xmax}")

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
        blist_minref, bx, by, bz, bmax = self._find_blocks(
            block_list=blist_raw, min_ref_lvl=min_ref, max_ref_lvl=max_ref, block_size=block_size, brlvl=brlvl, bsmin=bsmin,
            coords=coords, gid=gid, refine_level=refine_level, center=(xmin+xmax)/2., is_cuboid=is_cuboid
        )

        while np.any([bx != bmax, by != bmax, bz != bmax]):
            print(
                'Trying extended region given by including blocks whose center is not in predefined region but part '
                'of their block volume is.')

            blist_minref, bx, by, bz, bmax = self._find_blocks(
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
                blist_minref, bx, by, bz, bmax = self._find_blocks(
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

        blist_maxref, b_tot_nr = self._create_blists(
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

        return min_ref, max_ref, blist_maxref, bx, by, bz


    def _blocks_on_grid(self, b_ids, coords, bnx=0, bny=0, bnz=0):

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
    

    def _create_blists(self, minref_blist, max_ref_lvl, block_level, gid, coords, bnx=0, bny=0, bnz=0, is_cuboid=False):
        '''Sort all blocks at minimum refinement level and replace block with their higher refinement level counterparts.'''

        # Put the block of the minimum refinement level on a grid correspoding to their coordinates.
        # Change axes to be in order of x, y and z.
        blist_minsort_tmp = self._blocks_on_grid(b_ids=minref_blist, coords=coords, bnx=bnx, bny=bny, bnz=bnz).swapaxes(0, 2)
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
    
    def _add_blocks(self, minref_blist, coords, block_size, gid, bnx, bny, bnz, bnmax, center, axis):

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
    
    # Find all blocks at lowest refinement level which are correspoding to our initial guess
    def _find_blocks(
        self, block_list, min_ref_lvl, max_ref_lvl, brlvl, coords, block_size, bsmin, refine_level, gid, center, is_cuboid=False):

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

            minref_blist, bnx, bny, bnz = self._add_blocks(
                minref_blist=minref_blist, coords=coords, block_size=block_size, gid=gid, bnx=bnx, bny=bny, bnz=bnz,
                bnmax=bnmax, center=center, axis=0)

            minref_blist, bnx, bny, bnz = self._add_blocks(
                minref_blist=minref_blist, coords=coords, block_size=block_size, gid=gid, bnx=bnx, bny=bny, bnz=bnz,
                bnmax=bnmax, center=center, axis=1)

            minref_blist, bnx, bny, bnz = self._add_blocks(
                minref_blist=minref_blist, coords=coords, block_size=block_size, gid=gid, bnx=bnx, bny=bny, bnz=bnz,
                bnmax=bnmax, center=center, axis=2)

        return minref_blist, bnx, bny, bnz, bnmax
