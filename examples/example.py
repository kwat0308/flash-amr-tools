import amr_tools
import numpy as np
import h5py

# Input parameters
fname = 'SILCC_hdf5_plt_cnt_0150'
xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)

# get_true_blocks gives a complete list of blocks which are in the box chosen with xmin and xmax (corner coordinates)
# The routine extends the region accordingly to complete the set of blocks such that it has no holes and fits into a cubic shape.
# Additionally we can also choose a cuboid shape. (There might be some bugs regarding this feature)
# Now you can select a lower maximum refinement to reduce the shape, this will save you some memory.
bmin, bmax, blist, bnx, bny, bnz = amr_tools.get_true_blocks(fname, xmin, xmax, cuboid=True, max_ref_given=False)

# The outputs are the minimum and maximum refinement level within the block list (blist) where bnx, bny and bny 
# give the number of blocks in x, y and z direction on the lowest refinement level.

# After loading the file we can select the subsets using the block list (blist)
pf = h5py.File(fname)
dens = pf['dens'][()][blist]
bbox = pf['bounding box'][()][blist]
bs = pf['block size'][()][blist]
ref_lvl = pf['refine level'][()][blist]
vel_vec = [pf['velx'][()][blist], pf['vely'][()][blist], pf['velz'][()][blist]]

print('Data cube has size of %s bytes' % amr_tools.calc_cube_size(bmin, bmax, bnx, bny, bnz, dtype=dens.dtype))

# Lower and upper corner of the region. Can be useful for other methods.
low_cor, up_cor = (bbox[0, :, 0], bbox[-1, :, 1])

# In general we could make calculations with these subsets but in some cases it is useful to project the data onto a uniform grid.
# This can be done using the following routine.
density_grid = amr_tools.get_cube(
	data=dens, bbox=bbox, bs=bs, ref_lvl=ref_lvl, bmin=bmin, bmax=bmax, bnx=bnx, bny=bny, bnz=bnz
)
np.save(arr=density_grid, file='gas_density')
del density_grid

refinement_grid = amr_tools.get_reflvl_cube(
	bbox=bbox, bs=bs, ref_lvl=ref_lvl, bmin=bmin, bmax=bmax, bnx=bnx, bny=bny, bnz=bnz
)
np.save(arr=refinement_grid, file='ref_level')
del refinement_grid

velocity_grid = amr_tools.get_vector_cube(
	data_vec=vel_vec, bbox=bbox, bs=bs, ref_lvl=ref_lvl, bmin=bmin, bmax=bmax, bnx=bnx, bny=bny, bnz=bnz
)
np.save(arr=velocity_grid, file='gas_velocity')
del velocity_grid
