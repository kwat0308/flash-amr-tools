'''Basic example to get vector data as uniform cube'''

import flash_amr_tools
import numpy as np
import h5py

#################################################
# INPUT YOUR FILENAME & REGION OF INTEREST HERE
#################################################

filename = "SILCC_hdf5_plt_cnt_0150"

# Optional, defaults to the whole domain
xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)

#################################################

###### 1. Get the block list of region ########

blist, brefs, bns = flash_amr_tools.get_true_blocks(filename, xmin, xmax)

##################################################

###### 2. Get data within region of interest    ########

# read in the data
pf = h5py.File(filename)
vel_vec = [pf["velx"][()][blist], pf["vely"][()][blist], pf["velz"][()][blist]] # ex. velocity
ref_lvl = pf["refine level"][()][blist]
bbox = pf["bounding box"][()][blist]
bsize = pf["block size"][()][blist]
pf.close()

##################################################

###### 3. Transform data into uniform cube   ####
vel_cube = flash_amr_tools.get_vector_cube(vel_vec, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

###### 3. (Optional) Save your data       ########

import h5py
with h5py.File("./output.h5", "w") as f:
    f.create_dataset("vel_cube", data=vel_cube)

##################################################