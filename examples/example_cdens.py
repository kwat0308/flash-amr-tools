'''Basic example to get column density plot with amrtools'''

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
dens = pf["dens"][()][blist]  # ex. density
ref_lvl = pf["refine level"][()][blist]
bbox = pf["bounding box"][()][blist]
bsize = pf["block size"][()][blist]
pf.close()

##################################################

###### 2. Get the column density (ex. along z-axis)    ########

cdens = flash_amr_tools.get_cdens(dens, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

##################################################

###### 3. Plot the slice       ########

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

low_cor, up_cor = bbox[0,:,0], bbox[-1,:,1]

extent = [low_cor[0], up_cor[0], low_cor[1], up_cor[1]]
norm = mcolors.LogNorm(vmin=cdens.min(), vmax=cdens.max())

plt.imshow(cdens.T, norm=norm, origin="lower", extent=extent)

##################################################