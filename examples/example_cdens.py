'''Basic example to run amr tools'''

from flash_amr_tools import AMRToolkit
import numpy as np

#################################################
# INPUT YOUR FILENAME HERE
#################################################

filename = "SILCC_hdf5_plt_cnt_0150"

#################################################

###### 1. Initialise the AMRTools object ########

toolkit = AMRToolkit(filename)

# optional, if one wants to look at a specific region
# xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
# xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)

# toolkit = AMRToolkit(filename, xmin, xmax)

# optional, if one wants to force refinement levels
# toolkit = AMRToolkit(filename, xmin, xmax, max_ref_given=10, min_ref_given=3)

##################################################

###### 2. Get the column density (ex. along z-axis) #

cdens = toolkit.get_cdens("dens", axis=2)

##################################################

###### 3. Plot the column density       ##########

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

extent = [toolkit.low_cor[0], toolkit.up_cor[0], toolkit.low_cor[1], toolkit.up_cor[1]]
norm = mcolors.LogNorm(vmin=cdens.min(), vmax=cdens.max())

plt.imshow(cdens.T, norm=norm, origin="lower", extent=extent)

##################################################