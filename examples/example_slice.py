'''Basic example to run amr tools'''

from flash_amr_tools.amr_tools import AMRTools
import numpy as np

#################################################
# INPUT YOUR FILENAME HERE
#################################################

filename = "SILCC_hdf5_plt_cnt_0150"

#################################################

###### 1. Initialise the AMRTools object ########

amrtools = AMRTools(filename)

# optional, if one wants to look at a specific region
# xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
# xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)

# amrtools = AMRTools(filename, xmin, xmax)

# optional, if one wants to force refinement levels
# amrtools = AMRTools(filename, xmin, xmax, max_ref_given=10, min_ref_given=3)

##################################################

###### 2. Get the slice (ex. along mid-plane)    ########

dens_sl = amrtools.get_slice("dens", pos=0.0, axis=2)

##################################################

###### 3. Plot the slice       ########

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

extent = [amrtools.low_cor[0], amrtools.up_cor[0], amrtools.low_cor[1], amrtools.up_cor[1]]
norm = mcolors.LogNorm(vmin=dens_sl.min(), vmax=dens_sl.max())

plt.imshow(dens_sl.T, norm=norm, origin="lower", extent=extent)

##################################################