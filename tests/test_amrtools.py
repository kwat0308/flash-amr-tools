'''Basic example to run amr tools'''
from flash_amr_tools.amr_tools import AMRTools
import numpy as np
import h5py

import pytest
import urllib.request

def get_sedov_datafile():
    # import sedov data from yt hub
    fname, _ = urllib.request.urlretrieve("https://hub.yt/api/v1/item/577c13b50d7c6b0001ad63f1/download", filename="sedov_hdf5_chk_0003")
    return fname


@pytest.fixture
def sedov_cube_data(elem : str):
    with h5py.File("./sedov_cube_data.h5", "r"):
        return f[elem][()]

def test_dims(sedov_cube_data):
    '''Check dimensionality of the cube'''

    # get data first
    filename = get_sedov_datafile()

    # initialise amrtools
    amrtools = AMRTools(filename)

    # compute cube
    dens_cube = amrtools.get_cube("dens")
    
    # check the dimensionality
    assert dens_cube.shape == sedov_cube_data("shape"), "Dimensions " + dens_cube.shape + " != " + sedov_cube_data["shape"] + "!"

def test_cube(sedov_cube_data):
    '''Check cube data'''

    # get data first
    filename = get_sedov_datafile()

    # initialise amrtools
    amrtools = AMRTools(filename)

    # compute cube
    dens_cube = amrtools.get_cube("dens")

    # check if data is close enough to true data
    assert np.isclose(dens_cube, sedov_cube_data("cube")), "Cube data does not align!"

def test_slice(sedov_cube_data):
    '''Check slice data'''

    # get data first
    filename = get_sedov_datafile()

    # initialise amrtools
    amrtools = AMRTools(filename)

    # compute slice
    dens_sl = amrtools.get_slice("dens", pos=0.5, axis=2)

    # check if data is close enough to true data
    assert np.isclose(dens_sl, sedov_cube_data("sl")), "Slice data does not align!"


def test_cdens(sedov_cube_data):
    '''Check column density data'''

    # get data first
    filename = get_sedov_datafile()

    # initialise amrtools
    amrtools = AMRTools(filename)

    # compute column density
    dens_cdens = amrtools.get_cdens("dens", axis=2)

    # check if data is close enough to true data
    assert np.isclose(dens_cdens, sedov_cube_data("cdens")), "Column density data does not align!"
    
    


    
        




    
    



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

###### 2. Get the data as uniform cube    ########

dens = amrtools.get_cube("dens")

##################################################

###### 3. (Optional) Save your data       ########

import h5py
with h5py.File("./output.h5", "w") as f:
    f.create_dataset("dens", data=dens)

##################################################