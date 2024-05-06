'''Basic example to run amr tools'''
from flash_amr_tools import AMRToolkit
import numpy as np
import h5py

import os 
this_dir = os.path.dirname(os.path.realpath(__file__))

import pytest
import urllib.request

@pytest.fixture
def sedov_cube_data():
    f = h5py.File(os.path.join(this_dir, "sedov_cube_data.h5"), "r")
    yield f
    f.close()

def get_sedov_datafile():
    # import sedov data from yt hub
    fname, _ = urllib.request.urlretrieve("https://hub.yt/api/v1/item/577c13b50d7c6b0001ad63f1/download", filename=os.path.join(this_dir,"sedov_hdf5_chk_0003"))
    return fname


def test_dims(sedov_cube_data):
    '''Check dimensionality of the cube'''

    # get data first
    filename = get_sedov_datafile()

    # initialise the AMR toolkit
    toolkit = AMRToolkit(filename)

    # compute cube
    dens_cube = toolkit.get_cube("dens")
    
    # check the dimensionality
    assert np.all(dens_cube.shape == sedov_cube_data["dims"][()]), "Dimensions " + dens_cube.shape + " != " + sedov_cube_data["dims"] + "!"

def test_cube(sedov_cube_data):
    '''Check cube data'''

    # get data first
    filename = get_sedov_datafile()

     # initialise the AMR toolkit
    toolkit = AMRToolkit(filename)

    # compute cube
    dens_cube = toolkit.get_cube("dens")

    # check if data is close enough to true data
    assert np.all(np.isclose(dens_cube, sedov_cube_data["cube"][()])), "Cube data does not align!"

def test_slice(sedov_cube_data):
    '''Check slice data'''

    # get data first
    filename = get_sedov_datafile()

     # initialise the AMR toolkit
    toolkit = AMRToolkit(filename)

    # compute slice
    dens_sl = toolkit.get_slice("dens", pos=0.5, axis=2)

    # check if data is close enough to true data
    assert np.all(np.isclose(dens_sl, sedov_cube_data["sl"][()])), "Slice data does not align!"


def test_cdens(sedov_cube_data):
    '''Check column density data'''

    # get data first
    filename = get_sedov_datafile()

     # initialise the AMR toolkit
    toolkit = AMRToolkit(filename)

    # compute column density
    dens_cdens = toolkit.get_cdens("dens", axis=2)

    # check if data is close enough to true data
    assert np.all(np.isclose(dens_cdens, sedov_cube_data["cdens"][()])), "Column density data does not align!"