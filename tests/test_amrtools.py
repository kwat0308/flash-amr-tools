'''Testing suite to verify flash-amr-tools'''
import flash_amr_tools
import numpy as np
import h5py

import os 
this_dir = os.path.dirname(os.path.realpath(__file__))

import pytest
import urllib.request

@pytest.fixture
def sedov_cube_data():
    '''Return the cubed Sedov data'''
    f = h5py.File(os.path.join(this_dir, "sedov_cube_data.h5"), "r")
    yield f
    f.close()

def get_sedov_datafile():
    '''Get sedov data from yt hub'''
    fname, _ = urllib.request.urlretrieve("https://hub.yt/api/v1/item/577c13b50d7c6b0001ad63f1/download", filename=os.path.join(this_dir,"sedov_hdf5_chk_0003"))
    return fname


def test_dims(sedov_cube_data):
    '''Check dimensionality of the cube'''

    # get data first
    filename = get_sedov_datafile()

    # get the block lists
    blist, brefs, bns = flash_amr_tools.get_true_blocks(filename)

    # read in the data
    pf = h5py.File(filename)
    dens = pf["dens"][()][blist]
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    pf.close()

    # compute cube
    dens_cube = flash_amr_tools.get_cube(dens, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
    
    # check the dimensionality
    assert np.all(dens_cube.shape == sedov_cube_data["dims"][()]), "Dimensions do not align!"

def test_cube(sedov_cube_data):
    '''Check cube data'''

    # get data first
    filename = get_sedov_datafile()

    # get the block lists
    blist, brefs, bns = flash_amr_tools.get_true_blocks(filename)

    # read in the data
    pf = h5py.File(filename)
    dens = pf["dens"][()][blist]
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    pf.close()

    # compute cube
    dens_cube = flash_amr_tools.get_cube(dens, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

    # check if data is close enough to true data
    assert np.all(np.isclose(dens_cube, sedov_cube_data["cube"][()])), "Cube data does not align!"

def test_slice(sedov_cube_data):
    '''Check slice data'''

    # get data first
    filename = get_sedov_datafile()

    # get the block lists
    blist, brefs, bns = flash_amr_tools.get_true_blocks(filename)

    # read in the data
    pf = h5py.File(filename)
    dens = pf["dens"][()][blist]
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    pf.close()

    # compute slice along z=0
    dens_sl = flash_amr_tools.get_slice(dens, pos=0.5, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

    # check if data is close enough to true data
    assert np.all(np.isclose(dens_sl, sedov_cube_data["dens_sl"][()])), "Slice data does not align!"


def test_cdens(sedov_cube_data):
    '''Check column density data'''

    # get data first
    filename = get_sedov_datafile()

    # get the block lists
    blist, brefs, bns = flash_amr_tools.get_true_blocks(filename)

    # read in the data
    pf = h5py.File(filename)
    dens = pf["dens"][()][blist]
    ref_lvl = pf["refine level"][()][blist]
    bbox = pf["bounding box"][()][blist]
    bsize = pf["block size"][()][blist]
    pf.close()

    # compute column density
    dens_cdens = flash_amr_tools.get_cdens(dens, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)

    # check if data is close enough to true data
    assert np.all(np.isclose(dens_cdens, sedov_cube_data["dens_cdens"][()])), "Column density data does not align!"