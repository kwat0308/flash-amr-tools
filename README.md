# flash-amr-tools
Small tool to create uniform data cubes of FLASH datasets. Port from [bitbucket.org/pierrenbg/flash-amr-tools](https://bitbucket.org/pierrenbg/flash-amr-tools/src/master/)

## Dependencies

- `h5py`
- `numpy`

## Installation

This can be done as simply as 

```
pip install flash-amr-tools
```

## Usage

1. Specify your filename, and optionally the region of interest, that you want to look at:
   
```
filename = "SILCC_hdf5_plt_cnt_0150"
xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)
```

If no xmin, xmax are provided, then it defaults to using the whole domain. 

2. Get the block list corresponding to the region of interest.

```
import flash_amr_tools

blist, brefs, bns = flash_amr_tools.get_true_blocks(filename, xmin, xmax)
```

This will calculate the complete list of blocks, the maximum and minimum refinement, and the number of blocks at the lowest refinement level within the region of interest.

3. Read in the data using `h5py`

```
import h5py

# read in the data
pf = h5py.File(filename)
dens = pf["dens"][()][blist]  # ex. density
ref_lvl = pf["refine level"][()][blist]
bbox = pf["bounding box"][()][blist]
bsize = pf["block size"][()][blist]
pf.close()
```

Note that the naming convention of the argument must follow the variable name in `flash.par`. Note that the refinement levels, the bounding box, and the block size are necessary to determine the coordinates and the refinement levels in each block.

4. Convert the data into a uniform cube

```
dens_cube = flash_amr_tools.get_cube(dens, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
```

This now transforms the density as a cube with gridsizes following the highest resolution.

### Plotting Routines

Optional plotting routines for slice & on-axis (weighted) projections are also available.

#### Slices

To retrive the slice, we require the dataset, the position, and the axis(0, 1, or 2) in which the slicing takes place. 

```
# ex. density slice along the mid-plane
dens_sl = flash_amr_tools.get_slice(dens, pos=0.5, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
```

#### On-Axis Projections

To obtain the projection, we require the dataset and the axis(0, 1, or 2) of the projection.

```
# ex. column density along the z-axis
cdens = flash_amr_tools.get_cdens(dens, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
```

Optionally, one can also specify weights to have weighted projections instead. Note that the shape of the weights must be the same as that of the dataset.

```
# ex. temperature-weighted projection along the z-axis

# first read in other data from h5py
# read in the data
pf = h5py.File(filename)
temp = pf["temp"][()][blist]  # temperature
pf.close()

# now get temperature-weighted projection along z-axis
cdens_wtemp = flash_amr_tools.get_cdens(dens, axis=2, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns, weights=temp)
```

### Optional routines

#### Further initialisation routines

One can also optionally force a region to have maximum / minimum refinement by passing the following arguments:

```
blist, brefs, bns = flash_amr_tools.get_true_blocks(filename, xmin, xmax, max_ref_given=10, min_ref_given=3)
```
which may be useful to conserve memory.

#### Cubes of refinement levels

One can also retrieve the refinement level as a uniform grid as well:

```
reflvl_cube = flash_amr_tools.get_reflvl_cube(ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
```

which can be used for (for example) plotting the AMR mesh grid.

#### Vector quantities

One can also retrive a uniform grid of vector quantities (ex. velocity, magnetic field) from the following:

```
# read in and save vectorial data as list
pf = h5py.File(filename)
vel_vec = [pf["velx"][()][blist], pf["vely"][()][blist], pf["velz"][()][blist]]
ref_lvl = pf["refine level"][()][blist]
bbox = pf["bounding box"][()][blist]
bsize = pf["block size"][()][blist]
pf.close()

# return uniform cube of vectorial data in each direction
vel_cube = flash_amr_tools.get_vector_cube(vel_vec, ref_lvl=ref_lvl, bbox=bbox, bsize=bsize, brefs=brefs, bns=bns)
```

This will return a 4D array consisting of a 3-D array in each direction (as the last axis).

## License
This code is under the BSD3 license. See [LICENSE](https://github.com/kwat0308/flash-amr-tools/blob/main/LICENSE) for more details.
