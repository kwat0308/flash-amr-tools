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

1. Specify your filename that you want to look at:
   
```
fname = "SILCC_hdf5_plt_cnt_0150"
```

2. Initialise the `AMRTools` object:

```
from flash_amr_tools.amr_tools import AMRTools

amrtools = AMRTools(fname)
```

The initialisation will calculate the complete list of blocks, the minimum and maximum refinement of the whole domain, and the number of blocks in each dimension.

Optionally, one can specify a particular region of interest to look into. The units should be the same as with those defined in your simulation:

```
# optional, if one wants to look at a specific region
xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)

amrtools = AMRTools(fname, xmin, xmax)
```

One can also optionally force a region to have maximum / minimum refinement by passing the following arguments:

```
amrtools = AMRTools(fname, xmin, xmax, max_ref_given=10, min_ref_given=3)
```

3. Retrive the data (as specified in `flash.par`) as a uniform grid:

```
# ex. density
dens = amrtools.get_cube("dens")
```

Note that the naming convention of the argument must follow the variable name in `flash.par`. This now transforms the density as a cube with gridsizes following the highest resolution.

### Plotting Routines

In principle the above routines are enough to do any plotting with, but we can also go one step further to get slices / on-axis (weighted) projections.

#### Slices

To retrive the slice, we require the data, the position, and the axis(0, 1, or 2) in which the slicing takes place. 

```
# ex. density slice along the mid-plane
dens_sl = amrtools.get_slice(dens, pos=0.0, axis=2)
```

#### On-Axis Projections

To obtain the projection, we require the data and the axis(0, 1, or 2) of the projection.

```
# ex. column density along the z-axis
cdens = amrtools.get_cdens(dens, axis=2)
```

Optionally, one can also specify weights to have weighted projections instead. Note that the shape of the weights must be the same as the data.

```
# ex. temperature-weighted projection along the z-axis
temp = amrtools.get_cube("temp")
cdens_wtemp = amrtools.get_cdens(dens, axis=2, weights=temp)
```

### Optional routines

#### Extracting data with preserved AMR structure

If you want to retrive the data **not** as a uniform cube, this can be done with the following function call:

```
# ex. density
dens = amrtools.get_data("dens")
```

This returns the data that is still preserving the AMR structure, which can be useful for ex. scatter plots.

#### Cubes of refinement levels

One can also retrieve the refinement level as a uniform grid as well:

```
reflvl_cube = amrtools.get_reflvl_cube()
```

which can be used for (for example) plotting the AMR mesh grid.

#### Vector quantities

One can also retrive a uniform grid of vector quantities (ex. velocity, magnetic field) from the following:

```
vel = amrtools.get_vector_cube("vel")
```

This will return a 4D array consisting of a 3-D array in each direction.

## License
This code is under the BSD3 license. See [LICENSE](https://github.com/kwat0308/flash-amr-tools/blob/main/LICENSE) for more details.
