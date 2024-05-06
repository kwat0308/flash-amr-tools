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
filename = "SILCC_hdf5_plt_cnt_0150"
```

2. Initialise the `AMRToolkit` object:

```
from flash_amr_tools import AMRToolkit

toolkit = AMRToolkit(filename)
```

The initialisation will calculate the complete list of blocks, the minimum and maximum refinement of the whole domain, and the number of blocks in each dimension.

3. Retrive the data (as specified in `flash.par`) as a uniform grid:

```
# ex. density
dens = toolkit.get_cube("dens")
```

Note that the naming convention of the argument must follow the variable name in `flash.par`. This now transforms the density as a cube with gridsizes following the highest resolution.

### Plotting Routines

Optional plotting routines for slice & on-axis (weighted) projections are also available.

#### Slices

To retrive the slice, we require the field name (as specified in `flash.par`), the position, and the axis(0, 1, or 2) in which the slicing takes place. 

```
# ex. density slice along the mid-plane
dens_sl = toolkit.get_slice("dens", pos=0.0, axis=2)
```

#### On-Axis Projections

To obtain the projection, we require the field name (as specified in `flash.par`) and the axis(0, 1, or 2) of the projection.

```
# ex. column density along the z-axis
cdens = toolkit.get_cdens("dens", axis=2)
```

Optionally, one can also specify weights to have weighted projections instead. Note that the field name for the weights must also be specified as in `flash.par`

```
# ex. temperature-weighted projection along the z-axis
cdens_wtemp = toolkit.get_cdens("dens", axis=2, weights_field = "temp")
```

### Optional routines

#### Further initialisation routines

Optionally, one can specify a particular region of interest to look into. The units should be the same as with those defined in your simulation:

```
# optional, if one wants to look at a specific region
xmin = np.array([2.8931249e+20, -5.78625013e+20, -1.9287499e+20], dtype=np.float32)
xmax = np.array([6.7506249e+20, -1.92874993e+20,  1.9287499e+20], dtype=np.float32)

toolkit = AMRToolkit(filename, xmin, xmax)
```

One can also optionally force a region to have maximum / minimum refinement by passing the following arguments:

```
toolkit = AMRToolkit(filename, xmin, xmax, max_ref_given=10, min_ref_given=3)
```
which may be useful to conserve memory.

#### Extracting data with preserved AMR structure

If you want to retrive the data **not** as a uniform cube, this can be done with the following function call:

```
# ex. density
dens = toolkit.get_data("dens")
```

This returns the data that is still preserving the AMR structure, which can be useful for ex. scatter plots.

#### Cubes of refinement levels

One can also retrieve the refinement level as a uniform grid as well:

```
reflvl_cube = toolkit.get_reflvl_cube()
```

which can be used for (for example) plotting the AMR mesh grid.

#### Vector quantities

One can also retrive a uniform grid of vector quantities (ex. velocity, magnetic field) from the following:

```
vel = toolkit.get_vector_cube("vel")
```

This will return a 4D array consisting of a 3-D array in each direction.

## License
This code is under the BSD3 license. See [LICENSE](https://github.com/kwat0308/flash-amr-tools/blob/main/LICENSE) for more details.
