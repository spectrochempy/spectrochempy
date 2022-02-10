# What's new

## VERSION 0.4

### version 0.4.0rc1

#### NEW FEATURES

##### NetCDF import/export
Features related to the export and import format (see [issue #97- comment](
https://github.com/spectrochempy/spectrochempy/issues/97#issuecomment-639590022))

* `write_netcdf`: Write to file in [netCDF](http://www.unidata.ucar.edu/software/netcdf/) format.
* This type of file is a binary file format for self-described datasets
  that originated in the geosciences. It is not much in chemistry (as far we know) but it is used by xarray as a
  recommended way of saving data.
  Also saving spectrochempy file in this format will allow an easy interchange between the two software.
  (Saving on this format is only possible if [xarray](https://xarray.pydata.org/en/stable/) package is available).

* `NDDataset.to_xarray`: A new dataset method creating a `xarray.DataArray` object suitable for use with `xarray`.

* `read_netcdf` : open netCDF files saved by spectrochempy or xarray (not tested with other sources).

* `NDDataset.from_xarray`: Create a new dataset from a `xarray.DataArray` object.

* refactoring related to the export to netCDF.

    Note the renaming of three attributes
    have been made to improve conversion from and to xarray.

    - title --> long_name
    - origin --> source
    - description --> comment

    Backwards compatibility is preserved in reading

##### Timezone info

* A new attribute allows the user to change the timezone of the dataset. This affect the way
  attributes such are `created`are displayed. Internally stored in UTC format, they are display according to the timezone info.

##### Datetime coordinates

* Coordinates can now be created with the numpy dtype  'datetime64'. Internally all datetimes will be stored in UTC.
  When reading a date or a datetime coordinates the return will be converted to the local timezone, except if the
  timezone property of the dataset is set differently.

#### Other changes

* History
  Its behavior have been improved.
  Dates for entries are set automatically and are timezone-aware.
  See the docs for more information:
[About-the-history-attribute](https://www.spectrochempy.fr/latest/userguide/dataset/dataset.html#About-the-history-attribute)

* Datetime64 coordinates of NDDataset.

    np.datetime64 numpy array can be used a coordinate transparently.  Regaring the math operation on such axis, only addtion or substraction are allowed.

* reading of metadata simplified:

```python
nd2 = IR_dataset_2D

# add some attribute
nd2.meta.pression = 34
nd2.meta.temperature = 3000
assert nd2.meta.temperature == 3000
assert nd2.temperature == 3000 # alternative way to get the meta attribute

# also for the coordinates
nd2.y.meta.pression = 3
assert nd2.y.meta["pression"] == 3
assert nd2.y.pression == 3  # alternative way to get the meta attribute
```
* Change the size of rotating log to 256K instead of 32K.

* Revision of the math operations :
  This function is at the heart of the mathematical calculations on the
  data sets and coordinates. However the complexity of its structure made
  it very difficult to maintain and debug. Thus, an
  important rewrite of this function has been performed in order to make
  it more understandable and therefore easier to maintain.
* Integration method are now located with the analysis methods
* Datetime axis now taken into account in plot methods.
* Datetime best units estimated automatically
*
#### BUGS FIXED

* Coordinate _sort method
* Add m and magnitude (properties)
  They are Alias of data but the previous definition was not working.
* Revision of some testing methods : comparison of multicoordinates now works.
* NDDataset/NDArray.squeeze() and  add test for this method.
  It was failing when *dim argument were passed.
  In addition Ia new argument keepdims has been added to define dimension
  to keep even if they are of size 1.
* Update iris.py to avoid muticoordinate's names change.
  Multicoordinates work bad for name indexing if their name is not _1, _2 ...
* Missing case in check_filename_to_save.
* Bug for math operation on LinearCoordinates: unit loss
  The integrate method so it takes into account Datetime64 axis.
* math _op method for LinearCoord.
* inplace binary methods to work with datetime64 coordinates.
* Copy method of Coord now copy the size attribute.
* display problem in PCA examples.
* Docs display problems for twinx (both axes are now unit-aware)
---
## VERSION 0.3

### version 0.3.3

#### NEW FEATURES

* add `download_nist_ir()` to download IR spectra from NIST/webbook

* `read_spa()` now allows extracting sample and background interferograms

#### BUGS FIXED

* fix gettingstarted/overview.py after IRIS refactoring

### version 0.3.2

#### NEW FEATURES

* add `download_nist_ir()` to download IR spectra from NIST/webbook

* `read_spa()` now allows extracting sample and background interferograms

* Add a log file (rotating)

#### BUGS FIXED

* fix gettingstarted/overview.py after IRIS refactoring

* TQDM progress bar

* Fix #360 : write a dataset with a specified filename do not open a dialog except if the file already exists and if
  confirm=True is set.

* `read` and `read_dir` now ignore non readable files.

* `read_labspec` now ignore non-labspec .txt files.

* Fix #296 : IRIS and quadprog version.

* Fix #375 : plotting issues.

### Version 0.3.1

#### NEW FEATURES

* Added a `show_versions` method in the API.

* Improved bug reports and pull request templates. Requests for
  help are now made in github discussions.

* Docs API reference has been hopefully improved.

### Version 0.3.0

#### NEW FEATURES

* Package refactoring which may break previous  behaviour. This is why we
  updated the minor version number from 0.2 to 0.3.

* GRAMS/Thermo Galactic .spc file reader.

* Fitting models updated and tested.

#### BUGS FIXED

* Bug in check_updates preventing working without connection.

---
## VERSION 0.2

### Version 0.2.23

#### BUGS FIXED

* Workflow/Codeclimate issues

### Version 0.2.22

#### BUGS FIXED

* QT save_dialog.

* Plot_multiple bug.

### Version 0.2.21

#### NEW FEATURES

* Indexing or slicing a NDArray with quantities is now possible.

* MatPlotLib Axes are subclassed in order to accept quantities for method arguments.

#### BUGS FIXED

* NDArray constructor now accept a homogeneous list of quantities as data input. Units are set accordingly.

* Qt Dialogs. This is related to issue #198, as tk dialogs can be replaced by Qt when working with a terminal.

* Custom exception.

* Qt Dialogs. This is related to issue #198, as tk dialogs can be replaced by Qt when working with a terminal.

* Doc display problems.

### Version 0.2.18

#### NEW FEATURES

* pip installation now possible

* Some code revision

* NNMF revision

* Documentation improvement

#### BUGS FIXED

* Issue #310

* The order of multicoordinates for a single dimension

* Integrate methods to avoid code-climate warnings (duplicate code)

* Documentation for the integrate methods

* skipping test_sh under windows


### Version 0.2.17

#### NEW FEATURES

* OPUS file reader: add filenames as labels.

* OMNIC file reader: Documented more .spa header keys.

#### BUGS FIXED

* Compatibility with matplotlib 3.5 (issue #316).

* Datasets were not properly centered in PCA analysis.

* Comparing dataset with only labels coordinates was failing.

* Issue #322: mean and other API reduce methods were sometimes failing.

### Version 0.2.16

#### NEW FEATURES

* IRIS: Added 1D datasets.

* IRIS: Added kernel function for diffusion .

* EFA: Added indication of progress.

* Cantera: Added differential evolution algorithm in cantera utilities.

* Cantera: Added PFR object in cantera utilities.

* DOC: Added list of papers citing spectrochempy.

* Github action workflows to test, build and publish conda package and docs in replacement of Travis CI.

* Use CodeClimate to show Coverage info

#### BUGS FIXED

* IRIS example after modification of readers.

* IRIS: automatic search of the L-curve corner.

* MCR-ALS returns the 'soft' concentration matrix.

* Document building configuration after update of external packages.

* DOC: several broken links.

* Baseline correction default changed.

* Compatibility with newest change in Colab

### Version 0.2.15

#### NEW FEATURES

* Added a baseline correction method: `basc`.

* Baseline ranges can be stored in meta.regions['baseline'] - basc will recognize them.

#### BUGS FIXED

* Comparison of dataset when containing metadata in testing functions.

* Project.

* Bug in the `to` function.

### Version 0.2.14

#### NEW FEATURES

* A default coordinate can now be selected for multiple coordinates dimensions.

#### BUGS FIXED

* Alignment along several dimensions (issue #248)

* to() and ito() methods to work correctly (issue #255)

* Baseline correction works on all dimensions

### Version 0.2.13

#### BUGS FIXED

* Solved the problem that reading of experimental datasets was too slow in v.0.2.12.

### Version 0.2.12

#### BUGS FIXED

* LinearCoord operations now working.

* Baseline default now "sequential" as expected.

**WARNING**: It was wrongly set to "mutivariate" in previous releases, so you should  expect some difference with processing you may have done before.

* Comparison of coordinates now correct for mathematical operations.

* Alignment methods now working (except for multidimensional alignment).

### Version 0.2.11

#### BUGS FIXED

* Plot2D now works when more than one coord in 'y' axis (#238).

* Spectrochempy_data location has been corrected (#239).

### Version 0.2.10

#### NEW FEATURES

* All data for tests and examples are now external.

They are now located in a separate conda package: `spectrochempy_data`.

* Installation in Colab with Examples is now supported.

#### BUGS FIXED

* Read_quadera() and examples now based on a correct asc file

### Version 0.2.9

#### BUGS FIXED

* Hotfix regarding display of NMR x scale

### Version 0.2.8

#### NEW FEATURES

* Added write_csv() dir 1D datasets

* Added read_quadera() for Pfeiffer Vacuum's QUADERA® MS files

* Added test for trapz(), simps(), readquadera()

* Improved displaying of Interferograms

#### BUGS FIXED

* Problem with trapz(), simps()

* interferogram x scaling

### Version 0.2.7

#### NEW FEATURES

* Test and data for read_carroucell(), read_srs(), read_dso()

* Added NMR processing of 2D spectra.

* Added FTIR interferogram processing.

#### BUGS FIXED

* Problem with read_carroucell(), read_srs(), read_dso()

* Colaboratory compatibility

* Improved check updates

### Version 0.2.6

#### NEW FEATURES

* Check for new version on anaconda cloud spectrocat channel.

* 1D NMR processing with the addition of several new methods.

* Improved handling of Linear coordinates.

#### BUGS FIXED

* Adding quantity to datasets with different scaling (#199).

* Math now operates on linear coordinates.

* Compatibility with python 3.6

### Version 0.2.5

#### NEW FEATURES

* Docker image building.

* Instructions to use it added in the documentation.

* Cantera installation optional.

* Use of pyqt for matplotlib optional.

#### BUGS FIXED

* Added fonts in order to solve missing fonts problems on Linux and windows.

### Version 0.2.4

#### NEW FEATURES

* Documentation largely revisited and hopefully improved. *Still some work to be done*.

* NDMath (mathematical and dataset creation routines) module revisited. *Still some work to be done*.

* Changed CoordRange behavior.

#### BUGS FIXED

* Problem with importing the API.

* Dim handling in processing functions.

### Version 0.2.0

#### NEW FEATURES

* Copyright update.

* Requirements and env yml files updated.

* Use of the coordinates in math operation improved.

* Added ROI and Offset properties to NDArrays.

* Readers / Writers revisited.

* Bruker TOPSPIN reader.

* Added LabSpec reader for .txt exported files.

* Simplified the format of scp file - now zipped JSON files.

* Rewriting json serialiser.

* Add function pathclean to the API.

* Add some array creation function to NDMath.

* Refactoring plotting preference system.

* Baseline correction now accepts single value for ranges.

* Add a waterfall plot.

* Refactoring plot2D and 1D methods.

* Added Simpson'rule integration.

* Addition of multiple coordinates to a dimension works better.

* Added Linear coordinates (EXPERIMENTAL).

* Test for NDDataset dtype change at initialization.

* Added subdir of txt files in ramandata.

* Comparison of datasets improved in testing.py.

* Comparison of datasets and projects.

#### BUGS FIXED

* Dtype parameter was not taken into account during initialization of NDArrays.

* Math function behavior for coords.

* Color normalization on the full range for colorscale.

* Configuration settings in the main application.

* Compatibility read_zip with py3.7.

* NDpanel temporary removed from the master.

* 2D IRIS.

* Trapz integration to return NDDataset.

* Suppressed a forgotten sleep statement that was slowing down the SpectroChemPy initialization.

* Error in SIMPLISMA (changed affectations such as C.data[...] = something by C[...] = something.

* Cleaning mplstyle about non-style parameters and corrected makestyle.

* Argument of set_xscale.

* Use read_topspin instead of the deprecated function read_bruker_nmr.

* Some issues with interactive baseline.

* Baseline and fitting tutorials.

* Removed dependency of isotopes.py to pandas.

---
## VERSION 0.1

### Version 0.1.x

* Initial development versions.
