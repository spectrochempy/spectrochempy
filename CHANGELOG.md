# What's new

## VERSION 0.4

### version 0.4.2

#### NEW FEATURES
* Concatenate and stack:
  - `force_stack` keyword in concatenate() now deprecated.
  - `stack()` method now generates a new dim, even if a dim of size one in present

#### BUG FIXED:
* issue #417
* Transmittance and absorbance units now correctly handled.

### version 0.4.1

#### BREAKING CHANGES
* Requires pint >= 0.18

#### NEW FEATURES
* Compatibility with Python 3.10
  - Spectrochempy is tested with 3.10, 3.9 version of python, and on Windows and linux platform. Older versions > 3.6 of python or different platforms may still work, but with no guaranty.

#### BUG FIXED
* Transmittance and absorbance units now correctly handled.
* Save dialog selection.

## VERSION 0.3

#### NEW FEATURES

* Remove the dependency to `nmrglue`.
* Improve `pip` installation (see Issue #402)
* Make `widget` as the default backend for matplotlib plots.
* Add `BaselineCorrector()` widget.
* Add `download_nist_ir()` to download IR spectra from NIST/webbook.
* Allow extracting background interferogram or spectrum in `read_srs()`.
* Allow extracting sample and background interferograms in `read_spa()`.

#### BUGS FIXED

* Fix bug in `read_srs`.
* Fix gettingstarted/overview.py after IRIS refactoring.

### version 0.3.2

#### NEW FEATURES

* Add a log file (rotating)

#### BUGS FIXED

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
* Custom exceptions.
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

  **WARNING**: It was wrongly set to "mutivariate" in previous releases, so you should
  expect some difference with processing you may have done before.
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
