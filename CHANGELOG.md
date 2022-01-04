# What's new

## Version 0.2.18rc2

**NEW FEATURES**


**NEW TASKS**

* pip installation now possible
 
* Some code revision

* NNMF revision
  
* Documentation improvement

**BUGS FIXED**

* FIX issue #310

* FIX: the order of multicoordinates for a single dimension

* FIX: integrate methods to avoid code-climate warnings (duplicate code)

* FIX: Documentation for the integrate methods

* FIX: skipping test_sh under windows


## Version 0.2.17

**NEW FEATURES**

* OPUS file reader: add filenames as labels.

* OMNIC file reader: Documented more .spa header keys.

**BUGS FIXED**

* Compatibility with matplotlib 3.5 (issue #316).

* Datasets were not properly centered in PCA analysis.

* Comparing dataset with only labels coordinates was failing.

* Issue #322 fixed : mean and other API reduce methods were sometimes failing.

## Version 0.2.16

**NEW FEATURES**

* IRIS: Added 1D datasets.

* IRIS: Added kernel function for diffusion .

* EFA: Added indication of progress.

* Cantera: Added differential evolution algorithm in cantera utilities.

* Cantera: Added PFR object in cantera utilities.

* DOC: Added list of papers citing spectrochempy.

* Github action workflows to test, build and publish conda package and docs in replacement of Travis CI.

* Use CodeClimate to show Coverage info

**BUGS FIXED**

* IRIS example after modification of readers.

* IRIS: automatic search of the L-curve corner.

* MCR-ALS returns the 'soft' concentration matrix.

* Document building configuration after update of external packages.

* DOC: fix several broken links.

* Baseline correction default changed.

* Compatibility with newest change in Colab

## Version 0.2.15

**NEW FEATURES**

* Added a baseline correction method: `basc`.

* Baseline ranges can be stored in meta.regions['baseline'] - basc will recognize them.

**BUGS FIXED**

* Comparison of dataset when containing metadata in testing functions.

* Some fixes relative to Project.

* Bug in the `to` function.

## Version 0.2.14

**NEW FEATURES**

* A default coordinate can now be selected for multiple coordinates dimensions.

**BUGS FIXED**

* Alignment along several dimensions (issue #248)

* to() and ito() methods have been fixed to work correctly (issue #255)

* Baseline correction works on all dimensions

## Version 0.2.13

**BUGS FIXED**

* Solved the problem that reading of experimental datasets was too slow in v.0.2.12.

## Version 0.2.12

**BUGS FIXED**

* LinearCoord operations now working.

* Baseline default now "sequential" as expected.

**WARNING**: It was wrongly set to "mutivariate" in previous releases, so you should  expect some difference with processing you may have done before.

* Comparison of coordinates now correct for mathematical operations.

* Alignment methods now working (except for multidimensional alignment).

## Version 0.2.11

**BUGS FIXED**

* Plot2D now works when more than one coord in 'y' axis (#238).

* Spectrochempy_data location has been corrected (#239).

## Version 0.2.10

**NEW FEATURES**

* All data for tests and examples are now external.

They are now located in a separate conda package: `spectrochempy_data`.

* Installation in Colab with Examples is now supported.

**BUGS FIXED**

* Read_quadera() and examples now based on a correct asc file

## Version 0.2.9

**BUGS FIXED**

* Hotfix regarding display of NMR x scale

## Version 0.2.8

**NEW FEATURES**

* Added write_csv() dir 1D datasets

* Added read_quadera() for Pfeiffer Vacuum's QUADERA® MS files

* Added test for trapz(), simps(), readquadera()

* Improved displaying of Interferograms

**BUGS FIXED**

* Problem with trapz(), simps()

* FIX: interferogram x scaling

## Version 0.2.7

**NEW FEATURES**

* Test and data for read_carroucell(), read_srs(), read_dso()

* Added NMR processing of 2D spectra.

* Added FTIR interferogram processing.

**BUGS FIXED**

* Problem with read_carroucell(), read_srs(), read_dso()

* Colaboratory compatibility

* Improved check updates

## Version 0.2.6

**NEW FEATURES**

* Check for new version on anaconda cloud spectrocat channel.

* 1D NMR processing with the addition of several new methods.

* Improved handling of Linear coordinates.

**BUGS FIXED**

* Adding quantity to datasets with different scaling (#199).

* Math now operates on linear coordinates.

* Compatibility with python 3.6

## Version 0.2.5

**TASKS**

* Docker image building.

* Instructions to use it added in the documentation.

**NEW FEATURES**

* Cantera installation optional.

* Use of pyqt for matplotlib optional.

**BUGS FIXED**

* Added fonts in order to solve missing fonts problems on Linux and windows.

## Version 0.2.4

**TASKS**

* Documentation largely revisited and hopefully improved. *Still some work to be done*.

* NDMath (mathematical and dataset creation routines) module revisited. *Still some work to be done*.

**NEW FEATURES**

* Changed CoordRange behavior.

**BUGS FIXED**

* Fix a problem with importing the API.

* Fix dim handling in processing functions.

## Version 0.2.0

**NEW FEATURES**

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

**BUGS FIXED**

* Dtype parameter was not taken into account during initialization of NDArrays.

* Math function behavior for coords.

* Color normalization on the full range for colorscale.

* Configuration settings in the main application.

* Compatibility read_zip with py3.7.

* NDpanel temporary removed from the master.

* 2D IRIS fixed.

* Trapz integration to return NDDataset.

* Suppressed a forgotten sleep statement that was slowing down the SpectroChemPy initialization.

* Error in SIMPLISMA (changed affectations such as C.data[...] = something by C[...] = something.

* Cleaning mplstyle about non-style parameters and fix makestyle.

* Argument of set_xscale.

* Use read_topspin instead of the deprecated function read_bruker_nmr.

* Some issues with interactive baseline.

* Baseline and fitting tutorials.

* Removed dependency of isotopes.py to pandas.

## Version 0.1.x

* Initial development versions.
