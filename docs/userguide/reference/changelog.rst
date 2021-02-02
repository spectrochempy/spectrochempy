What's new
==========

Version 0.2.6
-------------

**NEW FEATURES**

-  Check for new version on anaconda cloud spectrocat channel.

-  1D NMR processing with the addition of several new methods.

-  Improved handling of Linear coordinates.

**BUGS FIXED**

-  Adding quantity to datasets with different scaling (#199).

-  Math operates now on linear coordinates.

Version 0.2.5
-------------

**TASKS**

-  Docker image building.

-  instructions to use it added in the documentation.

**NEW FEATURES**

-  cantera installation optional.

-  use of pyqt for matplotlib optional.

**BUGS FIXED**

-  added fonts in order to solve missing fonts problems on linux and
   windows.

Version 0.2.4
-------------

**TASKS**

-  Documentation largely revisited and hopefully improved. *Still some
   work to be done*.

-  NDMath (mathematical and dataset creation routines) module revisited.
   *Still some work to be done*.

**NEW FEATURES**

-  Changed CoordRange behavior.

**BUGS FIXED**

-  Fix a problem with importing the API.

-  Fix dim handling in processing functions.

Version 0.2.0
-------------

**NEW FEATURES**

-  Copyright update.

-  Requirements and env yml files updated.

-  Use of the coordinates in math operation improved.

-  Added ROI and Offset properties to NDArrays.

-  Readers / Writers revisited.

-  Bruker TOPSPIN reader.

-  Added LabSpec reader for .txt exported files.

-  Simplified the format of scp file - now zipped JSON files.

-  Rewriting json serialiser.

-  Add function pathclean to the API.

-  Add some array creation function to NDMath.

-  Refactoring plotting preferences system.

-  Baseline correction now accept single value for ranges.

-  Add a waterfall plot.

-  Refactoring plot2D and 1D methods.

-  Added Simpson’rule integration.

-  Addition of multiple coordinates to a dimension works better.

-  Added Linear coordinates (EXPERIMENTAL).

-  Test for NDDataset dtype change at initialisation.

-  Added subdir of txt files in ramandata.

-  Comparison of datasets improved in testing.py.

-  Comparison of datasets and projects.

**BUGS FIXED**

-  Dtype parameter was not taken into account during initialisation of
   NDArrays.

-  Math function behavior for coords.

-  Color normalisation on the full range for colorscale.

-  Configuration settings in the main application.

-  Compatibility read_zip with py3.7.

-  NDpanel temporary removed from the master.

-  2D IRIS fixed.

-  Trapz integration to return NDDataset.

-  Suppressed a forgotten sleep statement that was slowing down the
   SpectroChemPy initialisation.

-  Error in SIMPLISMA (changed affectations such as C.data[…] =
   something by C[…] = something.

-  Cleaning mplstyle about non-style parameters and fix makestyle.

-  Argument of set_xscale.

-  Use read_topspin instead of the deprecated function read_bruker_nmr.

-  Some issues with interactive baseline.

-  Baseline and fitting tutorials.

-  Removed dependency of isotopes.py to pandas.

Version 0.1.x
-------------

-  Initial development versions.
