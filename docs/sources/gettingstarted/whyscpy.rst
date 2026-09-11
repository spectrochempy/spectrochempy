.. _whyscpy:

**********************
Why SpectroChemPy?
**********************

**Table of Contents**

.. contents:: Contents
   :local:
   :depth: 2

`SpectroChemPy` is an open-source Python framework for processing, analyzing,
and modeling spectroscopic data in chemistry. It is still mainly developed at the
`LCS <https://www.lcs.ensicaen.fr/>`__ (Laboratoire Catalyse et Spectrochimie)
in Caen, but is now used and potentially extended by a wider community.
This page explains
the problems SpectroChemPy addresses, what it provides, and the cases in which
another tool may be more appropriate.

The spectroscopy workflow problem
=================================

A spectrum is more than a list of numbers. The numerical intensities only make
sense together with the meaning of each axis (wavenumber, wavelength, frequency,
chemical shift, time, temperature, ...), the physical units, the identity of the
sample, and the acquisition conditions under which the measurement was made.

A realistic analysis is rarely a single algorithm. Instrumental files must be
read and organized, axes identified, units assigned, and metadata retained.
Data are then selected, smoothed, baseline-corrected, filtered, normalized, or
repeatedly transformed before visualization and univariate or multivariate
analysis. For a time-resolved or temperature-resolved series, operations must
preserve or explicitly transform the correspondence between spectra and the
experimental variables associated with them.

Commercial and graphical software can provide efficient environments for
routine and exploratory analysis. Depending on the software, however, the exact
algorithms, parameters, intermediate states, or processing history may not be
fully accessible or readily exportable. This can make detailed inspection and
reproduction of an analysis more difficult.

What spreadsheets and plain arrays do not provide
==================================================

Tabular tools are well suited to observations and variables that fit naturally
into rows and columns. Multidimensional spectroscopic datasets, however, may
require several experimental coordinates per dimension, physical units, masks,
acquisition metadata, and domain-specific conventions that are not represented
by a conventional table alone.

Plain ``numpy.ndarray`` objects are the foundation of scientific computing with
Python, but by themselves they carry no axes, units, masks, or labels. The
meaning of the data must be re-established by convention at every step: the
axis order, the unit conversion, the meaning of each column. These conventions
remain implicit, which makes a workflow harder to inspect, compare, reuse, or
turn into a reproducible record.

What NDDataset provides
=======================

The central object of SpectroChemPy, :class:`~spectrochempy.NDDataset`,
associates a multidimensional numerical array with everything needed to
interpret it:

- **named dimensions** with explicit `dimension` names (for example ``x`` for
  the spectral axis and ``y`` for the observation axis) that provide stable
  handles for operations;
- **coordinates** (`Coord` objects, grouped in a `CoordSet`) that carry the
  physical positions along each dimension, together with their titles and
  units;
- **coordinate-aware selection**: a region can be chosen by integer index, by a
  physical interval of a coordinate, or by a label;
- **units** on the data values and on each coordinate, with conversion of
  compatible units and rejection of incompatible ones;
- **labels** attaching categorical identifiers (for instance a sample name) to
  individual points;
- **masks** marking values that should be treated as invalid or absent, for
  example a saturated detector region;
- **metadata** describing the experimental and acquisition context when the
  imported format provides it;
- **history** recording timestamped entries for the operations that
  participate in the history contract since the dataset creation.

Because the scientific context is part of the object, operations implemented
for :class:`~spectrochempy.NDDataset` are designed to preserve or update the
relevant context according to their documented contracts. Slicing,
transposition, unit conversion, and arithmetic carry the relevant coordinates,
units, and masks along. This behavior is operation-dependent and should be
checked for the version and workflow being used, and operations that
participate in the history contract add an inspectable entry describing what
was done. The :ref:`user_guide` details
this data model and its boundaries.

From import to models: one environment
======================================

SpectroChemPy provides a coherent set of interfaces built on this data model.

- **Importing**: readers provided by the core package and by optional plugins
  translate supported instrumental and exchange formats into the common
  representation, including core readers such as OMNIC, OPUS, LabSpec, WDF,
  SPC, JCAMP-DX, MAT, DDR, and directory scans, as well as plugin readers such
  as those for NMR and PerkinElmer instruments. The coordinates and metadata
  that can be determined from each file are kept, and datasets can also be
  exported to CSV, JCAMP-DX, MATLAB, or converted to xarray datasets and
  NetCDF.
- **Processing**: unit-aware arithmetic, slicing and coordinate-based
  selection, baseline correction, automatic subtraction, smoothing and
  filtering, derivative calculation, normalization and scaling, denoising,
  FFT and Fourier-related operations, and masking.
- **Analysis**: SVD, PCA, EFA, MCR-ALS, PLS, NMF, ICA, SIMPLISMA, peak finding
  and integration, and curve fitting with documented lineshape models.
- **Visualization**: a plotting interface adapted to 1D, 2D, and 3D
  spectroscopic data, with multiple styles and reusable preferences.
- **Workflows**: many SpectroChemPy analysis estimators follow a
  scikit-learn-compatible interface, and
  :class:`~spectrochempy.analysis.pipeline.Pipeline` provides a linear
  composition of an approved set of preprocessing steps and supported terminal
  estimators. This makes the sequence and parameters explicit, but does not by
  itself capture the input data, software environment, or every requirement for
  reproducibility.

Many examples covering these features are collected in the
:ref:`gallery of examples <examples-index>`, which can be run as Jupyter
notebooks.

Designed for open science
=========================

SpectroChemPy is distributed under the
`CeCILL-B license <https://cecill.info/index.en.html>`__, a free-software
license with characteristics similar to BSD licenses, which permits use,
modification, and redistribution under its stated conditions. Users of the
software in scientific work are also asked to cite the project as described in
the :ref:`citation guidelines <citing>`.

Processing is scriptable, so a complete analysis can be written, versioned, and
shared as a script or notebook. The dataset ``history`` field provides an
inspectable record for supported operations. :ref:`Project <userguide.objects>`
can organize related datasets and subprojects, but complete reproducibility
still requires the scripts, parameters, input files, software versions, and
execution environment to be archived separately.

.. note::

   SpectroChemPy does not guarantee "data integrity": the validity of the raw
   data and the appropriate use of the tools remain the responsibility of the
   user. It is open-source software that may contain bugs; users should consult
   the release notes and known issues for information about identified problems
   affecting the versions they use.

Part of the scientific Python ecosystem
=======================================

SpectroChemPy is built on NumPy and interoperates with the wider Python
ecosystem. It is not a replacement for it. Numerical values can be exposed as
NumPy arrays, and datasets can be converted to xarray representations. The
scientific context preserved by each conversion depends on the target
representation and the documented mapping. The scikit-learn-compatible
estimator and pipeline interfaces allow SpectroChemPy operations to be combined
with general statistical learning tools.

A **scientific core** is separated from optional **plugins** that cover
domain-specific needs: readers for NMR experiments, PerkinElmer instruments,
and Carroucell systems, as well as hypercomplex data, tensor learning, and
2D-IRIS analysis tools. Plugins are installed when needed and are listed in the
:ref:`plugin documentation <plugins>`.

See the :ref:`See also <see_also>` page for complementary projects and
:ref:`bibliography <bibliography>` for scientific works using or relating to
SpectroChemPy.

When another tool is more appropriate
=====================================

You might NOT want to use SpectroChemPy if:

- you prefer a purely graphical workflow and wish to avoid scripting. As an
  API, SpectroChemPy requires writing commands and small scripts; the notebooks
  and examples of the gallery should make this accessible, but it is a
  programming interface.
- the data you work with is well described by a simple tabular model and the
  scientific context (axes, units, masks, history) does not need to be carried
  through the analysis. A spreadsheet or a data-frame library may then be all
  you need.
- you work on spectroscopic data that are difficult to process with
  SpectroChemPy, which is currently mainly focused on optical spectroscopy and
  NMR. Missing readers or methods can be suggested
  (:ref:`contributing.bugs_report`); requests are considered to broaden the
  supported scope.
- you work on very sensitive applications (health, chemical safety, ...) where
  the risk of using software under active development
  cannot be accepted. That is a legitimate choice.
- you are fully satisfied with your current tools. We do not dispute that, and
  we remain open to your opinion and suggestions
  (:ref:`contributing.bugs_report`).

Limitations
===========

- The richness of imported metadata depends on the source format: a reader can
  only map the information the file actually provides.
- The ``history`` field records what was done to a dataset object; it is not a
  complete provenance record. Reproducing a workflow also requires archiving
  the scripts, input data, parameters, software versions, and execution
  environment.
- Not every combination of coordinates, units, labels, masks, or metadata is
  supported by every operation, and features are documented for the version in
  which they are available.
- Supported formats and methods may be provided by the core or by a plugin;
  check the exact software environment for the claims you rely on.

Getting started
===============

* :ref:`Installation guide <installation>`
* :ref:`Gallery of examples <examples-index>`
* :ref:`User's guide and tutorials <user_guide>`
* :ref:`Getting help <getting_help>`
* :ref:`How to cite <citing>` and :ref:`papers using SpectroChemPy <papers>`
