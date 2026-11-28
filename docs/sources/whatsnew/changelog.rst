
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

- Plugins can supply known scientific constraints through
  ``importer.measurement_identity`` for automatic import assembly, including
  nested reads and native files. Known contradictions or plugin refusals keep
  inputs separate even with ``merge=True``; unknown facts remain unknown.
  Explicit combination operations retain their behavior
  (:pr:`2022`, :issue:`1795`).

- Composable Python 1D models accept ``scale="log"`` with an explicit positive
  ``log_reference`` and positive physical bounds in native least-squares mode.
  Initial configurations are copied; fitted values and uncertainty arrays remain
  physical. COMMON syntax and ND numerical conditioning retain their behavior
  (:pr:`2018`, :issue:`1989`).

- ``Optimize(bounds_mode='native')`` opts 1D COMMON and Python least-squares
  fits into physical-coordinate bounded TRF, including endpoint starts, with
  explicit relative-step termination and the existing uncertainty exclusions.
  The default transform policy and ND routes are preserved. Parameter
  declarations accept endpoints; Optimize validates the selected start policy. (:pr:`1985`)

- ``despike(return_mask=True)`` returns the corrected spectrum and a boolean
  dataset identifying modified values on the original coordinates. All methods
  support ``dim`` and independent 1D/ND traces. Masked and non-finite samples
  are preserved; short valid runs are documented. A synthetic example qualifies
  narrow-band distortion and shows why changes are not confirmed cosmic rays
  (:pr:`2011`).

- 1D ``Optimize.result.diagnostics`` qualifies physical first-order stationarity
  separately from solver success, including independent feasible probes near
  bounds. Starting values, solver selection, fit results and ND paths are
  unchanged; unsupported or unresolved checks are explicitly unavailable. (:pr:`1984`)

- ``Model``, ``Gaussian`` and ``Linear`` define composable Python models for
  1D ``Optimize(model=..., params=...)`` fits. ``Parameter`` and ``Parameters``
  configure initial values, fixed parameters, bounds and explicit arithmetic
  links. The existing canonical fitting engine and result uncertainty contract
  are reused; the DSL, ND callbacks and named recovery laws remain supported
  (:pr:`1981`).

- ``Optimize(model="saturation_recovery" or "exponential_decay", parameters=...)``
  fits a native 1D curve from an amplitude and a positive time constant, without
  a user function. Results retain nonuniform physical coordinates, masks and
  units, and expose the fitted amplitude and ``tau`` with their units (:pr:`1970`).

- ``Optimize(model=..., parameters=...)`` fits nonseparable physical models on
  a full multidimensional ``NDDataset`` grid, with optional positive linear
  amplitude projection and native ``FitResult`` outputs (:pr:`1968`).


- ``Baseline(model="arpls")`` and ``Baseline(model="aspls")`` add explicitly
  selected reweighted baseline correction using NumPy/SciPy, also through
  ``get_baseline`` and ``basc``. They accept finite real unmasked 1D or rowwise
  2D spectra on regular grids and record parameters in the correction history.
  Existing defaults are unchanged. Inspect broad and negative peak areas;
  convergence alone does not guarantee quantitative preservation (:pr:`1933`).

- ``Baseline(model="distribution")`` adds experimental automatic signal/noise
  classification, local interpolation and mean smoothing for finite real
  unmasked 1D or row-wise 2D spectra on regular grids.
  ``distribution_width`` is a half-width in samples. Inspect the fitted
  baseline and peak areas: broad weak signals can be removed as background.
  Existing models and defaults are unchanged; no NMR dependency is added
  (:pr:`1932`).

- Run-time hints for surprising defaults. ``align(method="inner")`` warns when
  the intersection keeps fewer than half of the points of the sparsest
  coordinate in the common range, or keeps irregularly spaced points from
  regular grids, and suggests ``method="interpolate"``. ``MCRALS.fit`` lists
  the active constraints, defaults included, in its ``INFO`` log.
  ``Baseline`` ``ranges`` that select no points are reported, in the error
  raised when no support remains or in a warning when they are ignored, with a
  reminder that integer limits are indices and float limits coordinates. The
  error for integer ranges outside the coordinate domain carries the same
  reminder (:issue:`1954`).



.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``sortbydate=False`` now preserves observation order in every assembled
  import group. Previously, the option was consumed by the first merged group,
  so later groups could still be sorted (:issue:`1795`, :pr:`2033`).

- Automatic OMNIC import assembly retains recognized native SPA/SRS axis and
  signal definitions, keeping known contradictions separate across containers
  and nested reads. SRS uses its selected series/background header; associated
  SPA interferograms retain their own signal semantics. SRS binary-content
  inputs, including ZIP members, use the shared reader opening path
  (:issue:`1795`, :pr:`2032`).

- OMNIC SPG imports reject contradictory recognized native axis and signal
  definitions even when their units match. Opt-in member separation preserves
  these constraints through import assembly, including nested ZIP archives.
  Timestamp and source-label rows
  remain paired for two-member groups and date sorting (:issue:`1795`, :pr:`2029`).

- ZIP imports flatten multi-dataset members, including nested archives and
  heterogeneous SPC files, instead of returning nested collections or failing
  during automatic merging. Existing compatibility checks and member-based
  ``only`` limits are preserved (:issue:`1795`, :pr:`2027`).

- Native profile fits with large coordinate origins evaluate their Jacobian
  and local covariance at the center actually returned, after its finite
  precision reconstruction. Local and shared centers retain consistent
  uncertainties with the fitted profile (:issue:`1989`, :pr:`2026`).

- Carroucell imports retain separate compatible groups from one sample holder
  instead of failing when OMNIC returns multiple datasets. Temperature labels
  follow each group's actual scan count and the complete imported time window,
  preserving observation coordinates and units (:issue:`1795`, :pr:`2025`).

- Automatic import merging keeps contradictory named absorbance, transmittance
  and Kubelka-Munk quantities or recognized JCAMP modalities separate instead of
  silently converting or combining them. Unknown identities remain unknown;
  compatible scales within a signal family and explicit combination operations
  are preserved (:pr:`2020`).

- Automatic import merging keeps incompatible retained axes or ordered
  dimensions in separate groups with a warning, instead of assembling equal-size
  spectra under one coordinate. Compatible coordinate-unit rescaling is checked
  with floating-point roundoff tolerance only; explicit ``stack`` and
  ``concatenate`` behavior is unchanged (:pr:`2019`).

- ``despike`` preserves masked values and raises ``ValueError`` for non-positive
  ``delta``, invalid window sizes, complex data and integer values that cannot
  be represented exactly as float64 (:pr:`2011`).

- Model constructors preserve the current global logging level when
  ``log_level`` is omitted or ``None``. Explicit levels still update the global
  setting. This also applies to the IRIS and tensor plugins. (:pr:`2008`)

- 1D ``Optimize`` no longer emits output-clearing control characters in quiet
  mode or terminal processes. Notebook output clearing is restricted to
  ``INFO`` and ``DEBUG`` reporting. The fitting gallery example also omits an
  empty introductory code block (:pr:`2006`).

- General separable ND fits use fitted scales for post-fit numerical derivatives
  of built-in widths and time constants, including shared parameters. Local
  covariance uses the same physical Jacobian. Solver iterations and fitted
  parameters are unchanged (:pr:`2004`).

- ``save_as(..., directory=...)`` respects the explicit destination directory
  for imported datasets and projects instead of using their source directory.
  Absolute filenames retain precedence (:pr:`2001`).

- Descending unsigned integer coordinates retain their values and signed
  spacings instead of being silently replaced by zeros during linearization
  (:pr:`1999`).
- The ``n_components`` argument of ``transform``, ``inverse_transform`` and
  ``get_components`` of decomposition estimators (``PCA``, ``NMF``, ``SVD``,
  ``FastICA``...), and of ``PCA.printev`` and ``PCA.plot_scree``, must be a
  positive integer (or ``None``): ``0`` used to fail with an unrelated error
  and a negative value silently dropped the last components. A ``ValueError``
  now names the invalid value. Values above the fitted number of components
  still use all the components (:pr:`1982`).

- ``inverse_transform(n_components=k)`` of ``PCA``, ``NMF`` and ``FastICA``
  now reconstructs the data from the first ``k`` components only; the
  argument used to be silently ignored. ``FastICA.inverse_transform`` also
  accepts scores with fewer columns than fitted components (for instance from
  ``transform(n_components=k)``) instead of raising a shape error.
  With ``whiten=False``, where scikit-learn ignores ``n_components`` and fits
  all the features, ``FastICA`` now uses all the fitted components by
  default (``get_components()`` used to return only ``n_components`` of
  them). ``FastICA`` with the default ``n_components=None`` no longer fails
  in ``fit`` (:pr:`1982`).

- 1D ``Optimize`` covariance and standard errors now account for the bound
  transformation and describe physical free parameters for both COMMON and
  Python models. Local uncertainty is unavailable near bounds, for
  non-converged or rank-deficient fits, with a diagnostic exclusion reason.
  The raw optimizer-coordinate ``jacobian`` remains unchanged (:pr:`1983`).

- ``SVD.transform``, ``SVD.fit_transform``, ``SVD.inverse_transform``,
  ``SVD.components`` and ``SVD.get_components`` no longer raise
  ``NotImplementedError``. They return ``NDDataset`` scores
  :math:`U \Sigma` (:math:`X V` for new data), right singular vectors
  :math:`V^T` and rank-``n_components`` reconstructions, with the same
  coordinates, units and metadata conventions as ``PCA``. Unlike ``PCA``,
  SVD does not center the data. ``SVD.n_components`` is now available after
  ``fit``. ``U``, ``s`` and ``VT`` are unchanged (:issue:`1972`, :pr:`1980`).

- The default ``WARNING`` log level hides INFO messages again, as documented:
  since 0.8.2, messages such as "This coordinate (...) is outside the axis
  limits", "... is already a masked array", "units forced to change" or reader
  notices (OMNIC, MATLAB, data download) were shown whatever the level.
  ``set_loglevel``, ``preferences.log_level`` and the ``SpectroChemPy.log_level``
  configuration option now apply to the ``spectrochempy`` logger and its child
  loggers, including plugins: ``set_loglevel("INFO")`` shows these messages,
  ``"DEBUG"`` also shows debug messages, and ``"ERROR"`` hides plugin warnings
  (:issue:`1952`).

- Units of datasets and coordinates read from ``.scp`` files are now created by
  SpectroChemPy's unit registry (``scp.ur``), as units created in memory are.
  Combining a reread dataset with a processed one, for example
  ``scp.trapezoid`` on the output of ``Baseline.transform()``, no longer raises
  ``AttributeError``. The file format is unchanged (:issue:`1951`).

- Reading files no longer hides reader errors. Reading a single file (path,
  one-element list or ``content=``) with ``read`` or a format reader
  (``read_omnic``, ``read_jcamp``, ``read_csv``, ...) raises the reader's
  original exception, with a note naming the file, instead of a warning and
  ``None``: for example a corrupted file, or an OMNIC ``.spg`` with inconsistent
  x axes, as documented. A local file or given content (even empty) that
  cannot be read is no longer looked for in the remote test data repository,
  except a file in ``datadir`` that reports a missing file (partially
  downloaded test data); a file downloaded from that repository that cannot be
  read raises too. When several files, a directory or zip members are read, an
  unreadable file is skipped with a warning giving its name and the exception
  type (previously only the message, and file-content ``SpectroChemPyError``
  errors aborted the whole read); if no file can be read, the first error is
  raised instead of returning ``None`` (:issue:`1950`).

- ``write_jcamp`` now writes the full label as the ``##TITLE`` of each spectrum
  when the ``y`` coordinate has single-column labels (previously only the first
  character was written). Single-column datetime labels no longer raise a
  ``TypeError`` and are used as acquisition dates (:issue:`1949`).

- ``asymmetricvoigtmodel`` and ``asymmetricvoigt``: a negative ``asym`` now
  gives the asymmetry on the other side of the peak instead of being silently
  ignored and giving the symmetric profile: the profile for ``asym = -a`` is
  the mirror image, about ``pos``, of the profile for ``+a``. Results for
  ``asym >= 0`` are unchanged (:issue:`1948`).

- ``multiplot`` now accepts fewer datasets than ``nrow * ncol`` panels: the
  unused panels are hidden and still returned with the other axes, instead of
  raising "Not enough datasets given in this list". A grid given with both
  ``nrow`` and ``ncol`` that is too small for the datasets now raises a clear
  ``ValueError`` instead of being silently replaced. When only one of them is
  given, the other is inferred to hold every dataset (``nrow=1`` with three
  datasets gives a 1 x 3 grid). Without ``nrow`` and ``ncol``, the automatic
  grid now holds every dataset: three datasets are plotted on a 2 x 2 grid
  instead of dropping one, and five or more no longer fail with an
  ``IndexError``; one, two and four datasets keep their layout. Calling
  ``multiplot`` without any dataset still raises a ``ValueError``, with a
  clearer message, and ``nrow`` or ``ncol`` values that are not positive
  integers raise a ``ValueError`` (:issue:`1946`).

- ``Baseline.plot`` and ``plot_baseline`` now emit a warning naming the keyword
  arguments they ignore, instead of silently dropping them. The documentation
  lists the supported options and the gallery example no longer passes unused
  ones (:issue:`1976`, :pr:`1977`).

- ``EFA.f_ev`` and ``EFA.b_ev`` are now titled "eigenvalues" instead of
  inheriting the title of the input data (:issue:`1973`, :pr:`1977`).

- The ``MCRALS`` ``normSpec="euclid"`` help now correctly states that spectra
  are normalized by their Euclidean (L2) norm, not their total area; behavior is
  unchanged (:issue:`1975`, :pr:`1977`).

- ``read_csv`` now warns, with the file name and the number of ignored columns,
  when a CSV file has more than one data column after the x column (only the
  first data column is read), and its docstring examples show the actual
  behavior (:issue:`1971`, :pr:`1978`).

- Printing or displaying an ``NDDataset``, ``Coord`` or ``CoordSet`` no longer
  changes the global NumPy print options; user arrays keep their own
  ``np.set_printoptions`` settings (:issue:`1974`, :pr:`1979`). The text of these objects,
  analysis-result parameters and diagnostics, and binary-operation history
  entries is unchanged. Array-valued pint
  ``Quantity`` objects (for instance ``Coord.values``), which are formatted by
  pint rather than by SpectroChemPy, now follow the user's NumPy print options.

- Processing context transfer, including baseline correction of imported NMR
  spectra, accepts protected nested metadata while preserving the target's
  read-only state and keeping source metadata independent. Copying metadata
  preserves independently protected subtrees without temporarily unlocking
  the source (:pr:`1969`).

- ``NNLS.coef`` and ``LSTSQ.coef`` now return coefficient arrays for NumPy
  inputs with multiple features, including multiple targets, instead of raising
  ``UnboundLocalError``. Dataset coefficients retain their units and scalar
  coefficients keep their existing return type (:pr:`1968`).

- ``concatenate`` and ``stack`` now take the coordinates of the
  non-concatenated dimensions from the first dataset, as specified by the
  concatenate/stack metadata policy, instead of silently from the last one.
  They warn when these coordinates differ between inputs: numeric values in
  only some inputs, or values differing by more than the absolute 3-decimal
  tolerance of dataset arithmetic (1.5e-3 in the units of the first dataset,
  after unit conversion); labels are not compared. Use
  ``align(..., method="interpolate")`` first to put the datasets on common
  coordinates. ``stack`` of datasets without coordinates no longer files the
  new stacking coordinate under the wrong dimension (:pr:`1955`).

- 2D plots: with ``clear=False``, line (stack) plots now keep the vertical
  limits of the lines already drawn on the axes instead of clipping them to
  the new dataset, as 1D plots do. ``x_reverse=False``, ``reverse=False`` and
  ``y_reverse=False`` now disable the automatic reversal of wavenumber and ppm
  axes, as documented; without these parameters the axes are reversed as
  before (:issue:`1945`, :pr:`1956`).

- ``LSTSQ.coef`` and ``NNLS.coef`` now work for several features, whether the
  inputs are ``NDDataset`` objects, with or without coordinates, or arrays. The
  coefficients have shape (targets, features), take the coordinates of the
  matching axes of ``Y`` and ``X`` when available, and are returned as an
  array for array inputs. Dataset outputs are independent copies, so changing
  them does not change the fitted model. ``intercept`` follows the same rules
  for several targets, and returns ``0.0`` when ``fit_intercept=False`` instead of raising
  an error (:issue:`1947`, :pr:`1958`).

- ``Optimize.usermodels`` now keeps each registered model in an independent
  adapter. Registering another function or configuring another optimizer no
  longer replaces existing model functions and parameter signatures, preventing
  incorrect fits with multiple user models. Model instances retain their bound
  state and have independent parameter-name lists (:pr:`1941`).

- ``ActionMassKinetics``: the analytical Jacobian (``use_jac=True``) is now correct
  for stoichiometric coefficients greater than one; ``fit_to_concentrations``
  honors ``method`` in ``optimizer_kwargs`` (it was silently ignored, and
  Nelder-Mead was always used); the optimizer ``left_op`` is applied
  once per evaluation instead of accumulating, and the model holds the optimal
  parameters (and returns the matching concentrations) after the fit. With a
  temperature profile ``T(t)``, ``k_dt`` is now optional (rate constants are then
  computed at each step), the gas constant matches the isothermal path
  (8.314462618 instead of 8.314, a relative change of about 1e-3 on rate
  constants), and plain rate constants raise a clear error (:pr:`1923`).


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Reading a single unreadable file with ``read`` or a format reader now raises
  the reader's exception instead of emitting a warning and returning ``None``;
  reading several files raises when none of them can be read. Replace
  ``if ds is None:`` checks by ``try: ... except Exception: ...`` around the
  read (:issue:`1950`).

- ``plot_compare`` (and ``plot_merit``) legends now use "experimental" and
  "calculated" instead of automatic dataset names such as
  ``NDDataset_1a2b3c4d``; names set on the datasets remain the default labels.
  The legend is built from the plotted lines, so it shows their line styles and
  transparency, and lists the other labelled artists of the axes whose label is
  not already shown (:issue:`1954`).

- ``Baseline`` ``ranges`` that select no points now raise a ``ValueError``
  instead of a ``TraitError`` when no baseline support remains
  (:issue:`1954`).


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- Passing ``labels`` to ``plot_compare`` is deprecated: it is ignored and emits
  a ``DeprecationWarning``. Use ``exp_label``, ``calc_label`` and
  ``resid_label`` instead (:issue:`1954`).

- The experimental ``spectrochempy-cantera`` plugin and its ``PFR`` model
  (``scp.cantera.PFR`` and the deprecated alias ``scp.PFR``) will be removed in
  the next minor release. ``PFR`` cannot be built with any Cantera version 3.0 or
  later and has no known users. Cases that need Cantera can use it directly and
  load the results into an ``NDDataset`` (:pr:`1925`).


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- TEST: Make lazy-import timing diagnostic on shared CI runners while retaining
  the Matplotlib loading guard. Capture expected Conda CLI errors, check the
  intended rejection reasons, and render gallery templates before strict
  Docutils heading checks.

- DOC: Document defaults that can surprise where they apply: coordinate
  rounding (``Coord``, ``interpolate``), contour start level and axis reversal
  (``plot``), ``Baseline`` range limits, parameter order of ``Optimize``
  uncertainties, ``MCRALS`` default unimodality and tolerance, and the
  ``blocks`` parameter of ``ModelProfile``. New gallery examples show Gaussian
  broadening in coordinate units and augmented MCR-ALS with fixed reference
  concentrations (:issue:`1953`).
