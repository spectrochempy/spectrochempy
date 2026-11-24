
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



.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

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

- Processing context transfer, including baseline correction of imported NMR
  spectra, accepts protected nested metadata while preserving the target's
  read-only state and keeping source metadata independent. Copying metadata
  preserves independently protected subtrees without temporarily unlocking
  the source (:pr:`1969`).

- ``NNLS.coef`` and ``LSTSQ.coef`` now return coefficient arrays for NumPy
  inputs with multiple features, including multiple targets, instead of raising
  ``UnboundLocalError``. Dataset coefficients retain their units and scalar
  coefficients keep their existing return type (:pr:`1968`).

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


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- The experimental ``spectrochempy-cantera`` plugin and its ``PFR`` model
  (``scp.cantera.PFR`` and the deprecated alias ``scp.PFR``) will be removed in
  the next minor release. ``PFR`` cannot be built with any Cantera version 3.0 or
  later and has no known users. Cases that need Cantera can use it directly and
  load the results into an ``NDDataset`` (:pr:`1925`).


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
