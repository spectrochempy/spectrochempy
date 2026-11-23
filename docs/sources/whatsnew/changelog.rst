
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

- ``write_jcamp`` now writes the full label as the ``##TITLE`` of each spectrum
  when the ``y`` coordinate has single-column labels (previously only the first
  character was written). Single-column datetime labels no longer raise a
  ``TypeError`` and are used as acquisition dates (:issue:`1949`).

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
