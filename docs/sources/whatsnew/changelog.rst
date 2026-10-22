
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


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Filter outputs now retain the applied method, effective dimension, and
  compact filter parameters in their structured history entries.
- Whittaker-Eilers filtering now applies along the requested dimension.
- Baseline correction outputs now record the fitted model and its effective
  parameters in their structured history entry instead of only an arithmetic
  subtraction.
- ``basc()`` and ``get_baseline()`` now honor ``ranges=`` passed as a keyword
  instead of silently overwriting it with the positional ranges.

- Restored parsing of unit expressions that use a dot as a separator (e.g.
  ``kg.m`` or ``cm.s^-1``), which pint 0.26.2 rejects. (#1781)
- Importing SpectroChemPy or plotting no longer floods the output with
  Matplotlib "Bad key" warnings with Matplotlib older than 3.11, e.g. on Google
  Colab. A generated ``mycustom.mplstyle`` shipped by mistake in 1.1.1 to 1.1.3
  is removed, along with its unmodified installed copies. (#1789)
- ``preferences.makestyle(..., to_mpl=True)`` no longer raises ``TypeError``.


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


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
