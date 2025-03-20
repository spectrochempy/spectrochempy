
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also, do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments,
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* **Lazy Import Mechanism**: SpectroChemPy now uses a lazy import mechanism to improve startup time.

  - **Optimized Import Process**: When importing SpectroChemPy, only a minimal set of packages and functions is loaded initially.
  - Additional functionality is loaded on demand when first accessed.

  This approach does not reduce the overall loading time but significantly improves the initial import speed.
  It is particularly beneficial for notebook workflows, where the first execution cell runs much faster.

* Markersize for PCA score and screeplots customizable. (Feature request #841)

  example:

  .. code-block:: python
    ... rest of the code ...

    # ScreePlot
    prefs = scp.preferences
    prefs.lines.markersize = 7
    pca.screeplot()

    # Score Plot
    prefs.lines.markersize = 10
    pca.scoreplot(scores, 1, 2)

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

* ``lazy-loader`` package is required
* ``numpydoc`` package has been added to required dependency list.

.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* **Global Preferences**: SpectroChemPy preferences are now global, so there is no need to store them in `NDDataset` objects.
  As a result, the `"preferences"` attribute has been removed from `NDDataset`.

  This means that old scripts written like the following:

   .. code-block:: python

         import spectrochempy as scp

         ... existing code ...

         prefs = X.preferences  # where X is an NDDataset, and preferences is an attribute of NDDataset
         prefs.figure.figsize = (7, 3)

         ... existing code ...

   should be modified as follows:

   .. code-block:: python

         import spectrochempy as scp

         ... existing code ...

         prefs = scp.preferences  # Preferences are now accessed directly from the "scp" object
         prefs.figure.figsize = (7, 3)

         ... existing code ...

* **Impact of Lazy Loading on Method Calls**: Some methods that were previously accessible as class methods are now only available as API or instance methods.
  This is because class methods are not loaded until the class is instantiated.

  For example, the following will no longer work:

   .. code-block:: python

         NDDataset.read("something")  # ❌ This will no longer work

  Instead, use one of the following:

   .. code-block:: python

         scp.read("something")        # ✅ API method
         scp.NDDataset().read("something")  # ✅ Instance method

  Code should be updated accordingly.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
