
What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the `revision` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* PCA score plot labelling (issue #543).
* Improved loading time
* Plot2D accept a color argument.  In addition to cmap=None,
  it produces single color 2D plot. It also accept a line style parameters.
  e.g.:

  ```
  nd.plot(cmap=None, color='red', ls='dashed')
  ```

  produces a dashed red stack plot.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Masks handling.
* Multicoordinates slicing work correctly.
* Removed some deprecation warnings from numpy library.
* Pin ipywidgets to avoid runtime errors until ipywidgets is fixed.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
