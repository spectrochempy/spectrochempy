
:orphan:

What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* add `FastICA` analysis module.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Issue #643. Conversion from linear to non linear coord was not working properly.
  This was due to the use of the LinearCoord class which is now deprecated and replaced by Coord.
* File logging has been removed due to its bad impact on the performance.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* The behavior of Coord has been slightly modified. During initialisation
  of a Coord object, a try is given to convert the `data` to a linear array, with
  values evenly spaced. If this is not possible, the data are kept as they are but rounded
  to a number of significant digits (given by the parameter `sigdigits`\ ).
  If the data are linear already, nothing is modified.
* The rounding of the data is now done in the `Coord` class automatically to at least
  2 decimals everytime the `data` are modified and during Coord initialisation,
  unless the parameter `bounding` is set to `False` during intialisation.
* Issue #647. ActionMassKinetics has been optimized and refactored.


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

* The `LinearCoord` class is now deprecated and will be removed in a future version.
  Use the `Coord` class instead which performs now an automatic linearization of the data.
