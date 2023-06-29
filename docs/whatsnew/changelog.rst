
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

* A new reader has been added: `read_wire` (alias `read_wdf``) to read data from
  the .wdf format (WDF) files produced by the ReniShaw WiRe software.
  This reader is based on the `py_wdf_reader <https://github.com/alchem0x2A/py-wdf-reader>`_ package.
* Added an example for NMR processing.
* Peak finding now handles dimensions other than x (important when data are transposed, or when working on a slice in dimension other than x)

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Fix a bug when slicing dataset with an array or list of index: Multi-coordinates
  were not correctly handled.
* Increase the value of the coordinate linearization condition from 0.1% to 1% spacing variation.
  (linearization was sometimes lost when slicing)
* Fix a missing correction for non-negative spectra in MCR-ALS.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
