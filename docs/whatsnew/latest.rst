:orphan:

What's new in revision 0.6.7.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.7.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* A new reader has been added: `read_wire` (alias `read_wdf``) to read data from
  the .wdf format (WDF) files produced by the ReniShaw WiRe software.
  This reader is based on the `py_wdf_reader <https://github.com/alchem0x2A/py-wdf-reader>`_ package.

Bug fixes
~~~~~~~~~

* Fix a bug when slicing dataset with an array or list of index: Multi-coordinates
  were not correctly handled.
