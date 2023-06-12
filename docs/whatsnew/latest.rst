:orphan:

What's new in revision 0.6.6.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.6.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* Two new baseline algorithms have been added: `asls` and `snip`. See :ref:`Baseline` for details.

Breaking changes
~~~~~~~~~~~~~~~~

* `BaselineCorrection` class has been renamed into
  `Baseline`, and there are changes in the way it
  is now used. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`baseline` for details.

* `abc` (and its alias `ab`) method has been removed in favor of `basc`.
