
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

Added discrete detector-sample fields to ``find_peaks(..., as_result=True)``
rows. ``PeakFindingResult`` and ``PeakTable`` now expose ``sample_index``,
``sample_position`` and ``sample_height`` alongside the existing refined
``position`` and ``height`` values, preserving the local SciPy detector index
without reconstructing it from interpolated positions.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


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

PERF: Skipped the redundant defensive copy of the first operand in out-of-place
binary and unary arithmetic result construction, removing one root dataset
reconstruction per operation (root copies from 2 to 1 for dataset/scalar and
from 4 to 3 for dataset/dataset) while preserving results, units, masks,
coordinates, metadata, and operand non-mutation semantics. In-place operators
keep their defensive copy unchanged.
