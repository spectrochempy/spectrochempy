
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
- Improvement of the installation process using pip.
  One can now install the package using command
  like `pip install spectrochempy[cantera]` to install the package with the cantera
  the `cantera` dependencies,
  or `pip install -e ".[dev]"` to install the package with the
  development dependencies and in editable mode.

.. section

Bug fixes
~~~~~~~~
.. Add here new bug fixes (do not delete this comment)
- Fix bug #777. Now `scp.show_version()`` works as expected.

.. section

Dependency updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
