:orphan:

What's new in revision 0.6.11.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.11.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~
- Improvement of the installation process using pip.
  One can now install the package using command
  like `pip install spectrochempy[cantera]` to install the package with the cantera
  the `cantera` dependencies,
  or `pip install -e ".[dev]"` to install the package with the
  development dependencies and in editable mode.

Bug fixes
~~~~~~~~
- Fix bug #777. Now `scp.show_version()`` works as expected.
