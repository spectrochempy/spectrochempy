:orphan:

What's new in revision 0.7.0
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.7.0.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* Improvement of the installation process using pip.
  One can now install the package using command
  like `pip install spectrochempy[cantera]` to install the package with the cantera
  the `cantera` dependencies,
  or `pip install -e ".[dev]"` to install the package with the
  development dependencies and in editable mode.
* Now support installation on the last version (3.10) of Google Colab.
* Add a `show_version` script executable from a terminal.
* Revision/update of the documentation to improve the user experience.

Bug fixes
~~~~~~~~~

* Bug #777. Now `scp.show_version()`` works as expected

Dependency updates
~~~~~~~~~~~~~~~~~~

* Major Python compatibility updates:
    - Now supports Python 3.13
    - Minimum Python version increased to 3.10
    - Dropped support for Python 3.9 and below

* New installation options:
    - Updated [dev], [test], and [docs] extras with latest versions
    - All dependencies now specify minimum versions for better compatibility

Breaking changes
~~~~~~~~~~~~~~~~

* Minimum Python version requirement increased to 3.10
* Several core dependencies require major version updates
* Installation process changes with new dependency groups
