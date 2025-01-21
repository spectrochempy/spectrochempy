:orphan:

What's new in revision 0.6.11.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.11.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~
* improvement of the installation process using pip.
  One can now install the package using command
  like `pip install spectrochempy[cantera]` to install the package with the cantera
  the `cantera` dependencies,
  or `pip install -e ".[dev]"` to install the package with the
  development dependencies and in editable mode.
* MCRALS now allows storing the C and St generated at each iteration (storeIteration parameter).
* add a despike method ('whitaker') and improves speed of execution of the default ('katsumoto') method.
* read_srs now accepts TGA and GC filetypes (issue #769).
* add a `show_version` script executable from a terminal.

Bug fixes
~~~~~~~~~
* bug #777. Now `scp.show_version()`` works as expected.
* google colab compatibility (issue #784).
* compatibility with pint>0.24 (issue #765).
* loading of dataset for MCR-ALS with kinetic constraints.
* update title in cdot.

Dependency updates
~~~~~~~~~~~~~~~~~~
* now compatible with numpy>2.0.

Breaking changes
~~~~~~~~~~~~~~~~
* minimum python version set to 3.10.
* jupyter lab and widget related dependencies are not installed by default anymore.
  They must be installed separately.

Deprecations
~~~~~~~~~~~~
* FileSelector and BaseFileSelector are deprecated
