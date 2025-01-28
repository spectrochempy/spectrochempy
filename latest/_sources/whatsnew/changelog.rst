
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

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)
* bug #777. Now `scp.show_version()`` works as expected.
* google colab compatibility (issue #784).
* compatibility with pint>0.24 (issue #765).
* loading of dataset for MCR-ALS with kinetic constraints.
* update title in cdot.
.. section

Dependency updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)
* now compatible with numpy>2.0.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)
* minimum python version set to 3.10.
* jupyter lab and widget related dependencies are not installed by default anymore.
  They must be installed separately.


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
* FileSelector and BaseFileSelector are deprecated
