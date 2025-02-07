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

* Improvement of the installation process using pip.
  One can now install the package using command
  like `pip install spectrochempy[cantera]` to install the package with the cantera
  the `cantera` dependencies,
  or `pip install -e ".[dev]"` to install the package with the
  development dependencies and in editable mode.
* Now support installation on the last version (3.10) of Google Colab. To do so, use the following command:
  `!pip install "spectrochempy[colab]"`. See documentation for more details
* Add a `show-version` script executable from a terminal.
* Revision of the documentation to improve the user experience.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Bug #777. Now `scp.show_version()`` works as expected

.. section

Dependency updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

* Major Python compatibility updates:
    - Now supports Python 3.13
    - Minimum Python version increased to 3.10
    - Dropped support for Python 3.9 and below

* New installation options:
    - Added [colab] extra for Google Colab compatibility
    - Updated [dev], [test], and [docs] extras with latest versions
    - All dependencies now specify minimum versions for better compatibility

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* Minimum Python version requirement increased to 3.10
* Several core dependencies require major version updates
* Installation process changes with new dependency groups

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
