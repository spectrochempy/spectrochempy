
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

* `PLSRegression` (Partial Least Squares regression) method added.

* `read` method now handle any url pointing
  to a spectrochempy readeable file. An url to a compressed (zip) files are also accepted.

  Example:

    .. code-block:: ipython

      lst = scp.read("https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip")
      # lst contains 7 NDDatasets,, display the last
      lst[-1].plot()

* Download from urls can also be done using the `download` method.
  However `read` offers more options such as merging.

* Automatically download the github repository ``spectrochempy_data`` which contains the files
  for the examples and tests. The files are downloaded in the directory scp.preferences.datadir.
  When the program is run for the first time, the availability of the files may take several
  minutes, depending on the quality of the internet connection.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

*  Documentation information for new releases.

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* The `read_remote` method has been removed. Use `read` instead.
* The `download` method has been removed. Use `read` instead.
* The `copy` parameter of `Decomposition` methods has been removed.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

* The `used_components` parameter and attribute of `PCA`, `NNMF`,
  `EFA` is replaced by `n_components`
