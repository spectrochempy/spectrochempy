:orphan:

What's new in revision 0.6.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.2.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

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

Bug fixes
~~~~~~~~~

*  Documentation information for new releases.

Breaking changes
~~~~~~~~~~~~~~~~

* The `read_remote` method has been removed. Use `read` instead.
