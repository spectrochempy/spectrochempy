:orphan:

What's new in revision 0.6.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.2.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* `read` and the more specific `read_remote`method now handle any url pointing
  to a spectrochempy readeable file. An url to a compressed (zip) files are also accepted.

  .. sourcecode::ipython
      lst = scp.read("https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip")
      lst[-1].plot()

Bug fixes
~~~~~~~~~

*  documentation information for new releases
