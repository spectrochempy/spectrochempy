
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

* `read` and the more specific `read_remote`method now handle any url pointing
  to a spectrochempy readeable file. An url to a compressed (zip) files are also accepted.

  .. sourcecode::ipython
      lst = scp.read("https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip")
      lst[-1].plot()


.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

*  documentation information for new releases

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
