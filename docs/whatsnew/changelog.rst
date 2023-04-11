:orphan:

What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the `revision` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

- Readers: add download() method

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

This version introduce a full refactoring of the Spectrochempy analysis functions.
See the examples in order to understand the changes.

* For a rapid transition note that calling the analysis function is now done in two steps.
  When before, one was writing::

  .. ipython:: python

    scp.set_loglevel("INFO")
    # init the MCRALS object and fit the model on X dataset using a guess.
    mcr = scp.MCRALS(X, guess, tol=0.001)
    # use the MCRALS object attributes
    mcr.C.T.plot()
    mcr.St.plot()

  now one would write:

  .. ipython:: python

    # init the MCRALS object
    mcr = scp.MCRALS(log_level="INFO", tol=0.001)
    # fit the model on X dataset using a guess.
    mcr.fit(X, guess)
    # use the MCRALS object attributes
    mcr.C.T.plot()
    mcr.St.plot()

* Analysis object methods such as  `reconstruct` and `reduce` have been
  renamed `inverse_transform` and  `transform`\ , respectively, in line with
  the method naming in `sklearn`.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
