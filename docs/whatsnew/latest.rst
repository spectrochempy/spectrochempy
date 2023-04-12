:orphan:

What's new in revision 0.5.6.dev238
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.5.6.dev238.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

- Readers: add download() method

Breaking changes
~~~~~~~~~~~~~~~~

This version introduce a full refactoring of the Spectrochempy analysis functions.
See the examples in order to understand the changes.

* For a rapid transition note that calling the analysis function is now done in two steps.
  When before, one was writing

  .. code-block:: ipython

      # change log level
      scp.set_loglevel("INFO")
      # init the MCRALS object and fit the model on X dataset using a guess.
      mcr = scp.MCRALS(X, guess, tol=0.001)
      # use the MCRALS object attributes
      mcr.C.T.plot()
      mcr.St.plot()


  now one would write:

  .. code-block:: ipython

     # init the MCRALS object
     mcr = scp.MCRALS(log_level="INFO", tol=0.001)
     # fit the model on X dataset using a guess.
     mcr.fit(X, guess)
     # use the MCRALS object attributes
     mcr.C.T.plot()
     mcr.St.plot()


* Analysis object methods such as  `reconstruct` and `reduce` have been
  renamed `inverse_transform` and `transform`\ , respectively, in line with
  the method naming in `sklearn`\ .
