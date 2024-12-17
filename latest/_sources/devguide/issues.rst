.. _contributing.bugs_report:

Reporting Issues
=================

All contributions, bug reports, bug fixes, enhancements requests and ideas are welcome

Bug reports are a very important part of any software project. Helping us to discover issues or proposing enhancements should allow making `SpectroChemPy` more stable, reliable and adapted to more scientific activities.

You can report the ``Bug(s)`` you discover or the ``Feature Requests`` to the `Issue Tracker <https://github.com/spectrochempy/spectrochempy/issues>`__

.. warning::

   The issue tracker is hosted on well known `GitHub <https://www.github.com/spectrochempy/spectrochempy>`__ platform. If you do not sign in, you will have only read right on the issues page. Thus, to be able to post issues or feature requests you need to register for a `free GitHub account <https://github.com/signup/free>`__.

Before creating a new issue, it is worth searching for existing bug reports and pull requests to see if the problem has already been reported and may be already fixed.

Bug reports should :

#.  Include a short stand-alone Python snippet reproducing the problem.

    You can format the code using `GitHub Flavored Markdown <http://github.github.com/github-flavored-markdown/>`__

    .. sourcecode:: ipython3

        >>> from spectrochempy import *
        SpectroChemPy's API ...

        >>> nd = NDDataset()

#.  Include the full version string of `SpectroChemPy` . You can use the built in property

    .. sourcecode:: ipython3

        >>> print(version)
        0.1.23...

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then ishow up to the `SpectroChemPy` community and be open to comments/ideas from others.
