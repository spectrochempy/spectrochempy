.. _contributing.bugs_report:

:orphan:  # Add this to mark it as standalone

Reporting Issues
=================

All contributions, bug reports, bug fixes, enhancement requests and ideas are welcome.

Bug reports are a very important part of any software project. Helping us discover issues or proposing enhancements will allow making `SpectroChemPy` more stable, reliable and better adapted to scientific activities.

You can report the ``Bug(s)`` you discover or the ``Feature Requests`` to the `Issue Tracker <https://github.com/spectrochempy/spectrochempy/issues>`__

.. warning::

   The issue tracker is hosted on well known `GitHub <https://www.github.com/spectrochempy/spectrochempy>`__ platform. If you do not sign in, you will have only read right on the issues page. Thus, to be able to post issues or feature requests you need to register for a `free GitHub account <https://github.com/signup/free>`__.

Before creating a new issue, please search existing bug reports and pull requests to see if the problem has already been reported and/or fixed.

Bug reports should:

#. Include a short stand-alone Python snippet that reproduces the problem.
   You can format the code using `GitHub Flavored Markdown`_

   .. sourcecode:: ipython3

       >>> from spectrochempy import *
       SpectroChemPy's API ...

       >>> nd = NDDataset()


#. Include the full version string of SpectroChemPy.
   You can use the built-in property:

   .. sourcecode:: ipython3

       >>> print(version)
       0.1.23...


#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the SpectroChemPy community and be open to comments/ideas from others.

.. _GitHub Flavored Markdown: http://github.github.com/github-flavored-markdown/
