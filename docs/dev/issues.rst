.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are a very important part of any sofware project. Helping us to discover issues or proposing enhancements
should allow making |scpy| more stable, reliable and adapted to more scientific activities.

Please report Bug issues you discover to the
`Issue Tracker  <https://redmine.spectrochempy.fr/projects/spectrochempy/issues>`_

If you do not sign in, you have readonly right on this page. So please register and sign in to be able to post issues.

After login, you should see this:

.. image:: images/issue_redmine_1.png
    :width: 600 px
    :alt: Issues
    :align: center

Before creating a new issue, it is worth searching for existing bug reports and
pull requests to see if the problem has already been reported and/or fixed.

Bug reports should :

#.  Include a short stand-alone Python snippet reproducing the problem.

    You can format the code using `GitHub Flavored Markdown
    <http://github.github.com/github-flavored-markdown/>`_

    .. sourcecode:: ipython

        >>> import spectrochempy as scp
        >>> nd = scp.NDDataset(...)
        ...

#.  Include the full version string of |scpy|. You can use the
    built in property

    .. sourcecode:: ipython

        >>> import spectrochempy as scp
        >>> scp.version

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the |scpy| community and be open to comments/ideas
from others.

.. image:: images/issue_redmine_2.png
    :width: 800 px
    :alt: Issue_fill_in
    :align: center


Template for bug reports
---------------------------

You can use the following template when repporting a problem with |scpy|.

.. sourcecode:: md

    Title
    ------
    <!-- Descriptive title as praise as possible but not too long (<80 chars)-->

    Description
    ------------

    **System information**

    - SpectroChemPy version: [e.g. 0.1.16]
    - OS: [e.g. Windows 10]
    - Python version: 3.7

    **Expected behavior**

    <!-- A clear and concise description of what you expected to happen. -->

    **Actual behavior**

    <!-- A clear and concise description of what the bug is. -->

    **To Reproduce**

    Steps to reproduce the behavior:

    1. Open '...'
    2. Run '....'
    3. See error '....'

    **Attachments**

    <!-- If applicable, attach scripts and/or input files to help explain your problem.
         Please do *not* attach screenshots of code or terminal output. -->

    **Additional context**

    <!-- Add any other context about the problem here. -->
