.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are a very important part of any sofware project. Helping us to
discover issues or proposing enhancements should allow making |scpy| more
stable, reliable and adapted to more scientific activities.

Please report Bug issues you discover to the
`Issue Tracker  <https://github.com/spectrochempy/spectrochempy/issues>`_

If you do not sign in, you have readonly right on this page.
To be able to post issues or feature requests you need to register.

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