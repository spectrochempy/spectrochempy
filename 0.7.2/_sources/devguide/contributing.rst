.. _contributing:

******************************
Contributing to SpectroChemPy
******************************

.. contents:: Table of contents:
   :local:

General Principles
==================

Any kind of contribution is welcome! While we encourage following these guidelines,
don't let them prevent you from contributing. This guide covers both basic
contributions like issue reporting and advanced topics like code development.

For information about reporting issues, please see the :doc:`issues page </devguide/issues>`.

Getting Started with Development
================================

Version Control Setup
---------------------

1. Create a `GitHub account <https://github.com/signup/free>`__ if you haven't
   already
2. Install Git:

   * macOS: Via `XCode <https://developer.apple.com/xcode/>`__
     or `git-scm.com <https://git-scm.com/download/mac>`__
     or `Homebrew <https://brew.sh>`__ using ``brew install git``
   * Windows: From `git-scm.com <https://git-scm.com/download/win>`__
   * Linux: ``sudo apt-get install git`` or ``pip install gitpython``

Optional: Install a GUI client like `SourceTree <https://www.sourcetreeapp.com>`__
or `GitHub Desktop <https://desktop.github.com>`__

.. _contributing.environment:

Setting Up Your Development Environment
---------------------------------------

1. Fork the repository on GitHub
2. Clone your fork:

    .. sourcecode:: bash

       git clone https://github.com/your-user-name/spectrochempy.git
       cd spectrochempy
       git remote add upstream https://github.com/spectrochempy/spectrochempy.git

3. Create a Python virtual environment:

   .. tabs::

    .. tab:: MacOS/Linux

       .. sourcecode:: bash

          python3 -m venv .venv
          source .venv/bin/activate

    .. tab:: Windows

       .. sourcecode:: bash

          python -m venv .venv
          .venv\Scripts\activate

4. Install SpectroChemPy in development mode:

   .. sourcecode:: bash

       python -m pip install --upgrade pip
       python -m pip install -e ".[dev]"  # Installs development and test dependencies
       # or
       python -m pip install -e ".[dev, docs]"  # Installs development, test and documentation dependencies

Making Changes
--------------

1. Create a feature branch:

   .. sourcecode:: bash

       git checkout -b my-new-feature

2. Make your changes and commit them:

   .. sourcecode:: bash

       git add modified-files
       git commit -m "ENH: Your descriptive commit message"

   Commit prefix conventions:
   * ENH: Enhancement
   * FIX: Bug fix
   * DOC: Documentation
   * TEST: Testing
   * BUILD: Build changes
   * PERF: Performance
   * MAINT: Maintenance

3. Push to GitHub:

   .. sourcecode:: bash

       git push origin my-new-feature

4. Create a Pull Request on GitHub

Maintaining Your PR
-------------------

To update your PR with upstream changes:

.. sourcecode:: bash

    git checkout my-new-feature
    git fetch upstream
    git merge upstream/master
    git push origin my-new-feature

Tips for Success
----------------

* Reference related issues
* Keep changes focused and PRs small
* Ensure tests pass
* Update your PR regularly
* Follow the code style guidelines
