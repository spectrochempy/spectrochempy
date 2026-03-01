.. _contributing:


******************************
Contributing to SpectroChemPy
******************************

.. contents:: Table of contents:
   :local:

General Principles
===================

The instructions below are a general guide. We are (or rather will be) doing our best to follow this guide, so if you want to contribute, we encourage you to follow it as well. But you don't have to follow everything to the letter: any kind of contribution is welcome!

In this guide, we will talk about some basic but very useful contributions such as issues reporting and of some more advanced topics concerning contributions to documentation and to the code base.

.. toctree::

    issues


Be prepared to work on the code
===============================

To contribute further to the code and documentation, you will need learning how to work with GitHub and the `SpectroChemPy` Code base.

.. _contributing.version_control:

Version control, Git, and GitHub
---------------------------------

The code of `SpectroChemPy` is hosted on `GitHub <https://www.github.com/spectrochempy/spectrochempy>`__. To contribute, you will need to register for a `free GitHub account <https://github.com/signup/free>`__. Actually, you may have already done so by submitting issues, so you are also almost ready to contribute to the code.

The reason we use `Git <https://git-scm.com/>`__ for version control of our code is to allow multiple people to work on this project simultaneously.

However, working with `Git <https://git-scm.com/>`__ is unfortunately not the easiest step for a newcomer when trying to contribute to open-source software. But learning this versioning system along with developing with Python can be very rewarding for your daily work.

To learn `Git <https://git-scm.com/>`__ , you may want to check out the `GitHub help pages <https://help.github.com/>`_ or the
`NumPy's documentation <https://numpy.org/doc/stable/dev/index.html>`__ . There is no shortage of resources on the web on this subject and many tutorials are available.

Below we give some essential information to get you started with **Git**, but, of course, if you encounter difficulties, don't hesitate to ask for help. This may help to solve your problems faster, but it also allows us to improve this part of our documentation based on your feedback.

Installing git
---------------

`GitHub <https://help.github.com/set-up-git-redirect>`__ has instructions for installing and configuring git.  All these steps need to be completed before you can work seamlessly between your local repository and GitHub.

Git is a free and open source distributed control system used in well-known software repositories, such as
`GitHub <https://github.com>`__ or `Bitbucket <https://bitbucket.org>`__ . For this project, we use a GitHub
repository: `spectrochempy repository <https://github.com/spectrochempy/spectrochempy>`__.

Depending on your operating system, you may refer to these pages for installation instructions:

-  `Download Git for macOS <https://git-scm.com/download/mac>`__ (One trivial option is to install `XCode <https://developer.apple.com/xcode/>`__ which is shipped with the git system).

-  `Download Git for Windows <https://git-scm.com/download/win>`__ .

-  `Download for Linux and Unix <https://git-scm.com/download/linux>`__ . For the common Debian/Ubuntu distribution, it is as simple as typing in the Terminal:

   .. sourcecode:: bash

       sudo apt-get install git

-  Alternatively, once miniconda or anaconda is installed (see :ref:`contributing_environment` ), one can use conda to install
   git:

   .. sourcecode:: bash

       conda install git

To check whether or not *git* is correctly installed, use

.. sourcecode:: bash

   git --version

Optional: installing a GUI git client
-------------------------------------

Once your installation of git is complete, it may be useful to install a GUI client for the git version system.

We have been using `SourceTree client <https://www.sourcetreeapp.com>`__ (which can be installed on both Windows and Mac operating systems). To configure and learn how to use the sourcetree GUI application, you can consult this
`tutorial <https://confluence.atlassian.com/bitbucket/tutorial-learn-bitbucket-with-sourcetree-760120235.html>`__

However, any other GUI can be interesting such as `Github-desktop <https://desktop.github.com>`__, or if you prefer, you can stay using the command line in a terminal.

.. note::

   Many IDE such as `PyCharm <https://www.jetbrains.com/fr-fr/pycharm/>`__ have an integrated GUI git client which can be used in place of an external application. This is an option that we use a lot (in combination to the more visual SourceTree application)

.. _contributing.forking:

Forking the spectrochempy repository
------------------------------------

You will need your own fork to work on the code. Go to the `SpectroChemPy project page <https://github.com/spectrochempy/spectrochempy>`__ and hit the ``Fork`` button to create an exact copy of the project on your account.

Then you will need to clone your fork to your machine. The fastest way is to type these commands in a terminal on your machine:

.. sourcecode:: bash

   git clone https://github.com/your-user-name/spectrochempy.git localfolder
   cd localfolder
   git remote add upstream https://github.com/spectrochempy/spectrochempy.git

This creates the directory ``localfolder`` and connects your repository to the upstream (main project) `SpectroChemPy` repository.

.. _contributing_environment:

Creating a Python development environment
------------------------------------------

To test out code and documentation changes, you'll need to build `SpectroChemPy` from source, which requires a Python environment.

* Install either `Anaconda <https://www.anaconda.com/download/>`_, `miniconda
  <https://conda.io/miniconda.html>`_, or `miniforge <https://github.com/conda-forge/miniforge>`_
* Make sure your conda is up to date (``conda update conda`` )
* Make sure that you have :ref:`cloned the repository <contributing.forking>`

* ``cd`` to the `SpectroChemPy` source directory (*i.e.,* ``localfolder`` created previously)

We'll now install `SpectroChemPy` in development mode following 2 steps:

1. Create and activate the environment. This will create a new environment and will not touch
   any of your other existing environments, nor any existing Python installation.
   (conda installer is somewhat very slow, this is why we prefer to replace it by
   `mamba <https://https://github.com/mamba-org/mamba>`__.

   .. sourcecode:: bash

      conda update conda -y
      conda config --add channels conda-forge
      conda config --add channels cantera
      conda config --add channels spectrocat
      conda config --set channel_priority flexible
      conda install mamba jinja2

   Here we will create un environment using python in its version 3.12
   but it is up to you to install any version from 3.10 to 3.12.
   Just change the relevant information in the code below (the first line uses a
   script to create the necessary yaml
   file containing all information about the packages to install):

   .. sourcecode:: bash

      python .ci/env_create.py -v 3.12 --dev scpy3.12.yml
      mamba env create -f .ci/scpy3.12.yml
      conda activate scpy3.12

2. Install `SpectroChemPy`

   Once your environment is created and activated, we must install SpectroChemPy
   in development mode.

   .. sourcecode:: bash

      (scpy3.12) $ cd <spectrochempy folder>
      (scpy3.12) $ python -m pip install -e .

   At this point you should be able to import spectrochempy from your local
   development version:

   .. sourcecode:: bash

      (scpy3.12) $ python

   This start an interpreter in which you can check your installation.

   .. sourcecode:: python


     >>> print(scp.version)
     SpectroChemPy's API ...
     >>> exit()

Controlling the environments
----------------------------

You can create as many environments you want, using the method above
(for example with different versions of Python)

To view your environments:

.. sourcecode:: bash

   conda info -e

To return to your root environment:

.. sourcecode:: bash

   conda deactivate

See the full conda docs `here <https://conda.pydata.org/docs>`__.

Creating a branch
-----------------

Generally we want the master branch to reflect only production-ready code, so you will have to create a
feature branch for making your changes. For example

.. sourcecode:: bash

    git branch my_new_feature
    git checkout my_new_feature

The above can be simplified to:

.. sourcecode:: bash

    git checkout -b my_new_feature

This changes your working directory to the ``my-new-feature`` branch.  Keep any changes in this branch specific to one bug or feature so it is clear what the branch brings to spectrochempy. You can have many ``my-other-new-feature``
branches and switch in between them using the:

.. sourcecode:: bash

    git checkout command.

When creating this branch, make sure your master branch is up to date with the latest upstream master version. To update your local master branch, you can do

.. sourcecode:: bash

    git checkout master
    git pull upstream master --ff-only


When you want to update the feature branch with changes in the master after you have created the
you have created the branch, see the section on
:ref:`updating a PR <contributing.update-pr>` .

Contributing your changes to SpectroChemPy
==========================================

.. _contributing.commit-code:

Commit your code
--------------------

.. note::

    If you are not easy with the command lines,
    Remember that all the following `git` operations can be done using a GUI application such as ``sourcetree``
    or whatever you prefer.

Keep style corrections in a separate commit to make your pull request more readable.

Once you've made changes, you can see them by typing:

.. sourcecode:: bash

    git status

If you created a new file, it is not tracked by git. Add it by typing:

.. sourcecode:: bash

    git add path/to/file-to-be-added

By typing `git status` again, you should get something like

.. sourcecode:: bash

    # On the my-new-feature branch
    #
    # modified: /path/to/file-to-be-added
    #

Finally, commit your changes to your local repository with an explanatory message.

It is recommended to use a convention for prefixes and the presentation of commit messages.
Here are some common prefixes:

* ENH: Enhancement, new feature
* FIX: Bug fixes
* DOC: Documentation additions/updates
* TEST: Test additions/updates
* BUILD: Build process/script updates
* PERF: Performance improvements
* MAINT: Code cleanup

The following defines how a commit message should be structured:

* a subject line with ``< 80`` chars (optionally starting with one of the above prefixes).
* One blank line.
* Optionally, a commit message body.

Please reference the
relevant GitHub issues in your commit message using GH1234 or #1234.


Now you can commit your changes to your local repository:

.. sourcecode:: bash

    git commit -m <message>

.. _contributing.push-code :

Push your changes
--------------------

When you want your changes to appear publicly on your GitHub page, push the commits of your forked feature branch:

.. sourcecode:: bash

    git push origin my-new-feature

Here, `origin` is the default name given to your remote repository on GitHub.
You can see the remote repositories:

.. sourcecode:: bash

    git remote -v

If you added the upstream repository as described above, you'll see something
like

.. sourcecode:: bash

    origin git@github.com:yourname/spectrochemp.git (fetch)
    origin git@github.com:yourname/spectrochempy.git (push)
    upstream git://github.com/spectrochempy/spectrochempy.git (fetch)
    upstream git://github.com/spectrochempy/spectrochempy.git (push)

Now your code is on GitHub, but it is not yet part of the SpectroChemPy project. For this to happen, a pull request must be submitted on GitHub.

Review Your Code
----------------

When you are ready to request a code review, file a review request. Before doing so, make sure
again that you have followed all the guidelines described in this document
regarding code style, testing, performance testing and documentation. You should also
check the changes in your branch against the branch on which it was based:

#. Navigate to your repository on GitHub -- https://github.com/your-user-name/spectrochempy
#. Click on ``Branches`` .
#. Click on the ``Compare`` button for your feature branch.
#. Select the ``base`` and ``compare`` branches, if necessary. This will be ``master`` and
   and ``my-new-feature`` , respectively.

Make the pull request (PR)
------------------------------

If everything looks good, you are ready to make a pull request.  A pull request is the way
code from a local repository is made available to the GitHub community can be
reviewed and eventually merged into the master version.  This request and its associated changes
will eventually be integrated into the master branch and available in the next release.  To submit a change request:

#. Navigate to your repository on GitHub.
#. Click the ``Pull Request`` button.
#. You can then click on ``Commits`` and ``Files Changed`` to make sure that everything is fine one last time
#. Write a description of your changes in the ``Preview Discussion`` tab.
#. Click on ``Send Pull Request`` .

This request will then be sent to the repository maintainers, and they will review
the code.

.. _contributing.update-pr:

Update your pull request
--------------------------

Depending on the evaluation of your pull request, you will probably need to make
some changes to the code. In this case, you can make them in your branch,
add a new commit to that branch, push it to GitHub. This will automatically update your pull request with the latest
code and restart the
:any:`Continuous Integration <contributing.ci>` tests.

Another reason you might need to update your pull request is to resolve conflicts
with changes that have been merged into the master branch since you opened your
pull request.

To do this, you need to ``merge upstream master`` in your branch:

.. sourcecode:: bash

    git checkout my-new-feature
    git fetch upstream
    git merge upstream/master

If there are no conflicts (or if they were able to be fixed automatically),
a file with a default commit message will open, and you can simply save and exit this file.

If there are merge conflicts, you must resolve the conflicts. See for example
example at https://help.github.com/articles/resolving-a-merge-conflict-using-the-command-line/
for an explanation of how to do this.

Once the conflicts are merged and the files where the conflicts were resolved are added, you can run
``git commit`` to save these corrections.

If you have uncommitted changes at the time you want to update the branch with
the master, you'll need to ``stash`` them before you update (see the
`stash docs <https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning>`__).
This will effectively store your changes and they can be reapplied after the update.

After the feature branch has been updated locally, you can now update your pull
request by pushing again to the branch on GitHub.

Automatically fix formatting errors
-----------------------------------

We use several style checks (i.e., ``black``, ``flake8``) that are run after
you make a download request. If there is a scenario where one of these checks fails then you
can comment:

.. sourcecode:: bash

    @github-actions pre-commit

On that pull request. This will trigger a workflow that will automatically correct the formatting errors.

To automatically correct formatting errors on every commit you make, you can
configure the pre-commit yourself. First, create a Python environment :ref:`environment
<contributing_environment>` , then configure :ref:`pre-commit <contributing.pre-commit>` .

Delete your merged branch (optional)
------------------------------------

Once your feature branch is accepted upstream, you will probably want to get rid of
the branch. First, merge upstream master into your branch so that git knows it's safe to
delete your branch:

.. sourcecode:: bash

    git fetch upstream
    git checkout master
    git merge upstream/master

Then you can do

.. sourcecode:: bash

    git branch -d my-new-feature

Make sure you use a lowercase ``d``, otherwise git won't tell you if your feature branch
branch hasn't been merged.

The branch will still exist on GitHub, so to delete it, do

    git push origin --delete my-new-feature


Tips for a successful pull request
==================================

To improve the chances that your pull request will be reviewed, you should

- **Reference an open issue** for non-trivial changes to clarify the purpose of the PR.
- **Make sure you have appropriate tests**.
- **Keep your PR requests as simple as possible**. Large PRs take longer to review.
- **Make sure the CI is in a green state**. Reviewers may not look otherwise
- **Keep** `Update your pull request`_, either per request or every few days.
