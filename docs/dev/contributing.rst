.. _develguide:

Contributing to |scpy| 
#######################


.. contents:: Table of Contents
   :local:


How to help ?
=============

Every  user of |scpy| can make useful contributions.

There is several way depending of your knowledge of programming:

* reporting bugs
* request for enhancement or new features
* contributing to the documentation
* making pull request to the |scpy| repository
* ...


.. note::

  Parts of this document come from the `Contributing to xarray
  guide <http://http://xarray.pydata.org/en/stable/contributing.html>`_.


.. _contributing.bug_reports:

Bug reports and enhancement requests
====================================

Bug reports are an important part of making |scpy| more stable.

Before creating a new issue, it is worth searching for existing bug reports and
pull requests to see if the issue has already been reported and/or fixed.

Bug reports must:

#. Include a short, self-contained Python snippet reproducing the problem.
   You can format the code nicely by using `GitHub Flavored Markdown
   <http://github.github.com/github-flavored-markdown/>`_::

      ```python
      >>> from spectrochempy import *
      >>> nd = NDDataset(...)
      ...
      ```

#. Include the full version string of |scpy| and its dependencies. You can use the
   built in function::

      >>> import xarray as xr
      >>> xr.show_versions()

#. Explain why the current behavior is wrong/not desired and what you expect instead.

The issue will then show up to the *xarray* community and be open to comments/ideas
from others.


Contributing to the code
=========================

Installing a developper version
********************************

The best to proceed with development is that the developers have a similar
python environment.

The official master repository has been tested on the 3.6 or 3.7 python version.
It may work with earlier
version of python, *e.g.*, <3.6 but this has not yet been tested.

For sure, it will not work for python 2.7.x and no attempt to get such
compatibility will be made.

Install Anaconda
----------------

#.  To install Anaconda (or Miniconda)

    Go to `Anaconda download website <https://www.anaconda.com/download/>`_ the
    and choose your platform. Download one of the available graphical
    installer, *e.g.*, the 3.6 or + version.


#.  Install the version of Anaconda which you just downloaded, following
    the instructions on the download page.

.. _clonescpy:

Install a developpement version of SpectroChemPy
------------------------------------------------

#.  Git clone the |scpy| `Bitbucket repository <https://bitbucket.org/spectrocat/spectrochempy/src/master/>`_

    .. sourcecode:: bash

       $ git clone git@bitbucket.org:spectrocat/spectrochempy.git <workspace>/spectrochempy
        
    where `<workspace>` is you programming workspace directory. 
    
    .. note::

       if you want to contribute and push your change to the Bitbucket repository,
       you will need to modify this step. Go fist to :ref:`forkscpy` and then come back here.


#.  Switch to the ``spectrochempy`` directory

    .. sourcecode:: bash

       $ cd <workspace>/spectrochempy


#.  Create a `conda` environment called, for example, **scpy**
    by entering the following commands:

    .. sourcecode:: bash

       $ conda env create -f=env/scpy-dev.yml

    This will add all (or most) of the necessary packages for development.

#.  Switch to this environment:

    .. sourcecode:: bash

        $ conda activate scpy-dev

    You can make it permanent by putting this command in your ``bash_profile``
    (MAC), ``.bashrc`` (LINUX) or using the following batch file (WIN)

    .. sourcecode:: bat

        @REM launch a cmd window in scpy-dev environment (path should be adapted)
        @CALL CD C:\your\favorite\folder
        @CALL CMD /K C:\your\anaconda\folder\Scripts\activate.bat scpy-dev

#. 	Install the spectrochempy package

    Execute the `setup.py` in developper mode

    .. sourcecode:: bash

       $ python setup.py develop

    or use the pip command in developper mode (flag `-e`)

    .. sourcecode:: bash

       $ pip install -e .

#.  If during set up or runtime, some packages with name <pkgname> appear to
    be missing, just install them using

    .. sourcecode:: bash

       $ conda install -n scpy <pkgname>

    ```n scpy`` is just to be sure we install in the correct environment.

.. _forkscpy:

Create a SpectroChemPy fork repository
------------------------------------------------------

The problem with the above procedure is that you can commit change
made to the application locally, but you won't be able to push any changes to the
``origin`` repository if the maintainer do not give `write` access to it.

To be able to contribute to |scpy|, you will need to create you own **fork** of the
|scpy| repository based on `Bitbucket <https://bitbucket.org/>`. And then from your fork, you can
create pull request to the main repository.

The workflow is the following:

* Create a fork on Bitbucket.
* Clone the forked repository to your local system.
* Modify the local repository.
* Commit your changes.
* Push your changes back to the remote fork on Bitbucket.
* Create a pull request from the forked repository (source) back to the original (destination).

The final step in the workflow is for the maintener of the original repository to merge your changes.

The simplest way is to perform this operation on the `bitbucket.org <https://bitbucket.org/>`_ web site.

* Create an account (if not yet done) or sign in:

  .. image:: images/signin.jpg
     :width: 500 px
     :alt: Sign in on Bitbucket
     :align: center


* Go to the |scpy| repository
  `<https://bitbucket.org/spectrocat/spectrochempy>`_. You should see something like this:

  .. image:: images/scpy_repo.png
     :width: 500 px
     :alt: Spectrochempy repository
     :align: center


* click ``+`` in the sidebar and select `Fork` this repository under `Get to work`.

  .. image:: images/forkit.png
     :width: 500 px
     :alt: Fork
     :align: center


  The system displays the Fork dialog.

  .. image:: images/forkit2.png
     :width: 500 px
     :alt: Fork dialog
     :align: center


* Now you can proceed with the previous installation steps :ref:`clonescpy`. The only change is the
  git command to clone your own |scpy| Bitbucket repository, instead of the official ones.

  .. sourcecode:: bash

     $ git clone git@bitbucket.org:<username>/spectrochempy.git <workspace>/spectrochempy

  where `<username>` is your bitbucket account user name and `<workspace>` is you programming workspace directory.


* After you fork a repository, the original repository is likely to evolve as other users commit changes to it.
  These changes do not appear in your fork automatically. To find out if your fork is missing commits,
  at the bottom of the Repository details card of your fork, you'll see a button with `Sync (# commits behind)`.
  Click this button to pull these commits into your fork.

  .. image:: images/details.png
     :width: 300 px
     :alt: Repository details
     :align: center


Testing SpectroChemPy
---------------------

Tests for SpectroChemPy are executed using
`pytest <https://docs.pytest.org/en/latest/>`_.
It should be present on the system, else install it:

.. sourcecode:: bash

   $ conda install pytest


To run the full suite of tests or only some of them, the best way is to use PyCharm.

However it is possible to execute also the full suite of test, using the following command
from inside the main spectrochempy directory (where the folder ``tests`` resides.

.. sourcecode:: bash

   $ cd <workspace>/spectrochempy/tests
   $ pytest .

Currently it is not possible to use arguments in this command line, as they
will be interpreted by spectrochempy and then produce errors.
To add arguments/options to pytest, use the ``pytest.ini`` file in the ``tests`` folder.


Compiling the docs
-------------------

To build the doc, we need the following packages:

* sphinx
* nbsphinx, to convert notebook to sphinx pages
* sphinx-gallery, to convert python \*.py files to examples for the gallery.
* sphinx-nbexamples, to convert \*.ipynb notebooks into example for the gallery

These packages are available on conda-forge or pypi. They should have been installed during the previous steps.

Assuming you are in the main spectrochempy directory,
to rebuild the doc, just do:

.. sourcecode:: bash

   $cd docs
   $python builddocs.py clean html

or to update it after some changes:

.. sourcecode:: bash

   $cd docs
   $python builddocs.py html

The generated file are located in a directory (spectrochempy_doc) at the same level as the spectrochempy directory.

To display the documentation (on mac. For window the command `start` should work or something equivalent on linux):

.. sourcecode:: bash

   $cd ../../spectrochempy_doc/html
   $open index.html

you can also double-click on the index.html file in your file explorer (may be simpler!).


Commit and push to the Bitbucket repository
--------------------------------------------

to do

