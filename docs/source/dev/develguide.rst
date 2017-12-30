.. _develguide:

Developer's Guide
==================

Installing a developper version
--------------------------------

The best to proceed with development is that we (the developers) all have a similar
environment for each python version.

The master is build on the 3.6 python version. It may work with earlier
version of python, *e.g.*, 3.x but this has not yet been tested.

For sure, it will not work for python 2.7.x and no attempt to get such
compatibility will be made.

1.  **Install** ***anaconda***
    Go to `Anaconda donwload website <https://www.anaconda.com/download/>`_ the
    and choose your platform. Download one of the available graphical
    installer, *e.g.*, the 3.6 or + version.

2.  Install the version of Anaconda which you downloaded, following the
    instructions on the download page.

3.  Open a terminal and create an environment called, for example, **scp36**
    by entering the following command:

    .. sourcecode:: bash

	    $ conda create -n scp36 python=3.6


4.  Switch to this environment:

    On WINDOWS, you should use ``activate scp36``.

    On LINUX or macOS,  ``source activate scp36``.

    .. sourcecode:: bash

	    $ source activate scp36


    You can make it permanent by putting this command in you ``bash_profile``
    (MAC).

    .. todo::

        what's the equivalent for windows???

4.  If during set up or runtime, some packages with name <pkgname> appear to
    be missing, just install them using:

    .. sourcecode:: bash

	    $ conda install -n scp36 <pkgname>


5.  Clone the **spectrochempy** bitbucket repository

    .. sourcecode:: bash

	$ cd <workspace>

where <workspace> is you programming workspace (any folder you like)

.. sourcecode:: bash

	$ git clone git@bitbucket.org:spectrocat/spectrochempy.git


6. Switch to the ``spectrchempy`` directory

.. sourcecode:: bash

	$ cd <wokspace>/spectrochempy


7. Get the packages necessary for running |scpy| of all required package:

.. sourcecode:: bash

    $ conda install -y -\-file requirements.txt

6. Install the spectrochempy package

Execute the setup.py in developper mode

.. sourcecode:: bash

	$ python setup.py develop


or use the pip command in developper mode (flag `-e`)

.. sourcecode:: bash

	$ pip install -e .



Testing SpectroChemPy
---------------------

Tests for SpectroChemPy are executed using pytest.
It should then be present on the system.

.. sourcecode:: bash

	$ conda install pytest

In order to accelerate the tests, it is useful to install the plugin
``pytest-xdist`` for parallelization of the tests.

.. sourcecode:: bash

	$ conda install pytest-xdist

To run the full suite of tests or only some of them, the best way is to do this using py charm.

However it is possible to execute also the full suite of test, using the following command
from inside the main spectrochempy directory (where the folder ``tests`` resides.

.. sourcecode:: bash

	$ cd <workspace>/spectrochempy
	$ pytest tests

Currently it is not possible to use arguments in this command line, as they
will be interpreted by spectrochempy and then produce errors.
To add arguments/options to pytest, use the ``pystest.ini`` file in the ``tests`` folder.


Compiling the docs
-------------------

To build the doc, we need the following packages:

* sphinx
* nbsphinx, to convert notebook to sphinx pages
* sphinx-gallery, to convert python \*.py files to examples for the gallery.
* sphinx-nbexamples, to convert \*.ipynb notebooks into example for the gallery

These package are available on conda-forge or pypi.

Assuming you are in the main spectrochempy directory,
to rebuild the doc, just do:

.. sourcecode:: bash

    $cd docs
    $python builddocs.py clean html

or to update it after some changes:

.. sourcecode:: bash

    $cd docs
    $python builddocs.py html

The generated file are located in a directory (spectrochempy_doc) at the same
level as the
spectrochempy directory.

To display the documentation (on mac. For widow the command `start` should
work or something equivalent on linux):

    $cd ../../spectrochempy_doc/html
    $open index.html

you can also double-click on the index.html file in your file explorer (may
be simpler!).


Commit and push to the Bitbucket repository
--------------------------------------------

to do
