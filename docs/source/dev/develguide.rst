.. _develguide:

Developer Guide
###############

Installing a developper version
===============================

The best to proceed with development is that we (the developers) all have a similar
environment for each python version.

The master is build on the 3.6 python version. 

1. Install anaconda3 or miniconda3

* OSX : `miniconda3 <https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh>`_

2. create an environment called **scp36**

.. sourcecode:: bash

	$ conda create -n scp36 python=3.6 -\-file requirements.txt


3. switch to this environment

.. sourcecode:: bash

	$ source activate scp36


You can make it permanent by putting this command in you bash_profile (MAC)


4. If during set up or runtime, some package appear to miss, just install them using:

.. sourcecode:: bash

	$ conda install -n scp36 <pkgname>


5. Clone the **spectrochempy** bitbucket repository

.. sourcecode:: bash

	$ cd <workspace>

where <workspace> is you programming workspace (any folder you like)

.. sourcecode:: bash

	$ git clone git@bitbucket.org:spectrocat/spectrochempy.git


6. Install the spectrochempy package

switch to the installation directory

.. sourcecode:: bash

	$ cd <wokspace>/spectrochempy


Execute the setup.py in developper mode

.. sourcecode:: bash

	$ python setup.py develop


or use the pip command in developper mode (flag `-e`)

.. sourcecode:: bash

	$ pip install -e .



Setup PyCharm
--------------



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

	$ cd <wokspace>/spectrochempy
	$ pytest tests

Currently it is not possible to use arguments in this command line, as they
will be interpreted by spectrochempy and then produce errors.
To add arguments/options to pytest, use the ``pystest.ini`` file in the ``tests`` folder.


Commit and push to the Bitbucket repository
--------------------------------------------


