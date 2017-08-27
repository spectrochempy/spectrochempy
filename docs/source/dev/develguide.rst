.. _develguide:

Developer Guide
###############

Installing a developper version
===============================

The best to proceed with development is that we (the developers) all have a similar environment for each python version.

The master is build on the 3.6 python version. 

1. Install anaconda3 or miniconda3

	* OSX : `miniconda3 <https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh>`_


2. create an environment called **scp36**

	conda create -n scp36 python=3.6 -\-file requirements.txt


3. switch to this environment

    source activate scp36


You can make it permanent by putting this command in you bash_profile (MAC)


4. If during set up or runtime, some package appear to miss, just install them using:

	conda install -n scp36 <pkgname>

5. Clone the **spectrochempy** bitbucket repository

    cd <workspace>

    where <workspace> is you progr<mming workspace (any folder you like)

    git clone git@bitbucket.org:spectrocat/spectrochempy.git

6. Install the spectrochempy package

    switch to the installation directory

    cd <wokspace/spectrochempy>



Setup PyCharm
--------------



Commit and push to the Bitbucket repository
--------------------------------------------


