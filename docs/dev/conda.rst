Conda
=====

Creating a local conda channel
-------------------------------

it may be interesting to have also a local conda channel, for instance to check building and uploading conda recipe,
and for testing these new builds.

First we have to create the necessary file structure:

.. sourcecode:: bash

    $ mkdir -p ~/conda-bld
    $ cd ~/conda-bld
    $ mkdir -p linux-64 osx-64 win-64 noarch

To prepare this channel to be use as a local channels, we must index it:

.. sourcecode:: bash

    $ conda index .

Then it will be possible, after putting some packages (for instance by downloading them from `<anaconda.org>`_)
into these directories, to use it as a local channel.
After each manual addition, it is necessary to update the index as above.

Now we can search packages in this local channel:

.. sourcecode:: bash

    $ conda search -c ~/conda-bld <name_of_package>>

and/or install them:

.. sourcecode:: bash

    $ conda install -c ~/conda-bld <name_of_package>

We will now use it to build |scpy| locally.


Building |scpy| locally
---------------------------------

To understand how a conda package is built, one can have a look
`https://docs.conda.io/projects/conda-build <https://docs.conda.io/projects/conda-build/en/latest/concepts/recipe.html>`_

The recipe is for |scpy| is located in the subfolder  ``recipe``.
This folder contains two files: `meta.yaml` which contain the metadata of the recipe, and `conda_upload.sh` which is a ``bash``
script to automatically build the recipe and uploading it to anaconda.org in the ``spectrocat`` channel, a personal channel
or in the local ``conda_bld`` channel.

.. note::

    For the moment, there is no script allowing to build using Windows ``cmd`` script. This still must be done if necessary.
    However it is alway possible to use the conventional method susing ``conda-build`` directly.

Before being able to perform the folllowing information, it is recommanded to install or update the needed tools:

.. sourcecode:: bash

    $ conda update conda -n base
    $ conda install conda-build anaconda-client -y
    $ conda config --set anaconda_upload no

or replace ``install`` by ``update``.

To build locally, one has to run the following from the main |scpy| directory:

.. sourcecode:: bash

    $ cd recipe
    $ bash conda_upload.sh

The build will be performed and store locally in the conda_bld local channel.

If you have setup an anaconda.org account associated to your fork, you can also get the package pushed on this repository.

Log to the anaconda.org account:

.. sourcecode:: bash

    $ anaconda login
    Using Anaconda API: https://api.anaconda.org
    Username: <account_name>
    <account_name>'s Password: ****
    login successful

and then look at the end of the output after the previous ``conda_upload.sh``. A line such as the following one should be present
(if the build was successful).

.. sourcecode:: bash

    # Automatic uploading is disabled
    # If you want to upload package(s) to anaconda.org later, type:

    anaconda upload ~/conda-bld/noarch/spectrochempy-0.1.xx-devy.tar.bz2

You have just to adapt it to your need: for exemple to push in you repository with the label ``test`` :

.. sourcecode:: bash

    anaconda upload ~/conda-bld/noarch/spectrochempy-0.1.xx-devy.tar.bz2 --label test

.. note::

    if your are building from the git branch ``master`` or ``develop`` , the script ``conda_upload``
    will uplaod automatically with the correct label.

    * ``master`` is pushed with label 'dev' and a version name such as : `<next-release>-latest`
    * ``develop`` with label 'test' and a version name : `<next-release>-dev`
    * tag are pushed in the main release repository with a version name: `tag-stable`.

Automatic building using TravisCI
-----------------------------------

If one has a TravisCI account associated to his own fork, the building of |scpy| as described above is automatic, whenever
a commit is pushed on the remote fork.

The main github upstream repository 'spectrochempy` has also a TravisCI accound associated, and thus wenever a new commit
or a new tag is pushed to the 'master' branch, a new build start immediately and the package are eventually uploaded into the
`spectrocat anaconda.org channel <https://anaconda.org/spectrocat/spectrochempy channel>`_ :

  .. image:: images/anaconda.png
     :width: 800 px
     :alt: anaconda
     :align: center

The configuration to fire this automatic building is in the main directory: `.travis.yml` and is executed only
if testing is successful.


.. |scpy| replace:: **SpectroChemPy**
