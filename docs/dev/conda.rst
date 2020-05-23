Conda
=======



Creating a local conda channel
-------------------------------

it may be interesting to have a local conda channel, for instance to check building and uploading conda recipe,
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
This folder contains two files: `meta.yaml`which contain the metadata of the recipe, and `conda_upload.sh` which is a ``bash``
script to automatically build the recipe and uploading it to anaconda.org in the ``spectrocat`` channel, a personal channel
or in the local ``conda_bld`` channel.

.. note::

    For the moment, there is no script allowing to build using Windows ``cmd`` script. This still must be done if necessary.
    However it is alway possible to use the conventional method susing ``conda-build`` directly.








.. |scpy| replace:: **SpectroChemPy**
