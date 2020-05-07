    Conda package for spectrochempy.
================================

.condarc
--------

the content in the `.condarc` configuration 
file in you $HOME directory should be something like this before starting:

    channels:
      - conda-forge
      - spectrocat
      - cantera
      - defaults
    
    ssl_verify: true
    anaconda_upload: false
    channel_priority: strict
    
Generate the yaml configuration file
------------------------------------

the meta.yaml was first generated from pypi (we do not need to redo it now because it has been modified since this time)

On windows, install git and patch which are required:

    conda install git m2-patch # not necessary on Mac.
       
    conda activate scpy-dev 
    cd ./conda
    conda skeleton pypi spectrochempy

then the recipes was built both for Windows (win64) and OSX (osx64)

    conda update conda-build
    conda update conda-verify
    conda config --set anaconda_upload no
    conda build spectrochempy
    conda build purge
 
You can check where the builded file was placed:

    (scpy-dev) W:\SCP\spectrochempy\conda>conda build spectrochempy --output
                                              
    --> C:\Users\christian\anaconda3\conda-bld\win-64\spectrochempy-0.1.16-py37_0.tar.bz2

    
Uploading to the spectrocat channel on anaconda.org
----------------------------------------------------

Channel info can be found here: https://anaconda.org/spectrocat/repo

We need additional packages:

    conda install anaconda-client

then after login on anaconda
    
    anaconda login
    
you upload the files:

    anaconda upload --user spectrocat ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2 --force

Now spectrochempy can be obtained for the spectrocat channel:

    conda deactivate
    conda update conda # IMPORTANT especially if you get errors in the installation step 
    
if not yet done 

    conda config --add channels conda-forge
    conda config --add channels cantera 
    conda config --add channels spectrocat
    
finally

    conda create -n <your-environment-name>  python=3.7
    conda activate <your-environment-name>
    conda install spectrochempy
    
    
