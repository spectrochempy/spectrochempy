Conda package for quadprog.
===========================

( for more information on these steps, look here:
[Building conda packages with conda skeleton](https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-pkgs-skeleton.html))

Build
-----
the meta.yaml was generated from pypi

It is safe to work in a development environment e.g. `scpy-dev`, 
and test in the user environment `scpy`:

On windows, install git and patch wich are required:

    conda install git m2-patch # not necessary on Mac.
       
    conda activate scpy-dev 
    cd ./conda
    conda skeleton pypi quadprog

then the recipes was built both for Windows (win64) and OSX (osx64)

    conda update conda-build
    conda config --set anaconda_upload no
    conda build quadprog
    conda build purge
 
You can check where the builded file was placed:

    (scpy-dev) W:\SCP\spectrochempy\conda>conda build quadprog --output
    
    --> C:\Users\christian\anaconda3\conda-bld\win-64\quadprog-0.1.7-py37_0.tar.bz2

The building was done on both windows (using parallel deskstop on mac) 
and osx platforms. 

**Note**: If you end up on a error telling that `cl.exe` doen't exist
you need to install Build Tools for Visual Studio separately from the Visual Studio downloads page.
If you already have Visual Studio, you can also install Desktop development with C++ from the Visual Studio Installer.

To build the recipe I had to excute conda not in a `Anaconda Prompt`, but is `x64 Native Tools Command Prompt for VS`.

**TODO**: Still need to do the build for linux

Test
----

    conda activate scpy
    
remove any previous installation from pypi 
and then install quadprog from the local channel:

    pip remove quadprog
    conda install --use-local quadprog
    
Uploading to the spectrocat channel on anaconda.org
----------------------------------------------------

Channel info can be found here: https://anaconda.org/spectrocat/repo

We need additional packages:

    conda install anaconda-client

then after login on anaconda
    
    anaconda login
    
you upload the files:

    anaconda upload --user spectrocat C:\Users\christian\anaconda3\conda-bld\win-64\quadprog-0.1.7-py37_0.tar.bz2    

or for mac:

    anaconda upload /Users/christian/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/quadprog-0.1.7-py37_0.tar.bz2

From now quadprog can be obtained for the spectrocat channel:

    conda install -c spectrocat quadprog
    
    
