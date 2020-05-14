Conda package for brukeropusreader.
====================================

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
    cd ./recipe
    conda skeleton pypi brukeropusreader

then the recipes was built both for Windows (win64) and OSX (osx64)

    conda update conda-build
    conda config --set anaconda_upload no
    conda build brukeropusreader
    conda build purge
 
You can check where the builded file was placed:

    (scpy-dev) W:\SCP\spectrochempy\conda>conda build brukeropusreader --output
    
    --> C:\Users\christian\anaconda3\conda-bld\win-64\brukeropusreader-0.1.7-py37_0.tar.bz2

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
and then install brukeropusreader from the local channel:

    pip remove brukeropusreader
    conda install --use-local brukeropusreader
    
Uploading to the spectrocat channel on anaconda.org
----------------------------------------------------

Channel info can be found here: https://anaconda.org/spectrocat/repo

We need additional packages:

    conda install anaconda-client

then after login on anaconda
    
    anaconda login
    
you upload the files:

    anaconda upload --user spectrocat C:\Users\christian\anaconda3\conda-bld\win-64\brukeropusreader-0.1.7-py37_0.tar.bz2    

or for mac:

    anaconda upload --user spectrocat /Users/christian/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/brukeropusreader-0.1.7-py37_0.tar.bz2

Other platform

    conda convert --platform all /Users/christian/opt/miniconda3/envs/scpy-dev/conda-bld/osx-64/brukeropusreader-1.3.4-py37_0.tar.bz2

Then uplaod of all versions

     anaconda upload --user spectrocat win-32/brukeropusreader-1.3.4-py37_0.tar.bz2 
     
by changing the version string of the file to upload



From now brukeropusreader can be obtained for the spectrocat channel:

    conda install -c spectrocat brukeropusreader
    
    
