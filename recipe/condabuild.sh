#!/bin/sh

## execution right to set in the shell: chmod +x condabuild.sh
## todo: write a window version

if [ $1 ]
then
  echo  PREPARING CONDA $1 PACKAGE BUILDING ...
  echo please wait!
  echo
  echo We assume `.condarc` is already configured, if not execute the following instructions:
  echo \> conda config --add channels conda-forge
  echo \> conda config --add channels cantera
  echo \> conda config --add channels spectrocat
  echo \> conda config --set channel_priority strict
  echo \> conda config --set anaconda_upload no
  echo
  echo We assume also that all necessary package are already installed. If not, run:
  echo \> conda install pip setuptools wheel conda-build conda-verify anaconda-client -y
  echo
  echo UPDATING PACKAGES ...
  conda update conda -y
  conda update pip setuptools wheel conda-build conda-verify anaconda-client -y

  echo CONDA $1 PACKAGE IS BUILDING ...
  cd recipe
  conda config --set anaconda_upload yes
  conda build $1
  conda build purge
  conda config --set anaconda_upload no
  cd ..
  exit 0

else
  echo "Need the name of the library to build as a first argument"
  exit 1
fi

