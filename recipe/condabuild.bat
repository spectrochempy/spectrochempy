
@echo off
if "%1"=="" goto :Usage
echo PREPARING CONDA %1 PACKAGE BUILDING
echo We assume `.condarc` is already configured, if not execute the following instructions:
echo ^> conda config --add channels conda-forge
echo ^> conda config --add channels cantera
echo ^> conda config --add channels spectrocat
echo ^> conda config --set channel_priority strict
echo ^> conda config --set anaconda_upload no
echo We assume also that all necessary package are already installed. If not, run:
echo ^> conda install pip setuptools wheel conda-build conda-verify anaconda-client -y
echo.
echo UPDATING PACKAGES ...
call conda update conda -y
call conda update pip setuptools wheel conda-build conda-verify anaconda-client -y
echo.
echo CONDA %1 PACKAGE IS BUILDING ...
cd recipe
call conda config --set anaconda_upload yes
call conda build %1        <--------        NOT WORKING !!!!    (it works when the command is run in the shell  ?????)
call conda build purge
call conda config --set anaconda_upload no
cd ..
exit /B 0

:Usage
echo No arguments given !
exit /B 1