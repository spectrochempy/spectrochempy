#!/usr/bin/env bash

## adapted on https://gist.github.com/zshaheen/fe76d1507839ed6fbfbccef6b9c13ed9

## Show command and exit immediately if a command exits with a non-zero status.
set -ex

## Settings (we build essentially a noarch package
PKG_NAME=spectrochempy
OS=noarch

## get version string from setuptools_scm
PVS="$(python setup.py --version)"
echo "Current version string = $PVS"

## Extract components
IFS=$"+"
read -ra arr <<< "$PVS"

## latest version string
LATEST="${arr[0]}"
IFS=$"."
read -ra tag <<< "$LATEST";
DEVSTRING="${tag[3]}"
VERSION="${tag[0]}.${tag[1]}.${tag[2]}"
if [[ -z $DEVSTRING ]]; then
  DEVSTRING="stable"
fi
PKG_NAME_VERSION="$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"

## Avoid uploading automatically
conda config --set anaconda_upload no

export VERSION=$VERSION
export DEVSTRING=$DEVSTRING

export CONDA_BLD_PATH="$HOME/conda-bld"
mkdir -p "$CONDA_BLD_PATH"

## configure conda
conda config -q --set always_yes yes --set changeps1 no
conda update -q -n base conda
conda config -q --add channels conda-forge
conda config -q --add channels cantera
conda config -q --add channels spectrocat
conda config -q --set channel_priority flexible

PKG_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME_VERSION"
echo "---> Building $PKG_FILE"
conda build conda

echo "---> Uploading $PKG_FILE"
if [[ $TRAVIS_BRANCH == "master" ]]; then
  ## We build the current master release (i.e.the latest development version)
  ## This is a "dev" release
  if [[ $TRAVIS_PULL_REQUEST == false ]]; then
    anaconda -t "$CONDA_UPLOAD_TOKEN" upload --force -u $ANACONDA_USER -l dev "$PKG_FILE";
  fi;
elif [[ $TRAVIS_BRANCH == $TRAVIS_TAG ]]; then
  ## This is a "main" release
  if [[ $TRAVIS_PULL_REQUEST == false ]]; then
    anaconda -t "$CONDA_UPLOAD_TOKEN" upload --force -u $ANACONDA_USER "$PKG_FILE";
  fi;
fi