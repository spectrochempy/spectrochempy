#!/usr/bin/env bash

## adapted on https://gist.github.com/zshaheen/fe76d1507839ed6fbfbccef6b9c13ed9

## Exit immediately if a command exits with a non-zero status.
set -e

## Show commands
set -x

## Settings (we build essentially a noarch package
PKG_NAME=spectrochempy
OS=noarch

## Avoid uploading automatically
conda config --set anaconda_upload no

## TAG
TAG=$(git describe --tags)
IFS=$"-"
read -ra arr <<< "$TAG"
LAST_TAG="${arr[0]}"
IFS=$"."
read -ra tag <<< "$LAST_TAG"
NEXT_TAG="${tag[0]}.${tag[1]}.`expr ${tag[2]} + 1`"
NUMBER="${arr[1]}"

export CONDA_BLD_PATH="$HOME/conda-bld"
mkdir -p "$CONDA_BLD_PATH"

## Here we will choose depending on the way this script is run
if [[ $USER != "travis" ]]; then
  ## if we are in local
  TRAVIS_BRANCH=$(git rev-parse --abbrev-ref HEAD)
  if [[ $TRAVIS_BRANCH == $LAST_TAG ]]; then
    TRAVIS_TAG=$LAST_TAG
  fi
  ## else this run by TravisCI (this are env variables)
fi

echo "user: $USER   current branch: $TRAVIS_BRANCH last_tag: $LAST_TAG next_tag: $NEXT_TAG"

if [[ $TRAVIS_BRANCH == "master" ]]; then
  ## we build the current master repository (i.e.the last development version)
  export DEVSTRING="latest"
  export VERSION="$NEXT_TAG"
  PACKAGE_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
  echo "***************************************************************************************************************"
  echo "--> BUILDING $PACKAGE_FILE"
  echo "***************************************************************************************************************"
  conda build conda
  echo "***************************************************************************************************************"
  echo "--> UPLOADING $PACKAGE_FILE to <dev> anaconda repository"
  echo "***************************************************************************************************************"
  anaconda -t "$CONDA_UPLOAD_TOKEN" upload -f -u $ANACONDA_USER -l dev "$PACKAGE_FILE"
  exit $?
fi

if [[ $TRAVIS_BRANCH == $TRAVIS_TAG ]]; then
  ## this is a "stable" release
  export DEVSTRING="stable"
  export VERSION="$LAST_TAG"
  PACKAGE_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
  echo "***************************************************************************************************************"
  echo "--> BUILDING $PACKAGE_FILE"
  echo "***************************************************************************************************************"
  conda build conda
  echo "***************************************************************************************************************"
  echo "--> UPLOADING $PACKAGE_FILE to <main> anaconda repository"
  echo "***************************************************************************************************************"
  anaconda -t "$CONDA_UPLOAD_TOKEN" upload --force -u $ANACONDA_USER  "$PACKAGE_FILE"
  exit $?
fi

if [ ! $NUMBER ]
then
   NUMBER="0"
fi

if [[ $TRAVIS_BRANCH == "develop" ]]; then
  ## we build the current develop repository (for testing purpose)
  export DEVSTRING="test$NUMBER"
  export VERSION="$NEXT_TAG"
  PACKAGE_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
  echo "***************************************************************************************************************"
  echo "--> BUILDING $PACKAGE_FILE"
  echo "***************************************************************************************************************"
  conda build conda
  echo "***************************************************************************************************************"
  echo "--> UPLOADING $PACKAGE_FILE to <test> anaconda repository"
  echo "***************************************************************************************************************"
  anaconda -t "$CONDA_UPLOAD_TOKEN" upload --force -u $ANACONDA_USER -l test "$PACKAGE_FILE"
  exit $?
fi

## this is a local "dev" release not yet merged with develop (will not be uploaded)
export DEVSTRING="test$NUMBER"
export VERSION="$NEXT_TAG"
PACKAGE_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
echo "***************************************************************************************************************"
echo "--> BUILDING $PACKAGE_FILE"
echo "***************************************************************************************************************"
conda build conda
exit $?