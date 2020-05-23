#!/usr/bin/env bash

## adapted on https://gist.github.com/zshaheen/fe76d1507839ed6fbfbccef6b9c13ed9

PKG_NAME=spectrochempy
USER=spectrocat
OS=noarch ## OS=$TRAVIS_OS_NAME-64
conda config --set anaconda_upload no

## TAG
TAG=$(git describe --tags)
IFS="\-"
read -a arr <<< "$TAG"
export LAST_TAG="${arr[0]}"
IFS="\."
read -a tag <<< "$LAST_TAG"
export NEXT_TAG="${tag[0]}.${tag[1]}.`expr ${tag[2]} + 1`"
export NUMBER="${arr[1]}"

export CONDA_BLD_PATH="~/conda-bld"
mkdir -p "$CONDA_BLD_PATH"

export CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "master" ]
then
  if [ ! $NUMBER ]
  then
     NUMBER="0"
  fi
  dev="dev"
  export DEVSTRING="$dev$NUMBER"
  export VERSION="$NEXT_TAG"
  echo "***************************************************************************************************************"
  echo "--> BUILDING $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
  echo "***************************************************************************************************************"
  conda build .
  echo "***************************************************************************************************************"
  echo "--> UPLOADING $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2 to <dev> anaconda repository"
  echo "***************************************************************************************************************"
  anaconda upload --force -u $USER -l dev "$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
else
  export DEVSTRING="stable"
  export VERSION="$LAST_TAG"
  echo "***************************************************************************************************************"
  echo "--> BUILDING $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
  echo "***************************************************************************************************************"
  conda build .
  echo "***************************************************************************************************************"
  echo "--> UPLOADING $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2 to <main> anaconda repository"
  echo "***************************************************************************************************************"
  anaconda upload --force -u $USER "$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
fi

exit $?