#!/usr/bin/env bash

PKG_NAME=spectrochempy

## Exit immediately if a command exits with a non-zero status.
set -e

## get version string from setuptools_scm
PVS=$(python setup.py --version)
ECHO "Current version string = $PVS"

## Extract components
IFS=$"+"
read -ra arr <<< "$PVS"

## latest version string
LATEST="${arr[0]}"
ECHO "LATEST = $LATEST"

## find the stable tag
IFS=$"."
read -ra tag <<< "$LATEST"
STABLE="${tag[0]}.${tag[1]}.`expr ${tag[2]} - 1`"
ECHO "STABLE = $STABLE"

DEVSTRING="${tag[3]}"
ECHO "DEVSTRING = $DEVSTRING"

## Here we will choose depending on the way this script is run
if [[ $USER != "travis" ]]; then
  ## if we are in local
  TRAVIS_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  if [[ $TRAVIS_BRANCH == $LAST_TAG ]]; then
    TRAVIS_TAG="$LAST_TAG"
  fi
  ## else this is run by TravisCI (TRAVIS_BRANCH and TRAVIS_TAG are environment variables)
fi

if [[ $TRAVIS_BRANCH == "master" ]]; then
  ## we are in the current master branch
  ## i.e., this is the latest development version
  VERSION="$NEXT_TAG"
  DEVSTRING="dev$NUMBER"
  PKG_NAME_VERSION="$PKG_NAME-$VERSION-$DEVSTRING.tar.bz2"
elif [[ $TRAVIS_BRANCH == $TRAVIS_TAG ]]; then
  ## this is a "stable" release
  VERSION="$LAST_TAG"
  PKG_NAME_VERSION="$PKG_NAME-$VERSION.tar.bz2"
fi

echo "VERSION=$VERSION"
echo "DEVSTRING=$DEVSTRING"
echo "PKG_NAME_VERSION=$PKG_NAME_VERSION"
echo "TRAVIS_TAG=$TRAVIS_TAG"
echo "TRAVIS_BRANCH=$TRAVIS_BRANCH"