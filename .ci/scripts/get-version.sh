#!/usr/bin/env bash

PKG_NAME=spectrochempy

## Exit immediately if a command exits with a non-zero status.
set -e

## get version string from setuptools_scm
PVS="$(python setup.py --version)"
echo "Current version string = $PVS"

## Extract components
IFS=$"+"
read -ra arr <<< "$PVS"

## latest version string
LATEST="${arr[0]}"

## find the stable tag
IFS=$"."
read -ra tag <<< "$LATEST";
DEVSTRING="${tag[3]}"
VERSION="${tag[0]}.${tag[1]}.${tag[2]}"
if [[ $DEVSTRING ]]; then
  PKG_NAME_VERSION="$PKG_NAME-$VERSION.$DEVSTRING.tar.bz2"
else
  PKG_NAME_VERSION="$PKG_NAME-$VERSION.tar.bz2"
fi

echo "VERSION = $VERSION"
echo "DEVSTRING = $DEVSTRING"
echo "PKG_NAME_VERSION=$PKG_NAME_VERSION"
