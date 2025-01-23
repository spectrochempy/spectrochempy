#!/usr/bin/env bash

## Install setuptools_scm
pip install setuptools_scm

## get version string from setuptools_scm
PVS=$(python -c "from setuptools_scm import get_version; print(get_version())")
echo "Current version string = $PVS"

## Extract components
IFS=$"+"
read -ra arr <<< "$PVS"

## latest version string
LATEST="${arr[0]}"
IFS=$"."
read -ra tag <<< "$LATEST";
VERSION="${tag[0]}.${tag[1]}.${tag[2]}"

DEVSTRING="${tag[3]}"
if [[ -z $DEVSTRING ]]; then
  DEVSTRING="stable"
fi

echo "$LATEST"
echo "$VERSION"
echo "$DEVSTRING"

## Export variable in the github environment
echo "VERSIONSTRING=$LATEST" >> $GITHUB_ENV
echo "VERSION=$VERSION" >> $GITHUB_ENV
echo "DEVSTRING=$DEVSTRING" >> $GITHUB_ENV
