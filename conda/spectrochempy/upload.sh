#!/usr/bin/env bash

anaconda upload --user spectrocat ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2 --force

conda convert --platform linux-64 ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2
anaconda upload --user spectrocat linux-64/spectrochempy-$1.tar.bz2 --force

conda convert --platform win-64 ~/opt/anaconda3/envs/scpy-dev/conda-bld/osx-64/spectrochempy-$1.tar.bz2
anaconda upload --user spectrocat win-64/spectrochempy-$1.tar.bz2 --force
