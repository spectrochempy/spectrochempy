#!/usr/bin/env bash

conda convert --platform linux-64 /Users/christian/anaconda3/conda-bld/osx-64
/spectrochempy-$1.tar.bz2

anaconda upload --user spectrocat linux-64/spectrochempy-$1.tar.bz2 --force

conda convert --platform linux-32 /Users/christian/anaconda3/conda-bld/osx-64
/spectrochempy-$1.tar.bz2

anaconda upload --user spectrocat linux-32/spectrochempy-$1.tar.bz2 --force

conda convert --platform win-64 /Users/christian/anaconda3/conda-bld/osx-64
/spectrochempy-$1.tar.bz2

anaconda upload --user spectrocat win-64/spectrochempy-$1.tar.bz2 --force

conda convert --platform win-32 /Users/christian/anaconda3/conda-bld/osx-64
/spectrochempy-$1.tar.bz2

anaconda upload --user spectrocat win-32/spectrochempy-$1.tar.bz2 --force
