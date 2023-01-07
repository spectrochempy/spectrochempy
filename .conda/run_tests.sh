#!/usr/bin/env bash

mamba env update -f environment.yml
python <<< "import spectrochempy as scp"
