#!/usr/bin/env bash

echo 'environnement is:'  $CONDA_PREFIX

conda config --append channels conda-forge

conda install --yes --file requirements.txt
conda list
conda-develop .