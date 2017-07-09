#!/usr/bin/env bash
while read requirement; do conda install --yes $requirement; done < requirements.txt