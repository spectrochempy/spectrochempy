#!/bin/bash

coverage run --rcfile=tests/.coveragerc -m pytest tests
coverage report
coverage xml
