#!/bin/sh
set -eu
printf '%s\n' '[startup shell] id=scpy-rtd-installed-startup stage=before-python'
utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
python_command=$(command -v python)
printf '[startup shell] utc=%s\n' "$utc"
printf '[startup shell] python-command=%s\n' "$python_command"
printf '%s\n' '[startup shell] completed=0; following Python command not yet observed'
