#!/usr/bin/env python3

# https://stackoverflow.com/a/47743010

# Imports

import re
import shutil
from pathlib import Path
from matplotlib import matplotlib_fname
from matplotlib import get_cachedir

# Copy files over
_dir_data = Path(re.sub('/matplotlibrc$', '', matplotlib_fname()))

dir_source = Path(__file__).parent.parent.parent / 'scp_data' / 'fonts'
if not dir_source.exists():
    raise IOError(f'directory {dir_source} not found!')

dir_dest = _dir_data / 'fonts' / 'ttf'
if not dir_dest.exists():
    dir_dest.mkdir(parents=True)
# print(f'Transfering .ttf and .otf files from {dir_source} to {dir_dest}.')
for file in dir_source.glob('*.[ot]tf'):
    if not (dir_dest / file.name).exists():
        print(f'Adding font "{file.name}".')
        shutil.copy(file, dir_dest)

# Delete cache
dir_cache = Path(get_cachedir())
for file in list(dir_cache.glob('*.cache')) + list(dir_cache.glob('font*')):
    if not file.is_dir():  # don't dump the tex.cache folder... because dunno why
        file.unlink()
        print(f'Deleted font cache {file}.')
