"""
`brukeropus` is a Python package for interacting with Bruker's OPUS spectroscopy software. Currently, the package can
read OPUS data files and communicate/control OPUS software using the DDE communication protocol).

### Installation
`brukeropus` requires `python 3.6+` and `numpy`, but `matplotlib` is needed to run the plotting examples.  You can
install with pip:
```python
pip install brukeropus
```

### Namespace
`brukeropus` provides direct imports to the following:
```python
from spectrochempy.brukeropus import find_opus_files, read_opus, OPUSFile, Opus, parse_file_and_print
```
All other file functions or classes can be directly imported from the `brukeropus.file` or `brukeropus.control`
submodules, e.g.:
```python
from spectrochempy.extern.brukeropus.file import parse_header
```
It is recommended that you do **not** import from the fully qualified namespace, e.g.:
```python
from spectrochempy.extern.brukeropus.file.parse import parse_header
```
as that namespace is subject to change. Instead import directly from `brukeropus` or its first level submodules.

### Reading OPUS Files (Basic Usage)
`brukeropus` can read the proprietary binary files saved by Bruker's OPUS software.
```python
from spectrochempy.brukeropus import read_opus
from matplotlib import pyplot as plt

opus_file = read_opus('file.0')  # Returns an OPUSFile class

opus_file.print_parameters()  # Pretty prints all metadata in the file to the console

if 'a' in opus_file.data_keys:  # If absorbance spectra was extracted from file
    plt.plot(opus_file.a.x, opus_file.a.y)  # Plot absorbance spectra
    plt.title(opus_file.sfm + ' - ' + opus_file.snm)  # Sets plot title to Sample Form - Sample Name
    plt.show()  # Display plot
```
More detailed documentation on the file submodule can be found in `brukeropus.file`

"""

# ruff: noqa: F401
from spectrochempy.extern.brukeropus.file import OPUSFile
from spectrochempy.extern.brukeropus.file import find_opus_files
from spectrochempy.extern.brukeropus.file import parse_file_and_print
from spectrochempy.extern.brukeropus.file import read_opus
