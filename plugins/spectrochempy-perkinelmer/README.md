# spectrochempy-perkinelmer

Optional SpectroChemPy plugin providing a reader for PerkinElmer `.sp` binary files.

## Installation

The next release, 0.1.6, requires SpectroChemPy 1.1.0 or later (and remains
restricted to SpectroChemPy versions below 2). Users who stay on
SpectroChemPy 1.0.0 should keep an earlier compatible version of this plugin.


```bash
pip install plugins/spectrochempy-perkinelmer
```

## Usage

```python
import spectrochempy as scp

# Preferred short form
ds = scp.perkinelmer.read("path/to/file.sp")

# Alternative forms
ds = scp.read_perkinelmer("path/to/file.sp")
ds = scp.read_sp("path/to/file.sp")
```

## Limitations

- Only single-spectrum `.sp` files are supported.
- The `.prf` format is not supported.
- Metadata extraction depends on the presence of standard PerkinElmer blocks; files with incomplete metadata will still load but with reduced `meta` information.
