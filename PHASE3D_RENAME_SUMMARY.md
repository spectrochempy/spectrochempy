# Phase 3D Rename Summary

## Overview

This document describes the rename of the `spectrochempy.plot` module to `spectrochempy.plotting`.

## What Was Renamed

### Directory
- `src/spectrochempy/plot/` → `src/spectrochempy/plotting/`

### Internal Modules
All plotting modules were moved from the `plot` directory to the `plotting` directory:
- `plot1d.py` - 1D plotting functions
- `plot2d.py` - 2D plotting functions  
- `plot3d.py` - 3D plotting functions
- `multiplot.py` - Multi-panel plotting
- `plot_setup.py` - Matplotlib lazy initialization
- `dispatcher.py` - Main plotting dispatcher
- `backends/` - Backend implementations

### Lazy Import Mappings
Updated `lazyimport/api_methods.py` and `lazyimport/dataset_methods.py` to reference `spectrochempy.plotting.*` instead of `spectrochempy.plot.*`.

### Internal Imports
Updated all internal imports throughout the codebase:
- `src/spectrochempy/core/`
- `src/spectrochempy/plotting/`
- `src/spectrochempy/core/plotters/` (deprecated re-export wrappers)

## Backward Compatibility Policy

A compatibility shim was created at `src/spectrochempy/plot/__init__.py` that:
- Imports all symbols from `spectrochempy.plotting`
- Emits a `DeprecationWarning` when the module is imported

### What Still Works
```python
# Old import path still works (with deprecation warning)
from spectrochempy.plot import plot1d, plot2d, plot3d
from spectrochempy.plot import plot_pen, plot_map

# Public API still works
import spectrochempy as scp
scp.plot(dataset)  # Works!
dataset.plot()      # Works!

# Direct imports from new location
from spectrochempy.plotting.plot1d import plot_pen
```

### What Changed
- The canonical location is now `spectrochempy.plotting`
- Internal implementation uses `spectrochempy.plotting` exclusively

## Planned Removal Timeline

The backward compatibility shim will be removed in a **future major release** (e.g., v1.0.0). 

Users should migrate to:
- `spectrochempy.plotting` instead of `spectrochempy.plot`
- `scp.plot()` function remains unchanged
- `dataset.plot()` method remains unchanged

## Key Implementation Details

### Module/Function Collision Handling
The `plot` function is exposed via lazy loading from `spectrochempy.plotting.dispatcher`. The dispatcher module exports:
- `plot_dataset` - main plotting function
- `plot` - alias for `plot_dataset`

This ensures `scp.plot(dataset)` works correctly without being shadowed by the `spectrochempy.plot` module.

### Lazy Loading
- `import spectrochempy` - NO matplotlib loaded
- `from spectrochempy import NDDataset` - NO matplotlib loaded
- `NDDataset([1,2,3])` - NO matplotlib loaded
- `ds.plot()` or `scp.plot(ds)` - matplotlib IS loaded

### Tests
All tests pass, including:
- `test_lazy_initialization.py` - 13 tests
- `test_docstrings_plaintext.py` - 7 tests
- `test_dataset.py` - 77 tests

## Files Modified

### Core Changes
- `src/spectrochempy/plot/` → renamed to `src/spectrochempy/plotting/`
- `src/spectrochempy/plot/__init__.py` - NEW backward compatibility shim
- `src/spectrochempy/core/dataset/nddataset.py` - updated imports and docstrings
- `src/spectrochempy/core/dataset/arraymixins/ndplot.py` - updated error message

### Lazy Import Changes
- `src/spectrochempy/lazyimport/api_methods.py` - updated mappings
- `src/spectrochempy/lazyimport/dataset_methods.py` - updated mappings

### Test Changes
- `tests/test_core/test_plotters/test_lazy_initialization.py` - updated imports
- `tests/test_core/test_plotters/test_docstrings_plaintext.py` - updated imports
