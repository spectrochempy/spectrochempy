# AGENTS.md - SpectroChemPy Development Guide

This guide helps agentic coding agents work effectively with the SpectroChemPy codebase.

## Build & Development Commands

### Installation & Setup
```bash
# Install with development dependencies
python -m pip install ".[dev]"

# Install with test dependencies  
python -m pip install ".[test]"

# Install all optional dependencies
python -m pip install ".[interactive,test,docs]"
```

### Testing Commands
```bash
# Run all tests with coverage
coverage run --source=spectrochempy -m pytest tests -s --durations=10

# Run single test file
pytest tests/test_core/test_dataset/test_dataset.py -v

# Run specific test function
pytest tests/test_core/test_dataset/test_dataset.py::test_nddataset_docstring -v

# Run tests with markers
pytest tests -m "not slow"  # Skip slow tests
```

### Code Quality
```bash
# Lint and fix with ruff
ruff check --fix src/
ruff format src/

# Pre-commit hooks (automatically run on commit)
pre-commit run --all-files
```

## Code Style & Formatting

### Ruff Configuration
- **Line length**: 88 characters
- **Target Python**: 3.12
- **Import style**: Single-line imports forced (`force-single-line = true`)
- **Quote style**: Double quotes
- **Indententation**: Spaces

### Key Ruff Rules Enabled
- `D` - pydocstyle (docstring conventions)
- `E` - pycodestyle errors
- `F` - pyflakes
- `I` - isort (import sorting)
- `N` - pep8-naming
- `UP` - pyupgrade
- `S` - bandit (security)
- `B` - flake8-bugbear
- `RET` - flake8-return

### Import Patterns & Organization
```python
# Standard library imports first
import os
import sys
from datetime import datetime

# Third-party imports
import numpy as np
import traitlets as tr

# Local imports
from spectrochempy.core.units import ur
from spectrochempy.utils.exceptions import SpectroChemPyError
```

**Lazy Loading System:**
- Uses `lazy_loader` for performance optimization
- Main API accessible via `spectrochempy.` without explicit imports
- Lazy imports defined in `src/spectrochempy/lazyimport/api_methods.py`

## Type Annotations & Documentation

### Type Hints
- Extensive use of type annotations in modern code
- Support for Python 3.10+ type features
- Custom type utilities in `spectrochempy.utils.typeutils`

### Documentation Standards
- **Docstring Style**: NumPy-style docstrings
- **Validation**: `numpydoc_validation` enabled with specific checks
- **Format**: Comprehensive parameter descriptions, examples, and see also sections
- **Tools**: Uses `docrep` for docstring reuse in some cases

## Naming Conventions

### Classes
- CapWords convention (e.g., `NDDataset`, `Coord`, `Project`)
- Some exceptions allowed (`N801` ignored)

### Functions & Methods
- snake_case (e.g., `read_csv`, `set_coordinates`)
- Some camelCase allowed (`N802` ignored)

### Variables
- lowercase with underscores (e.g., `ref_data`, `coord_set`)
- Mixed case allowed in some contexts (`N806` ignored)

### Constants
- UPPER_CASE (e.g., `DEFAULT_DIM_NAME`, `MASKED`)

## Error Handling Patterns

### Custom Exception Hierarchy
```python
# Base exception
class SpectroChemPyError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)

# Specific exceptions
class UnitsCompatibilityError(SpectroChemPyError): ...
class NotFittedError(SpectroChemPyError): ...
class ShapeError(SpectroChemPyError): ...
```

### Warning System
- Custom warning subclasses (e.g., `UnitErrorWarning`, `KeyErrorWarning`)
- Context managers for exception handling (`ignored()` function)

## Testing Patterns

### Test Structure
- **Framework**: pytest with extensive fixtures
- **Location**: `tests/` directory mirroring source structure
- **Fixtures**: Comprehensive fixtures in `conftest.py` for test data
- **Markers**: `@pytest.mark.slow` for time-intensive tests

### Test Utilities
```python
from spectrochempy.utils.testing import (
    assert_array_almost_equal,
    assert_dataset_equal,
    RandomSeedContext,
    set_env,
)
```

### Test Data Management
- Automatic download of test data in `conftest.py`
- Cleanup of temporary files after test runs
- Mock fixtures for filesystem operations

## Core Data Structures

### NDDataset
- Main data container with labeled axes and metadata
- Supports units, masking, and coordinates
- Mathematical operations with unit awareness
- Integration with numpy array interface

### Coord & CoordSet
- Coordinate systems for datasets
- Support for labeled and numerical coordinates
- Unit-aware coordinate operations

## Development Workflow

### Pre-commit Hooks
- Automatic requirement regeneration
- Ruff linting and formatting
- YAML validation
- Trailing whitespace cleanup

### CI/CD Pipeline
- Multi-platform testing (Linux, macOS, Windows)
- Python version matrix (3.11, 3.14)
- Coverage reporting with Codecov
- Automated package building and publishing

## Special Considerations for Agents

1. **Lazy Loading**: Be aware that some imports may not be available until accessed
2. **Plugin System**: The codebase supports optional plugins (e.g., quaternion support)
3. **Documentation Generation**: API files are auto-generated - do not edit manually
4. **Test Data**: Tests require downloading external data files
5. **Units Integration**: Heavy use of Pint for unit handling - always consider unit compatibility
6. **Matplotlib Integration**: Plotting functionality with custom matplotlib setup
7. **Performance**: Lazy loading used for startup performance optimization

## Key Files to Understand

- `src/spectrochempy/lazyimport/api_methods.py` - Lazy import definitions
- `tests/conftest.py` - Test fixtures and setup
- `src/spectrochempy/utils/exceptions.py` - Exception hierarchy
- `src/spectrochempy/core/dataset/nddataset.py` - Main data container
- `src/spectrochempy/core/plotters/_mpl_setup.py` - Matplotlib configuration

