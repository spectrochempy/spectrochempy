"""Core test fixtures - lightweight synthetic datasets for unit testing.

All fixtures are:
- deterministic (fixed random seed)
- small (tiny arrays)
- fast (no files, no downloads)
- self-contained (no external dependencies)
"""

import numpy as np
import pytest

from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndcomplex import NDComplexArray
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.testing import RandomSeedContext

# ===========================================================================
# Deterministic reference data (module-level constants)
# ===========================================================================

# Simple 1D data (10 elements)
_ref_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# Simple 2D data (5x5) 
with RandomSeedContext(42):
    _ref_2d = np.round(10.0 * np.random.random((5, 5)), 2)

# 3D data (3x4x2)
with RandomSeedContext(123):
    _ref_3d = np.round(10.0 * np.random.random((3, 4, 2)), 2)

# Deterministic mask
_ref_mask_1d = np.array([False, True, False, False, True, False, False, False, True, False])
_ref_mask_2d = _ref_2d < 3.0

# Coordinate data
_coord_x = Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)
_coord_y = Coord(
    data=np.linspace(0.0, 60.0, 100),
    units="s",
    title="time",
)
_coord_z = Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)
_coord_2d_x = Coord(np.linspace(0.0, 10.0, 5), title="x_coord")
_coord_2d_y = Coord(np.linspace(0.0, 5.0, 5), title="y_coord")


# ===========================================================================
# Fixtures: Simple NDArray objects
# ===========================================================================


@pytest.fixture(scope="function")
def ndarray_1d():
    """Simple 1D NDArray (10 elements, no units, no mask)."""
    return NDArray(_ref_1d.copy(), copy=True)


@pytest.fixture(scope="function")
def ndarray_2d():
    """Simple 2D NDArray (5x5, no units, no mask)."""
    return NDArray(_ref_2d.copy(), copy=True)


@pytest.fixture(scope="function")
def ndarray_1d_unit():
    """1D NDArray with units."""
    return NDArray(_ref_1d.copy(), units="m/s", copy=True)


@pytest.fixture(scope="function")
def ndarray_2d_mask():
    """2D NDArray with mask."""
    return NDArray(
        _ref_2d.copy(),
        mask=_ref_mask_2d.copy(),
        units="absorbance",
        copy=True,
    )


@pytest.fixture(scope="function")
def ndarray_complex():
    """Complex-valued NDArray."""
    return NDComplexArray(
        _ref_2d.copy().astype(np.complex128), units="m/s", copy=True
    )


# ===========================================================================
# Fixtures: Coord objects
# ===========================================================================


@pytest.fixture(scope="function")
def coord_x():
    """Coord with wavenumber axis (10 points, 4000-1000 cm^-1)."""
    return _coord_x.copy()


@pytest.fixture(scope="function")
def coord_y():
    """Coord with time axis (100 points, 0-60 s)."""
    return _coord_y.copy()


@pytest.fixture(scope="function")
def coord_z():
    """Coord with temperature axis (3 points, 200-300 K)."""
    return _coord_z.copy()


@pytest.fixture(scope="function")
def coord_2d_x():
    """Coord for 2D dataset x-axis (5 points)."""
    return _coord_2d_x.copy()


@pytest.fixture(scope="function")
def coord_2d_y():
    """Coord for 2D dataset y-axis (5 points)."""
    return _coord_2d_y.copy()


# ===========================================================================
# Fixtures: Synthetic NDDataset objects
# ===========================================================================


@pytest.fixture(scope="function")
def ndataset_1d():
    """Simple 1D dataset (10 elements, no coords, no units)."""
    return NDDataset(_ref_1d.copy()).copy()


@pytest.fixture(scope="function")
def ndataset_2d():
    """Simple 2D dataset (5x5, no coords, no units)."""
    return NDDataset(_ref_2d.copy()).copy()


@pytest.fixture(scope="function")
def ndataset_3d():
    """3D dataset (3x4x2) with coordinates."""
    return NDDataset(
        _ref_3d.copy(),
        coordset=[_coord_x.copy(), _coord_y.copy(), _coord_z.copy()],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_1d_unit():
    """1D dataset with units."""
    return NDDataset(_ref_1d.copy(), units="m").copy()


@pytest.fixture(scope="function")
def ndataset_2d_units():
    """2D dataset with units."""
    return NDDataset(_ref_2d.copy(), units="absorbance").copy()


@pytest.fixture(scope="function")
def ndataset_2d_masked():
    """2D dataset with mask and coordinates."""
    return NDDataset(
        _ref_2d.copy(),
        mask=_ref_mask_2d.copy(),
        coordset=[_coord_2d_y.copy(), _coord_2d_x.copy()],
        title="masked_data",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_1d_masked():
    """1D dataset with mask."""
    return NDDataset(
        _ref_1d.copy(),
        mask=_ref_mask_1d.copy(),
        title="masked_1d",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_complex():
    """Complex-valued dataset."""
    return NDDataset(
        _ref_1d.copy().astype(np.complex128) * (1.0 + 0.5j),
        title="complex_data",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_nan():
    """Dataset with NaN values."""
    data = _ref_2d.copy()
    data[0, 0] = np.nan
    data[2, 3] = np.nan
    return NDDataset(data, title="with_nan").copy()


@pytest.fixture(scope="function")
def ndataset_broadcast():
    """Small dataset suitable for broadcasting tests (2x3)."""
    return NDDataset(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).copy()


# ===========================================================================
# Fixtures: NDDataset pairs for alignment/compatibility tests
# ===========================================================================


@pytest.fixture(scope="function")
def ndataset_aligned_pair(coord_x):
    """Two 1D datasets with identical coordinates (for alignment tests)."""
    c = coord_x
    d1 = NDDataset(np.sin(np.linspace(0, np.pi, 10)), coordset=[c])
    d2 = NDDataset(np.cos(np.linspace(0, np.pi, 10)), coordset=[c.copy()])
    return d1, d2


@pytest.fixture(scope="function")
def ndataset_misaligned_pair():
    """Two 1D datasets with different coordinates (for mismatch tests)."""
    c1 = Coord(np.linspace(0.0, 10.0, 10))
    c2 = Coord(np.linspace(1.0, 11.0, 10))
    d1 = NDDataset(np.ones(10), coordset=[c1])
    d2 = NDDataset(np.ones(10), coordset=[c2])
    return d1, d2
