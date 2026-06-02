# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

# Import plotting fixtures from reorganized test_plotting directory
pytest_plugins = ["tests.test_plotting.plotting_fixtures"]

# Force Agg backend BEFORE any other imports to prevent interactive backend initialization
import matplotlib

matplotlib.use("Agg", force=True)

import os
from pathlib import Path

import numpy as np
import pytest

import spectrochempy


def pytest_collection_modifyitems(items):
    """Keep docstring validation out of xdist workers."""
    for item in items:
        if "docstrings" in item.name:
            item.add_marker(pytest.mark.serial)


# ======================================================================================
# OPTIONAL DIAGNOSTIC STATE GUARDS
# ======================================================================================
# These fixtures detect state leaks (env vars, rcParams, preferences).
# They are DISABLED by default (set RUN_STATE_GUARDS=1 to enable).
# Actual test isolation is handled by isolate_matplotlib_state fixture below.

RUN_STATE_GUARDS = os.environ.get("SCP_TEST_GUARDS", "0") == "1"


@pytest.fixture(autouse=True)
def _matplotlib_state_guard(request):
    """Detect matplotlib state mutations (diagnostic only, no fixing)."""
    if not RUN_STATE_GUARDS:
        yield
        return

    import sys

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    test_name = request.node.name
    print(f"\n=== GUARD START: {test_name} ===", file=sys.stderr)

    backend_before = mpl.get_backend()
    rc_before = dict(mpl.rcParams)
    interactive_before = plt.isinteractive()
    fignums_before = list(plt.get_fignums())

    yield  # test runs here

    # Check for mutations (after test, before isolation fixture restores)
    backend_after = mpl.get_backend()
    interactive_after = plt.isinteractive()
    rc_after = dict(mpl.rcParams)
    fignums_after = list(plt.get_fignums())

    # Report only if changes detected
    leaked = fignums_after != fignums_before
    backend_changed = backend_after != backend_before
    interactive_changed = interactive_after != interactive_before
    mutated = [k for k in rc_before if rc_before[k] != rc_after.get(k)]

    if leaked:
        print(
            f"\n\n===== [STATE GUARD] {test_name}: Leaked figures: {fignums_after} (was: {fignums_before}) =====",
            file=sys.stderr,
        )
        plt.close("all")

    if backend_changed:
        print(
            f"\n\n===== [STATE GUARD] {test_name}: Backend changed: {backend_before} -> {backend_after} =====",
            file=sys.stderr,
        )

    if interactive_changed:
        print(
            f"\n\n===== [STATE GUARD] {test_name}: Interactive mode changed: {interactive_before} -> {interactive_after} =====",
            file=sys.stderr,
        )

    if mutated:
        print(
            f"\n\n===== [STATE GUARD] {test_name}: rcParams mutated (first 10): {mutated[:10]} =====",
            file=sys.stderr,
        )


@pytest.fixture(autouse=True)
def guard_environment():
    """Guard against environment mutations (cwd, os.environ)."""
    if not RUN_STATE_GUARDS:
        yield
        return

    old_env = dict(os.environ)
    old_cwd = Path.cwd()

    yield

    # Check for cwd change
    cwd_changed = Path.cwd() != old_cwd

    # Check for environ changes
    new_env = dict(os.environ)
    added = set(new_env.keys()) - set(old_env.keys())
    removed = set(old_env.keys()) - set(new_env.keys())
    changed = {k for k in old_env if k in new_env and old_env[k] != new_env[k]}

    # Print only if changes detected
    if cwd_changed:
        print(
            f"\n\n===== [ENV GUARD] Working directory changed: {old_cwd} -> {Path.cwd()} ====="
        )
    if added:
        print(f"\n\n===== [ENV GUARD] Added env vars: {added} =====")
    if removed:
        print(f"\n\n===== [ENV GUARD] Removed env vars: {removed} =====")
    if changed:
        print(
            f"\n\n===== [ENV GUARD] Changed env vars (first 5): {list(changed)[:5]} ====="
        )


@pytest.fixture(autouse=True)
def _prefs_guard():
    """Detect SpectroChemPy preferences mutations (diagnostic only)."""
    if not RUN_STATE_GUARDS:
        yield
        return

    from spectrochempy.application.preferences import preferences as prefs

    style_before = prefs.style
    type_before = type(prefs.style)

    yield

    style_after = prefs.style
    type_after = type(prefs.style)

    # Print only if changes detected
    if style_before != style_after:
        print(
            f"\n[STATE GUARD] prefs.style changed: {style_before!r} -> {style_after!r}"
        )
    if type_before != type_after:
        print(
            f"\n[STATE GUARD] prefs.style type changed: {type_before} -> {type_after}"
        )


# ======================================================================================
# ISOLATION FIXTURES - Run before/after EVERY test
# ======================================================================================


@pytest.fixture(autouse=True)
def isolate_matplotlib_state():
    """Ensure matplotlib state is clean before and after each test."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Save original state
    original_rc = mpl.rcParams.copy()
    original_figs = plt.get_fignums()

    yield

    # Restore rcParams
    mpl.rcParams.update(original_rc)

    # Close all figures created during test
    plt.close("all")

    # Also close any figures that weren't in the original set
    for fig_num in plt.get_fignums():
        if fig_num not in original_figs:
            plt.close(fig_num)


@pytest.fixture(autouse=True)
def isolate_scp_preferences():
    """Ensure SpectroChemPy preferences are restored after each test."""
    from spectrochempy.application.preferences import preferences as prefs

    # Take a snapshot of preferences
    snapshot = prefs.to_dict()

    yield

    # Restore preferences from snapshot
    prefs.update(snapshot)


# ----------------------------
# Cleaning when exiting pytest
# ----------------------------
def pytest_sessionfinish(session, exitstatus):  # pragma: no cover
    """Whole test run finishes."""

    # cleaning
    cwd = Path(__file__).parent.parent

    for f in list(cwd.glob("**/*.?scp")):
        f.unlink()
    for f in list(cwd.glob("**/*.jdx")):
        f.unlink()
    for f in list(cwd.glob("**/*.json")):
        if (  # important: add any parts to exclude from deleting json files
            f.name not in ["zenodo.json", "versions.json", "preferences.json"]
            and ".vscode" not in f.parts
            and ".venv" not in f.parts
            and ".git" not in f.parts
        ):
            f.unlink()
    for f in list(cwd.glob("**/*.log")):
        f.unlink()


# ======================================================================================
# FIXTURES
# ======================================================================================


# --------------------------------------------------------------------------------------
# initialize a ipython session before calling spectrochempy
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def session_ip():
    try:
        from IPython.testing.globalipapp import start_ipython

        return start_ipython()
    except ImportError:
        return None


@pytest.fixture(scope="module")
def ip(session_ip):
    yield session_ip


# --------------------------------------------------------------------------------------
# Path utilities
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="function")
def mock_cwd(monkeypatch, tmp_path):
    # Mock the current directory to use a temporary directory."""
    monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
    # Mock Path.resolve to ensure predictable path resolution in tests
    monkeypatch.setattr(Path, "resolve", lambda self: tmp_path)
    return tmp_path


from spectrochempy.application.preferences import preferences as prefs
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.basearrays.ndcomplex import NDComplexArray
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.project.project import Project
from spectrochempy.core.script import Script
from spectrochempy.utils.file import pathclean
from spectrochempy.utils.testing import RandomSeedContext

# Test data directory
datadir = pathclean(prefs.datadir)

# --------------------------------------------------------------------------------------
# Test data download policy
#
# By default, the full testdata directory is NOT downloaded automatically.
#
# To enable downloads, set the environment variable:
#   SCP_TEST_DATA_DOWNLOAD=1
#
# When enabled, the download happens once and is cached on disk.
# When disabled (default), tests that require external data will skip
# gracefully if the data is not already present.
#
# This protects CI and local runs from:
#   - unintended network access during test collection
#   - slow downloads during simple unit test runs
#   - dependency on GitHub archive availability
#
# Tests that require downloaded data MUST be marked with @pytest.mark.data
# and will be skipped when the data is not available.
# --------------------------------------------------------------------------------------
_SCP_TEST_DATA_DOWNLOAD = os.environ.get("SCP_TEST_DATA_DOWNLOAD", "0") == "1"
_has_testdata = (datadir / "__downloaded__").exists()

if _SCP_TEST_DATA_DOWNLOAD and not _has_testdata:
    from spectrochempy.application.testdata import download_full_testdata_directory

    download_full_testdata_directory(datadir, force=False)
    _has_testdata = True


# --------------------------------------------------------------------------------------
# create reference arrays
# --------------------------------------------------------------------------------------
with RandomSeedContext(12345):
    ref_data = 10.0 * np.random.random((10, 8)) - 5.0
    ref3d_data = 10.0 * np.random.random((10, 100, 3)) - 5.0
    ref3d_2_data = np.random.random((9, 50, 4))

ref_mask = ref_data < -4
ref3d_mask = ref3d_data < -3
ref3d_2_mask = ref3d_2_data < -2


# --------------------------------------------------------------------------------------
# Fixtures: some NDArray's
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def refarray():
    return ref_data.copy()


@pytest.fixture(scope="function")
def refmask():
    return ref_mask.copy()


@pytest.fixture(scope="function")
def ndarray():
    # return a simple ndarray with some data
    return NDArray(ref_data, desc="An array", copy=True).copy()


@pytest.fixture(scope="function")
def ndarrayunit():
    # return a simple ndarray with some data and units
    return NDArray(ref_data, units="m/s", copy=True).copy()


@pytest.fixture(scope="function")
def ndarraymask():
    # return a simple ndarray with some data and units
    return NDArray(
        ref_data, mask=ref_mask, units="m/s", history="Creation with mask", copy=True
    ).copy()


# --------------------------------------------------------------------------------------
# Fixtures: Some NDComplex arrays
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def ndarraycplx():
    # return a complex ndarray
    return NDComplexArray(ref_data, units="m/s", dtype=np.complex128, copy=True).copy()


# --------------------------------------------------------------------------------------
# Fixtures: Some NDDatasets
# --------------------------------------------------------------------------------------
coord0_ = Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="function")
def coord0():
    return coord0_.copy()


coord1_ = Coord(data=np.linspace(0.0, 60.0, 100), units="s", title="time-on-stream")


@pytest.fixture(scope="function")
def coord1():
    return coord1_.copy()


coord2_ = Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def coord2():
    return coord2_.copy()


coord2b_ = Coord(
    data=np.linspace(1.0, 20.0, 3),
    labels=["low", "medium", "high"],
    units="tesla",
    title="magnetic field",
)


@pytest.fixture(scope="function")
def coord2b():
    return coord2b_.copy()


coord0_2_ = Coord(
    data=np.linspace(4000.0, 1000.0, 9),
    labels=list("abcdefghi"),
    units="cm^-1",
    title="wavenumber",
)


@pytest.fixture(scope="function")
def coord0_2():
    return coord0_2_.copy()


coord1_2_ = Coord(data=np.linspace(0.0, 60.0, 50), units="s", title="time-on-stream")


@pytest.fixture(scope="function")
def coord1_2():
    return coord1_2_.copy()


coord2_2_ = Coord(
    data=np.linspace(200.0, 1000.0, 4),
    labels=["cold", "normal", "hot", "veryhot"],
    units="K",
    title="temperature",
)


@pytest.fixture(scope="function")
def coord2_2():
    return coord2_2_.copy()


@pytest.fixture(scope="function")
def nd1d():
    # a simple ddataset
    return NDDataset(ref_data[:, 1].squeeze()).copy()


@pytest.fixture(scope="function")
def nd2d():
    # a simple 2D ndarrays
    return NDDataset(ref_data).copy()


@pytest.fixture(scope="function")
def ref_ds():
    # a dataset with coordinates
    return ref3d_data.copy()


@pytest.fixture(scope="function")
def ds1():
    # a dataset with coordinates
    return NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coord2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ds2():
    # another dataset
    return NDDataset(
        ref3d_2_data,
        coordset=[coord0_2_, coord1_2_, coord2_2_],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def dsm():
    # dataset with coords containing several axis and a mask

    coordmultiple = CoordSet(coord2_, coord2b_)
    return NDDataset(
        ref3d_data,
        coordset=[coord0_, coord1_, coordmultiple],
        mask=ref3d_mask,
        title="absorbance",
        units="absorbance",
    ).copy()


# Flag indicating whether large external testdata is available
# (downloaded via download_full_testdata_directory or by setting SCP_TEST_DATA_DOWNLOAD=1)

_IR_DATA_PATH = datadir / "irdata" / "nh4y-activation.spg"
_has_ir_data = _has_testdata and _IR_DATA_PATH.exists()


@pytest.fixture(scope="function")
def IR_dataset_2D():
    if not _has_ir_data:
        pytest.skip(
            "IR test data not available. "
            "Set SCP_TEST_DATA_DOWNLOAD=1 to download, "
            "or use lightweight synthetic fixtures."
        )
    nd = spectrochempy.read(_IR_DATA_PATH)
    nd.name = "IR_2D"
    return nd


@pytest.fixture(scope="function")
def IR_dataset_1D():
    if not _has_ir_data:
        pytest.skip(
            "IR test data not available. "
            "Set SCP_TEST_DATA_DOWNLOAD=1 to download, "
            "or use lightweight synthetic fixtures."
        )
    nd = spectrochempy.read(_IR_DATA_PATH)[0].squeeze()
    nd.name = "IR_1D"
    return nd


# --------------------------------------------------------------------------------------
# Fixture: NMR spectra
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def NMR_dataset_1D():
    pytest.importorskip("spectrochempy_nmr", reason="requires the NMR plugin")
    if not _has_testdata:
        pytest.skip("NMR test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d" / "1" / "fid"
    dataset = spectrochempy.read_topspin(
        path, remove_digital_filter=True, name="NMR_1D"
    )
    return dataset.copy()


@pytest.fixture(scope="function")
def NMR_dataset_2D():
    pytest.importorskip("spectrochempy_nmr", reason="requires the NMR plugin")
    if not _has_testdata:
        pytest.skip("NMR test data not available (set SCP_TEST_DATA_DOWNLOAD=1)")
    path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d" / "1" / "ser"
    dataset = spectrochempy.read_topspin(
        path, expno=1, remove_digital_filter=True, name="NMR_2D"
    )
    return dataset.copy()


@pytest.fixture(scope="function")
def JDX_2D():
    return """##TITLE=IR_2D
    ##JCAMP-DX=5.01
    ##DATA TYPE=LINK
    ##BLOCKS=3
    ##TITLE=vz0466.spa, Wed Jul 06 21:00:38 2016 (GMT+02:00)
    ##JCAMP-DX=5.01
    ##ORIGIN=omnic
    ##OWNER=christian@MacCF.local
    ##LONGDATE=2016/07/06
    ##TIME=19:03:14
    ##XUNITS=1/CM
    ##YUNITS=ABSORBANCE
    ##FIRSTX=5999.555664
    ##LASTX=5981.234938
    ##MAXX=5999.555664
    ##MINX=5981.234938
    ##XFACTOR=1.0
    ##FIRSTY=2.057242
    ##LASTY=2.047518
    ##MAXY=2.060812
    ##MINY=2.002743
    ##YFACTOR=1e-08
    ##NPOINTS=20
    ##XYDATA=(X++(Y..Y))
    5999.555664 205724239.000000 206081247.000000 206077337.000000 205761122.000000
    5995.698669 205383539.000000 205176067.000000 205163621.000000 205173397.000000
    5991.841674 205024647.000000 204717803.000000 204435229.000000 204356765.000000
    5987.984679 204492545.000000 204698991.000000 204843306.000000 204916620.000000
    5984.127684 204970645.000000 204992580.000000 204916238.000000 204751849.000000
    ##END
    ##TITLE=vz0467.spa, Wed Jul 06 21:10:38 2016 (GMT+02:00)
    ##JCAMP-DX=5.01
    ##ORIGIN=omnic
    ##OWNER=christian@MacCF.local
    ##LONGDATE=2016/07/06
    ##TIME=19:13:14
    ##XUNITS=1/CM
    ##YUNITS=ABSORBANCE
    ##FIRSTX=5999.555664
    ##LASTX=5981.234938
    ##MAXX=5999.555664
    ##MINX=5981.234938
    ##XFACTOR=1.0
    ##FIRSTY=2.057242
    ##LASTY=2.047518
    ##MAXY=2.060812
    ##MINY=2.002743
    ##YFACTOR=1e-08
    ##NPOINTS=20
    ##XYDATA=(X++(Y..Y))
    5999.555664 203330016.000000 203735828.000000 203964066.000000 203920483.000000
    5995.698669 203658008.000000 203311467.000000 203041100.000000 202970385.000000
    5991.841674 203081583.000000 203193688.000000 203127312.000000 202899098.000000
    5987.984679 202673625.000000 202544808.000000 202475214.000000 202450990.000000
    5984.127684 202554702.000000 202821636.000000 203135919.000000 203363680.000000
    ##END
    ##TITLE=vz0468.spa, Wed Jul 06 21:20:38 2016 (GMT+02:00)
    ##JCAMP-DX=5.01
    ##ORIGIN=omnic
    ##OWNER=christian@MacCF.local
    ##LONGDATE=2016/07/06
    ##TIME=19:23:14
    ##XUNITS=1/CM
    ##YUNITS=ABSORBANCE
    ##FIRSTX=5999.555664
    ##LASTX=5981.234938
    ##MAXX=5999.555664
    ##MINX=5981.234938
    ##XFACTOR=1.0
    ##FIRSTY=2.057242
    ##LASTY=2.047518
    ##MAXY=2.060812
    ##MINY=2.002743
    ##YFACTOR=1e-08
    ##NPOINTS=20
    ##XYDATA=(X++(Y..Y))
    5999.555664 200489759.000000 200328993.000000 200274324.000000 200440287.000000
    5995.698669 200729274.000000 200897192.000000 200821328.000000 200623416.000000
    5991.841674 200488710.000000 200454282.000000 200441122.000000 200419855.000000
    5987.984679 200450921.000000 200571250.000000 200716447.000000 200771117.000000
    5984.127684 200681734.000000 200519037.000000 200432109.000000 200508308.000000
    ##END
    ##END=

    """


# --------------------------------------------------------------------------------------
# fixture Project
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="function")
def simple_project():
    proj = Project(
        # subprojects
        Project(name="P350", label=r"$\mathrm{M_P}\,(623\,K)$"),
        Project(name="A350", label=r"$\mathrm{M_A}\,(623\,K)$"),
        Project(name="B350", label=r"$\mathrm{M_B}\,(623\,K)$"),
        # attributes
        name="project_1",
        label="main project",
    )

    assert proj.projects_names == ["P350", "A350", "B350"]

    ir = NDDataset([1.1, 2.2, 3.3], coordset=[[1, 2, 3]])
    tg = NDDataset([1, 3, 4], coordset=[[1, 2, 3]])
    proj.A350["IR"] = ir
    proj.A350["TG"] = tg
    script_source = (
        "set_loglevel(INFO)\n"
        'info_(f"samples contained in the project are {proj.projects_names}")'
    )

    proj["print_info"] = Script("print_info", script_source)
    return proj


# ===========================================================================
# Core test fixtures (deterministic synthetic datasets for unit testing)
# Merged from tests/test_core/conftest.py to avoid ImportPathMismatchError.
# All are: deterministic, small, fast, self-contained.
# ===========================================================================

# -- Reference data constants --
_ref_1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

with RandomSeedContext(42):
    _ref_2d = np.round(10.0 * np.random.random((5, 5)), 2)

with RandomSeedContext(123):
    _ref_3d = np.round(10.0 * np.random.random((3, 4, 2)), 2)

_ref_mask_1d = np.array(
    [False, True, False, False, True, False, False, False, True, False]
)
_ref_mask_2d = _ref_2d < 3.0

_core_coord_x = Coord(
    data=np.linspace(4000.0, 1000.0, 10),
    labels=list("abcdefghij"),
    units="cm^-1",
    title="wavenumber",
)
_core_coord_y = Coord(
    data=np.linspace(0.0, 60.0, 100),
    units="s",
    title="time",
)
_core_coord_z = Coord(
    data=np.linspace(200.0, 300.0, 3),
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)
_core_coord_2d_x = Coord(np.linspace(0.0, 10.0, 5), title="x_coord")
_core_coord_2d_y = Coord(np.linspace(0.0, 5.0, 5), title="y_coord")


@pytest.fixture(scope="function")
def ndarray_1d():
    return NDArray(_ref_1d.copy(), copy=True)


@pytest.fixture(scope="function")
def ndarray_2d():
    return NDArray(_ref_2d.copy(), copy=True)


@pytest.fixture(scope="function")
def ndarray_1d_unit():
    return NDArray(_ref_1d.copy(), units="m/s", copy=True)


@pytest.fixture(scope="function")
def ndarray_2d_mask():
    return NDArray(
        _ref_2d.copy(),
        mask=_ref_mask_2d.copy(),
        units="absorbance",
        copy=True,
    )


@pytest.fixture(scope="function")
def ndarray_complex():
    return NDComplexArray(_ref_2d.copy().astype(np.complex128), units="m/s", copy=True)


@pytest.fixture(scope="function")
def coord_x():
    return _core_coord_x.copy()


@pytest.fixture(scope="function")
def coord_y():
    return _core_coord_y.copy()


@pytest.fixture(scope="function")
def coord_z():
    return _core_coord_z.copy()


@pytest.fixture(scope="function")
def coord_2d_x():
    return _core_coord_2d_x.copy()


@pytest.fixture(scope="function")
def coord_2d_y():
    return _core_coord_2d_y.copy()


@pytest.fixture(scope="function")
def ndataset_1d():
    return NDDataset(_ref_1d.copy()).copy()


@pytest.fixture(scope="function")
def ndataset_2d():
    return NDDataset(_ref_2d.copy()).copy()


@pytest.fixture(scope="function")
def ndataset_3d():
    return NDDataset(
        _ref_3d.copy(),
        coordset=[_core_coord_x.copy(), _core_coord_y.copy(), _core_coord_z.copy()],
        title="absorbance",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_1d_unit():
    return NDDataset(_ref_1d.copy(), units="m").copy()


@pytest.fixture(scope="function")
def ndataset_2d_units():
    return NDDataset(_ref_2d.copy(), units="absorbance").copy()


@pytest.fixture(scope="function")
def ndataset_2d_masked():
    return NDDataset(
        _ref_2d.copy(),
        mask=_ref_mask_2d.copy(),
        coordset=[_core_coord_2d_y.copy(), _core_coord_2d_x.copy()],
        title="masked_data",
        units="absorbance",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_1d_masked():
    return NDDataset(
        _ref_1d.copy(),
        mask=_ref_mask_1d.copy(),
        title="masked_1d",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_complex():
    return NDDataset(
        _ref_1d.copy().astype(np.complex128) * (1.0 + 0.5j),
        title="complex_data",
    ).copy()


@pytest.fixture(scope="function")
def ndataset_nan():
    data = _ref_2d.copy()
    data[0, 0] = np.nan
    data[2, 3] = np.nan
    return NDDataset(data, title="with_nan").copy()


@pytest.fixture(scope="function")
def ndataset_broadcast():
    return NDDataset(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])).copy()


@pytest.fixture(scope="function")
def ndataset_aligned_pair(coord_x):
    c = coord_x
    d1 = NDDataset(np.sin(np.linspace(0, np.pi, 10)), coordset=[c])
    d2 = NDDataset(np.cos(np.linspace(0, np.pi, 10)), coordset=[c.copy()])
    return d1, d2


@pytest.fixture(scope="function")
def ndataset_misaligned_pair():
    c1 = Coord(np.linspace(0.0, 10.0, 10))
    c2 = Coord(np.linspace(1.0, 11.0, 10))
    d1 = NDDataset(np.ones(10), coordset=[c1])
    d2 = NDDataset(np.ones(10), coordset=[c2])
    return d1, d2


# --------------------------------------------------------------------------------------
# fixture mpl dirs
# --------------------------------------------------------------------------------------
@pytest.fixture
def fake_mpl_dirs(tmp_path, monkeypatch):
    """Fake Matplotlib configdir, datadir and cachedir."""
    configdir = tmp_path / "config"
    datadir = tmp_path / "data"
    cachedir = tmp_path / "cache"

    configdir.mkdir()
    datadir.mkdir()
    cachedir.mkdir()

    monkeypatch.setattr("matplotlib.get_configdir", lambda: str(configdir))
    monkeypatch.setattr("matplotlib.get_data_path", lambda: str(datadir))
    monkeypatch.setattr("matplotlib.get_cachedir", lambda: str(cachedir))

    return configdir, datadir, cachedir
