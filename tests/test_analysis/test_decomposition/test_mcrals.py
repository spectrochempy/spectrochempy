# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

from os import environ

import numpy as np
import pytest
import traitlets as tr

import spectrochempy as scp
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.application.application import set_loglevel
from spectrochempy.core.dataset.nddataset import Coord, NDDataset
from spectrochempy.processing.transformation.npy import dot
from spectrochempy.utils import docutils as chd
from spectrochempy.utils import testing


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_MCRALS_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.mcrals"
    chd.check_docstrings(
        module,
        obj=scp.MCRALS,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01", "PR06"],
    )


def gaussian(x, h, c, w, noise):
    with testing.RandomSeedContext(12345):
        # seed the rndom generator to ensure the output is always the same for comparison
        return h * (np.exp(-1 / 2 * ((x.data - c) / w) ** 2)) + noise * np.random.randn(
            len(x)
        )  # a gaussian with added noise


def expon(t, c0, l, noise):
    with testing.RandomSeedContext(1234589):
        return c0 * (np.exp(l * t.data)) + noise * np.random.randn(len(t.data))


def get_C(C):
    return C


def get_C_a(C, a):
    return C / a


def get_C_kb(C, b=1):
    return C * b


def get_C_akb(C, a, b=1):
    return C * b / a


def get_St(St):
    return St


# --- generators for the hard-constraint dispatch branches (PR1 regression) ---


def get_St_a(St, a):
    # args-only path
    return St / a


def get_St_kb(St, b=1):
    # kwargs-only path
    return St * b


def get_St_akb(St, a, b=1):
    # args + kwargs path
    return St * b / a


def get_St_tuple(St):
    # 2-tuple return: (profiles, new_args)
    return St, ()


def get_St_tuple_extra(St):
    # 3-tuple return: (profiles, new_args, extra)
    return St, (), {"extra": "spec"}


def get_St_zero(St):
    # returns a profile with a zero row (to exercise normSpec zero guard)
    St = St.copy()
    St[0] = 0
    return St


def get_C_tuple(C):
    return C, ()


def get_C_tuple_extra(C):
    return C, (), {"extra": "conc"}


@pytest.fixture()
def model():
    class Model(object):
        def __init__(self):
            n_t = 10
            n_wl = 100
            h = [1, 1]
            c = [250, 750]
            w = [100, 100]
            noise_spec = [0.0, 0.0]
            noise_conc = [0.0, 0.0]

            c0 = [10, 1]
            l = np.array([-2, 2]) * 1e-2

            n_PS = 2  # number of pure species
            t_c = Coord(np.arange(0, 100, 100 / n_t), title="time", units="s")
            wl_c = Coord(
                np.arange(0, 1000, 1000 / n_wl), title="wavelength", units="nm"
            )
            PS_c = Coord(
                range(n_PS),
                title="species",
                units=None,
                labels=["PS#" + str(i) for i in range(n_PS)],
            )

            self.St0 = St0 = NDDataset.zeros((n_PS, len(wl_c)), coordset=(PS_c, wl_c))
            self.C0 = C0 = NDDataset.zeros((len(t_c), n_PS), coordset=(t_c, PS_c))

            self.St = St = NDDataset.zeros((n_PS, len(wl_c)), coordset=(PS_c, wl_c))
            self.C = C = NDDataset.zeros((len(t_c), n_PS), coordset=(t_c, PS_c))

            for i, id in enumerate((0, 1)):
                self.C.data[:, i] = expon(t_c, c0[id], l[id], noise_conc[id])
                self.St.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], noise_spec[id])

                self.C0.data[:, i] = expon(t_c, c0[id], l[id], 0)
                self.St0.data[i, :] = gaussian(wl_c, h[id], c[id], w[id], 0)

    return Model()


@pytest.fixture
def data(model):
    D = dot(model.C, model.St)
    D.title = "intensity"
    D.units = "absorbance"
    D.set_coordset(None, None)
    D.y.title = "elution time"
    D.x.title = "wavelength"
    D.y.units = "hours"
    D.x.units = "cm^-1"
    return D


def _assert_mcrals_shapes(mcr, data, n_components=2):
    assert mcr.C.shape == (data.shape[0], n_components)
    assert mcr.St.shape == (n_components, data.shape[1])


def _assert_reconstructs(mcr, data, tol=1.0e-12):
    reconstructed = mcr.inverse_transform()
    assert reconstructed.shape == data.shape
    assert (reconstructed - data).abs().max() < tol


def test_mcrals_basic_fit_converges(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS(log_level="INFO")
    mcr.tol = 30.0
    result = mcr.fit(D, C0)

    assert result is mcr
    assert mcr.log.endswith("converged !")
    _assert_mcrals_shapes(mcr, D)
    assert mcr.transform(D) == mcr.C

    params = mcr.params()
    assert len(params) == 32
    assert np.all(params.closureTarget == [1] * 10)
    assert params.tol == 30.0

    params = mcr.params(default=True)
    assert params.tol == 0.1
    assert mcr.C_ls_list == []
    _assert_reconstructs(mcr, D)


def test_mcrals_iteration_storage(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.fit(D, C0)
    mcr.storeIterations = True
    mcr.fit(D, (mcr.C, mcr.St))
    assert mcr.C_ls_list != []

    mcr1 = MCRALS()
    mcr1.fit(D, C0)
    assert np.max(np.abs(mcr.C - mcr1.C)) < 1.0e-12
    assert np.max(np.abs(mcr.St - mcr1.St)) < 1.0e-12


def test_mcrals_diverging_path_stops_cleanly(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.fit(D, C0)
    mcr.monoIncConc = [0, 1]
    mcr.monoIncTol = 1.0
    mcr.unimodSpec = [0, 1]
    mcr.normSpec = "euclid"
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.fit(D, C0)
    assert mcr.log.endswith("Stop ALS optimization.")


def test_mcrals_closure_all_regression_issue_911(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.closureConc = "all"
    mcr.closureMethod = "constantSum"
    mcr.maxdiv = 1

    mcr.fit(D, C0)
    assert mcr.log.endswith("Stop ALS optimization.")


def test_mcrals_hard_concentration_model_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C
    mcr.argsGetConc = ()
    mcr.kwargsGetConc = {}
    mcr.tol = 30.0

    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_constructor_constraints_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS(
        monoIncConc=[0],
        monoDecConc=[1],
        closureConc=[0, 1],
        normSpec="max",
        max_iter=1,
    )
    set_loglevel("WARNING")
    mcr.fit(D, C0)
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_hard_spectral_model_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St
    mcr.tol = 30.0

    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_closure_with_array_guess_smoke(model, data):
    D = data
    C0 = model.C0

    mcr = MCRALS(
        normSpec="euclid",
        closureConc=[0, 1],
        closureMethod="constantSum",
        max_iter=1,
    )
    mcr.fit(D, C0.data)
    assert "Convergence criterion ('tol')" in mcr.log[-100:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_spectral_guess_converges(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0)
    mcr.fit(D, St0.data)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)


def test_mcrals_nnls_solver_smoke(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0, nonnegConc=[], solverConc="nnls", solverSpec="nnls")
    mcr.fit(D, St0.data)

    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert mcr.C.min() >= -1.0e-12
    assert mcr.St.min() >= -1.0e-12


def test_mcrals_pnnls_solver_smoke(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(
        tol=15.0, nonnegConc=[0], solverConc="pnnls", nonnegSpec=[0], solverSpec="pnnls"
    )
    mcr.fit(D, St0.data)

    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert mcr.C[:, 0].min() >= -1.0e-12
    assert mcr.St[0].min() >= -1.0e-12


def test_mcrals_pnnls_concentration_solver_reconstructs(model, data):
    D = data
    St0 = model.St0

    mcr = MCRALS(tol=15.0, solverConc="pnnls")
    mcr.fit(D, St0.data)

    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    _assert_reconstructs(mcr, D)


def test_MCRALS_errors(model, data):
    # Text exceptions
    D = data
    C0 = model.C0
    mcr = MCRALS()

    # inexistant keyword parameters
    try:
        _ = MCRALS(max_iter=25, inexistant=0, log_level="DEBUG")
    except KeyError as exc:
        assert "'inexistant' is not a valid" in exc.args[0]

    # guess with wrong size of dimensions
    try:
        mcr.fit(
            D,
            np.ones((11, 2)),
        )
    except ValueError as e:
        assert "None of the dimensions of the given profile" in e.args[0]

    # guess with wrong nonnegConc parameter
    mcr = MCRALS()
    mcr.fit(D, C0)

    with pytest.raises(ValueError) as e:
        mcr.nonnegConc = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.nonnegConc = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    with pytest.raises(tr.TraitError):
        mcr.unimodSpec = "alls"

    with pytest.raises(tr.TraitError):
        mcr.unimodConc = None

    with pytest.raises(tr.TraitError):
        mcr.nonnegSpec = None

    with pytest.raises(tr.TraitError):
        mcr.nonnegSpec = None

    # guess with wrong unimodConc parameter
    with pytest.raises(ValueError) as e:
        mcr.unimodConc = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.unimodConc = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # wrong closureTarget
    with pytest.raises(ValueError) as e:
        mcr.closureTarget = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.getC_to_C_idx = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # wrong unimodSpec
    with pytest.raises(ValueError) as e:
        mcr.unimodSpec = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.unimodSpec = [0, 1, 1]
    assert "please check the" in e.value.args[0]

    # wrong nonnegSpec
    with pytest.raises(ValueError) as e:
        mcr.nonnegSpec = [2]
    assert "please check the" in e.value.args[0]

    with pytest.raises(ValueError) as e:
        mcr.nonnegSpec = [0, 1, 1]
    assert "please check the" in e.value.args[0]


# --------------------------------------------------------------------------------------
# PR1 regression tests
# --------------------------------------------------------------------------------------


def test_MCRALS_pr1_getspec_with_args(model, data):
    """B1: `argsGetSpecc` typo used to raise AttributeError here."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_a
    mcr.argsGetSpec = (1.0,)
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert np.all(np.isfinite(mcr.C.data))
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_with_kwargs(model, data):
    """Dispatch branch: kwargs-only path for getSpec."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_kb
    mcr.kwargsGetSpec = {"b": 1.0}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_with_args_and_kwargs(model, data):
    """Dispatch branch: args + kwargs path for getSpec."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_akb
    mcr.argsGetSpec = (1.0,)
    mcr.kwargsGetSpec = {"b": 1.0}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_tuple_return(model, data):
    """B2: a 2-tuple return from getSpec used to crash (`.data` on a tuple)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_tuple
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert np.all(np.isfinite(mcr.St.data))


def test_MCRALS_pr1_getspec_tuple_with_extra(model, data):
    """B2: a 3-tuple return from getSpec must populate extraOutputGetSpec."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_tuple_extra
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    # the extra output is a list with one dict per iteration
    assert mcr.extraOutputGetSpec != []
    assert mcr.extraOutputGetSpec[0] == {"extra": "spec"}


def test_MCRALS_pr1_getspec_tuple_extra_with_args(model, data):
    """B1+B2 combined: 3-tuple return with non-empty argsGetSpec."""
    D = data
    C0 = model.C0

    def _gen(St, a):
        return St / a, (a,), {"extra": "spec"}

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = _gen
    mcr.argsGetSpec = (1.0,)
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    assert mcr.extraOutputGetSpec[0] == {"extra": "spec"}


def test_MCRALS_pr1_getconc_tuple_return(model, data):
    """getConc 2-tuple return branch (previously untested)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_tuple
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_getconc_tuple_with_extra(model, data):
    """getConc 3-tuple return must populate extraOutputGetConc."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_tuple_extra
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    assert mcr.extraOutputGetConc != []
    assert mcr.extraOutputGetConc[0] == {"extra": "conc"}


def test_MCRALS_pr1_getconc_with_args(model, data):
    """getConc args-only dispatch branch (previously untested)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_a
    mcr.argsGetConc = (1.0,)
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_getconc_with_args_and_kwargs(model, data):
    """getConc args+kwargs dispatch branch (previously untested)."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardConc = [0, 1]
    mcr.getConc = get_C_akb
    mcr.argsGetConc = (1.0,)
    mcr.kwargsGetConc = {"b": 1.0}
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_getst_to_st_idx_none_entries(model, data):
    """B5: `getSt_to_St_idx` with `None` entries used to crash in `max()`."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.fit(D, C0)
    # assigning a list with a None entry must not raise (validator must skip None)
    mcr.getSt_to_St_idx = [0, None]
    assert mcr.getSt_to_St_idx == [0, None]

    # still rejects an out-of-range non-None index
    with pytest.raises(ValueError) as e:
        mcr.getSt_to_St_idx = [0, 5]
    assert "please check the" in e.value.args[0]


def test_MCRALS_pr1_closure_empty_list_is_noop(model, data):
    """B4: with the default `closureConc=[]`, closure must not run."""
    D = data
    C0 = model.C0

    # constantSum with an empty closure list used to run a wasteful / risky
    # block; it must now be a no-op and produce finite profiles.
    mcr = MCRALS()
    mcr.closureMethod = "constantSum"
    mcr.closureConc = []
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.C.data))

    # result must match a run where closure is fully disabled
    mcr_ref = MCRALS()
    mcr_ref.tol = 30.0
    mcr_ref.fit(D, C0)
    assert np.allclose(mcr.C.data, mcr_ref.C.data, atol=1e-12)


def test_MCRALS_pr1_closure_constantsum_smoke(model, data):
    """B4/B9: constantSum closure must run without producing nan/inf."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.closureConc = [0, 1]
    mcr.closureMethod = "constantSum"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.C.data))
    # the closure target (default -> ones) must be respected approximately
    assert np.allclose(
        np.sum(mcr.C_constrained.data[:, [0, 1]], axis=1),
        np.ones(D.shape[0]),
        atol=1e-6,
    )


def test_MCRALS_pr1_closure_single_component_is_active(model, data):
    """B4 regression: `closureConc=[0]` must activate closure on component 0.

    The earlier `np.any(self.closureConc)` guard evaluated `np.any([0]) == False`
    and silently disabled closure for a single selected component. The
    truthiness guard (`if self.closureConc:`) must keep it active.
    """
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.closureConc = [0]
    mcr.closureMethod = "constantSum"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.C.data))
    # component 0 must be closed to the default target (ones)
    assert np.allclose(
        mcr.C_constrained.data[:, 0],
        np.ones(D.shape[0]),
        atol=1e-6,
    )


def test_MCRALS_pr1_normspec_max_zero_guard(model, data):
    """B9: `normSpec='max'` with a zero spectrum row must not produce nan."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_zero
    mcr.normSpec = "max"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.St.data))
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_normspec_euclid_zero_guard(model, data):
    """B9: `normSpec='euclid'` with a zero spectrum row must not produce nan."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.hardSpec = [0, 1]
    mcr.getSpec = get_St_zero
    mcr.normSpec = "euclid"
    mcr.max_iter = 2
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert np.all(np.isfinite(mcr.St.data))
    assert np.all(np.isfinite(mcr.C.data))


def test_MCRALS_pr1_unimodconc_not_duplicated(model, data):
    """B3: `unimodConc` is defined only once; setting it must behave normally."""
    D = data
    C0 = model.C0

    mcr = MCRALS()
    mcr.unimodConc = "all"
    mcr.unimodConcMod = "strict"
    mcr.unimodConcTol = 1.1
    mcr.tol = 30.0
    mcr.fit(D, C0)
    assert "converged !" in mcr.log[-15:]
    _assert_mcrals_shapes(mcr, D)
    # the trait must be a single, functional trait
    assert mcr.unimodConc == list(range(mcr._n_components))


def test_MCRALS_pr1_unimodal_smooth_bounds():
    """B6: smooth unimodality must terminate, stay finite and in-bounds.

    The original `_unimodal_1D` could index out of bounds or infinite-loop
    for pathological tolerances (including ``tol == 1.0`` and ``tol < 1``).
    The fix keeps byte-identical results for the documented regime
    (``tol >= 1.1``) and only guarantees safety (termination + finite +
    correct shape) for the pathological cases.
    """
    from spectrochempy.analysis.decomposition.mcrals import _unimodal_1D

    cases = (
        np.array([6.0, 10.0]),
        np.array([10.0, 6.0]),
        np.array([0.0, 5.0, 1.0, 3.0, 10.0]),
        np.array([10.0, 1.0, 5.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 10.0, 1.0, 5.0, 1.0]),
        np.array([0.0, 4.0, 1.0, 3.0, 0.0, 2.0, 1.0]),
    )

    # documented regime (tol >= 1.1): finite, shape-preserving, both modes
    for a0 in cases:
        for mod in ("smooth", "strict"):
            a = _unimodal_1D(a0.copy(), tol=1.1, mod=mod)
            assert a.shape == a0.shape
            assert np.all(np.isfinite(a))

    # pathological tolerances that used to IndexError / infinite-loop
    # (including tol == 1.0): must terminate, stay in bounds and stay finite.
    for tol in (0.1, 0.3, 0.5, 0.9, 1.0):
        for a0 in cases:
            a = _unimodal_1D(a0.copy(), tol=tol, mod="smooth")
            assert a.shape == a0.shape
            assert np.all(np.isfinite(a))


# --------------------------------------------------------------------------------------
# PR3 — internal constraint engine
#
# These tests assert properties of the *internal* constraint abstraction
# introduced by PR3. The constraint classes are private (``_``-prefixed) and
# must never leak into the public API; the tests here only guard against
# accidental regressions of the internal contract (privacy, builder wiring,
# byte-equivalence of constraint application vs. the historical helpers).
# They are intentionally small and do not re-test the numerical behavior
# already covered by the end-to-end MCRALS regression suite.
# --------------------------------------------------------------------------------------


def test_MCRALS_pr3_constraint_classes_are_private():
    """Internal constraint classes must not be exported in ``__all__``."""
    import spectrochempy.analysis.decomposition.mcrals as mod

    assert mod.__all__ == ["MCRALS"]
    expected_classes = (
        "_Constraint",
        "_NonNegativeConstraint",
        "_UnimodalConstraint",
        "_MonotonicIncreaseConstraint",
        "_MonotonicDecreaseConstraint",
        "_ClosureConstraint",
        "_NormalizationConstraint",
        "_HardProfileConstraint",
    )
    for name in expected_classes:
        cls = getattr(mod, name, None)
        assert cls is not None, f"{name} missing from module"
        assert name.startswith("_"), f"{name} must remain private"


def test_MCRALS_pr3_constraint_apply_is_virtual():
    """The base ``_Constraint.apply`` must raise NotImplementedError."""
    from spectrochempy.analysis.decomposition.mcrals import _Constraint

    c = _Constraint()
    raised = False
    try:
        c.apply(None, None)
    except NotImplementedError:
        raised = True
    assert raised


def test_MCRALS_pr3_builders_translate_traitlets():
    """``_build_*`` constraints must reflect the active traitlets one-to-one.

    Inactive soft constraints are still emitted (they no-op internally via
    the ``np.any(indices)`` guard), while the hard-profile wrapper is always
    appended so that the per-iteration extra-output buffer is reset on
    inactive ``hardConc`` / ``hardSpec``.
    """
    import numpy as np

    from spectrochempy.analysis.decomposition.mcrals import (
        _ClosureConstraint,
        _HardProfileConstraint,
        _MonotonicDecreaseConstraint,
        _MonotonicIncreaseConstraint,
        _NonNegativeConstraint,
        _NormalizationConstraint,
        _UnimodalConstraint,
        MCRALS,
    )

    mcr = MCRALS()
    mcr.fit(np.zeros((4, 3)), np.ones((4, 2)))
    conc = mcr._build_concentration_constraints()
    spec = mcr._build_spectral_constraints()
    norm = mcr._build_normalization()

    assert isinstance(conc[0], _NonNegativeConstraint)
    assert isinstance(conc[1], _UnimodalConstraint)
    assert isinstance(conc[2], _MonotonicIncreaseConstraint)
    assert isinstance(conc[3], _MonotonicDecreaseConstraint)
    # closure is off by default (closureConc == [])
    assert not any(isinstance(c, _ClosureConstraint) for c in conc)
    # hard-profile wrapper always present
    assert isinstance(conc[-1], _HardProfileConstraint)
    assert isinstance(spec[0], _NonNegativeConstraint)
    assert isinstance(spec[1], _UnimodalConstraint)
    assert isinstance(spec[-1], _HardProfileConstraint)
    # normSpec is None by default -> no normalization constraint
    assert norm is None


def test_MCRALS_pr3_closure_built_only_when_truthy():
    """``closureConc=[0]`` (PR1 #911 single-component case) must build closure."""
    import numpy as np

    from spectrochempy.analysis.decomposition.mcrals import (
        _ClosureConstraint,
        MCRALS,
    )

    mcr = MCRALS()
    mcr.closureConc = "all"
    mcr.fit(np.zeros((4, 3)), np.ones((4, 2)))
    conc = mcr._build_concentration_constraints()
    assert any(isinstance(c, _ClosureConstraint) for c in conc)

    mcr2 = MCRALS()
    mcr2.closureConc = [0]
    mcr2.fit(np.zeros((4, 3)), np.ones((4, 2)))
    conc2 = mcr2._build_concentration_constraints()
    assert any(isinstance(c, _ClosureConstraint) for c in conc2)


def test_MCRALS_pr3_normalization_built_only_when_set():
    """``normSpec`` must drive normalization constraint construction."""
    import numpy as np

    from spectrochempy.analysis.decomposition.mcrals import (
        _NormalizationConstraint,
        MCRALS,
    )

    mcr = MCRALS(normSpec="max")
    mcr.fit(np.zeros((4, 3)), np.ones((4, 2)))
    norm = mcr._build_normalization()
    assert isinstance(norm, _NormalizationConstraint)

    mcr2 = MCRALS(normSpec="euclid")
    mcr2.fit(np.zeros((4, 3)), np.ones((4, 2)))
    assert isinstance(mcr2._build_normalization(), _NormalizationConstraint)

    mcr3 = MCRALS(normSpec=None)
    mcr3.fit(np.zeros((4, 3)), np.ones((4, 2)))
    assert mcr3._build_normalization() is None


def test_MCRALS_pr1_monotonic_tol_docstrings():
    """B7/B8: docstring/annotation sanity for the monotonic tolerances."""
    from spectrochempy.analysis.decomposition.mcrals import MCRALS as _M
    from spectrochempy.analysis.decomposition.mcrals import _unimodal_1D

    assert "monotonic increase" in _M.monoIncTol.help
    assert "monoIncTol" in _M.monoIncTol.help
    assert "monotonic decrease" in _M.monoDecTol.help
    assert "monoDecTol" in _M.monoDecTol.help
    # the type annotation of tol must be float, not str
    import inspect

    sig = inspect.signature(_unimodal_1D)
    assert sig.parameters["tol"].annotation is float
