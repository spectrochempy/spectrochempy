# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""
Cross-repository consumption contract for an installed spectrochempy-nmr wheel.

These tests exercise the public integration surface only: entry-point discovery,
namespace and aliases, a synthetic TopSpin read, the NMR unit context, and the
FFT/domain transition. They do not replace the plugin's own scientific suite.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

import numpy as np
import pytest

import spectrochempy as scp

pytestmark = pytest.mark.plugin

_MISSING = object()

pytest.importorskip(
    "spectrochempy_nmr",
    reason="requires the optional spectrochempy-nmr plugin",
)

_SYNTHETIC_ACQUS = """\
##TITLE= Parameter
##JCAMP-DX= 5.00
##DATA TYPE= NMR Spectrum
##OWNER= Bruker
##$TD= 2048
##$SW= 20.0
##$SW_h= 5000.0
##$SFO1= 400.13
##$BF1= 400.13
##$O1= 0.0
##$NUC1= <1H>
##$NS= 1
##$RG= 1.0
##$DECIM= 1
##$DSPFVS= 1
##$AQ_mod= 1
##$DATE= 0
##$DELTAT= 1.0
##$DIGMOD= 0
##$PARMODE= 0
##$BYTORDA= 0
##$DTYPA= 0
##$GRPDLY= 0
##$PULPROG= <zg>
##$SOLVENT= <CDCl3>
##END=
"""

_SYNTHETIC_PROCS = """\
##TITLE= Parameter
##JCAMP-DX= 5.00
##DATA TYPE= NMR Spectrum
##OWNER= Bruker
##$SF= 400.13
##$OFFSET= 10.0
##$SW_p= 20.0
##$SI= 1024
##$BYTORDA= 0
##$MC2= <1H>
##END=
"""


def _write_synthetic_topspin(root: Path) -> Path:
    """Write a minimal QSIM TopSpin experiment with no external dataset."""
    expno = root / "1"
    pdata = expno / "pdata" / "1"
    pdata.mkdir(parents=True)

    ncomplex = 1024
    samples = np.arange(ncomplex)
    fid = np.exp(-samples / 300.0) * np.exp(2j * np.pi * 0.05 * samples)
    interleaved = np.empty(2 * ncomplex, dtype="<i4")
    interleaved[0::2] = np.rint(fid.real * 1e6)
    interleaved[1::2] = np.rint(fid.imag * 1e6)
    (expno / "fid").write_bytes(interleaved.tobytes())
    (expno / "acqus").write_text(_SYNTHETIC_ACQUS)
    (pdata / "procs").write_text(_SYNTHETIC_PROCS)
    return root


def test_entry_point_is_discoverable():
    """The installed wheel registers the official nmr entry point."""
    matches = [
        entry_point
        for entry_point in entry_points(group="spectrochempy.plugins")
        if entry_point.name == "nmr"
    ]

    assert len(matches) == 1
    assert matches[0].value.split(":")[0] == "spectrochempy_nmr"


def test_public_namespace_and_aliases_resolve():
    """The public namespace and historical aliases remain callable."""
    from spectrochempy_nmr.units import set_nmr_context

    assert callable(scp.nmr.read)
    assert callable(scp.nmr.Experiment)
    assert callable(scp.nmr.read_topspin)
    assert callable(scp.read_topspin)
    assert callable(scp.nmr.set_nmr_context)
    assert scp.nmr.set_nmr_context is set_nmr_context


def test_plugin_inspection_reports_the_plugin_installed():
    """Plugin inspection exposes scp.nmr as an available namespace."""
    from spectrochempy.plugins.inspection import inspect_plugins  # noqa: PLC0415

    result = inspect_plugins()
    nmr = next(plugin for plugin in result.official if plugin.name == "nmr")

    assert nmr.status == "installed"
    assert "scp.nmr" in result.namespaces


def test_synthetic_topspin_directory_reads(tmp_path):
    """A synthetic TopSpin experiment reads through scp.nmr.read."""
    dataset = scp.nmr.read(_write_synthetic_topspin(tmp_path / "synthetic"))

    assert dataset.shape == (1024,)
    assert dataset.origin == "topspin"
    assert f"{dataset.meta.sfo1[0]:~.2f}" == "400.13 MHz"
    assert f"{dataset.meta.sw[0]:~.3f}" == "20.000 ppm"


def test_nmr_unit_context_converts_ppm_and_hertz():
    """The plugin contributes the ppm/frequency conversion context."""
    contexts = scp.ur._contexts
    previous_context = contexts.get("nmr")
    previous_larmor = (
        previous_context.defaults.get("larmor", _MISSING)
        if previous_context is not None
        else _MISSING
    )

    try:
        scp.nmr.set_nmr_context(400.13 * scp.ur.MHz)

        with scp.ur.context("nmr"):
            ppm = (400.13 * scp.ur.Hz).to("ppm")
            hertz = (1.0 * scp.ur.ppm).to("Hz")
    finally:
        if previous_context is None:
            contexts.pop("nmr", None)
        elif previous_larmor is _MISSING:
            previous_context.defaults.pop("larmor", None)
        else:
            previous_context.defaults["larmor"] = previous_larmor

    assert f"{ppm:~.3f}" == "1.000 ppm"
    assert f"{hertz:~.3f}" == "400.130 Hz"


def test_synthetic_fft_returns_a_frequency_domain_dataset(tmp_path):
    """The validated 1D workflow reaches the frequency domain."""
    dataset = scp.nmr.read(_write_synthetic_topspin(tmp_path / "synthetic"))
    experiment = scp.nmr.Experiment(dataset)

    assert experiment.is_time_domain

    processed = experiment.process(apodization="em", lb=2.0, size=4096)

    assert processed.shape == (4096,)
    assert scp.nmr.Experiment(processed).is_frequency_domain
    assert processed.coordset[0].units == scp.ur.ppm
