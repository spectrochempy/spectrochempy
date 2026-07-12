"""
Canonical NMR metadata extraction.

Centralises all vendor-specific metadata interpretation behind a single
layer.  Each reader plugin provides a ``extract_nmr_metadata`` function
that maps its raw metadata to the canonical :class:`NMRMetadata` object.

The ``Experiment`` class consumes ``NMRMetadata`` and never references
Bruker-specific field names (``nuc1``, ``pulprog``, ``FnMODE``,
``datatype``, etc.) directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Canonical NMR metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NMRMetadata:
    """
    Vendor-neutral NMR metadata.

    All fields use plain Python types (str, int, float, tuple, None)
    so that the dataclass remains free of pint Quantity or any
    vendor-specific object.
    """

    ndim: int
    """Number of data dimensions (from the dataset shape)."""

    domains: tuple[str, ...]
    """Per-dimension domain state.

    Each element is one of ``'time'``, ``'frequency'``, or ``'unknown'``.
    """

    encoding: tuple[str, ...] | None = None
    """Per-dimension quadrature encoding.

    Canonical strings: ``'QF'``, ``'QSIM'``, ``'QSEQ'``, ``'DQD'``,
    ``'TPPI'``, ``'STATES'``, ``'STATES-TPPI'``, ``'ECHO-ANTIECHO'``,
    ``'undefined'``, or ``None``.
    """

    nuclei: tuple[str, ...] | None = None
    """Observed nucleus per dimension (e.g. ``('1H',)``)."""

    pulse_program: str | None = None
    """Pulse program name (vendor-specific string, may be ``None``)."""

    source_kind: str = "unknown"
    """Source data classification.

    One of: ``'fid'``, ``'ser'``, ``'processed_1d'``,
    ``'processed_2d'``, ``'partially_processed'``, ``'unknown'``.
    """

    datatype: str | None = None
    """Reader-reported data type (e.g. ``'FID'``, ``'SER'``, ``'1D'``, ``'2D'``)."""

    iscomplex: tuple[bool, ...] | None = None
    """Whether each dimension carries complex-valued data."""

    spectral_width_hz: tuple[float | None, ...] | None = None
    """Spectral width in Hz per dimension (plain float or None)."""

    spectrometer_freq_mhz: tuple[float | None, ...] | None = None
    """Spectrometer frequency in MHz per dimension (plain float or None)."""


# ---------------------------------------------------------------------------
# Source-kind inference (shared logic)
# ---------------------------------------------------------------------------


def infer_source_kind(
    ndim: int,
    domains: tuple[str, ...],
    datatype: str | None = None,
) -> str:
    """
    Derive the source-kind string from canonical fields.

    This function is vendor-neutral — it uses only ``ndim``, ``domains``,
    and the reader-reported ``datatype``.
    """
    if ndim == 1:
        if domains == ("time",):
            return "fid"
        if domains == ("frequency",):
            return "processed_1d"
    elif ndim == 2:
        if domains == ("time", "time"):
            return "ser"
        if domains == ("frequency", "frequency"):
            return "processed_2d"
        if domains[0] != domains[1]:
            return "partially_processed"
    return "unknown"


# ---------------------------------------------------------------------------
# Domain summarisation (shared logic)
# ---------------------------------------------------------------------------


def summarise_domain(domains: tuple[str, ...]) -> str:
    """Return a single domain label from per-dimension domain tuple."""
    if all(d == "time" for d in domains):
        return "time"
    if all(d == "frequency" for d in domains):
        return "frequency"
    if any(d == "unknown" for d in domains):
        return "unknown"
    return "mixed"


# ---------------------------------------------------------------------------
# TopSpin / Bruker extraction
# ---------------------------------------------------------------------------

# FnMODE integer → canonical string (Bruker convention).
_FNMODE_TO_CANONICAL: list[str] = [
    "undefined",
    "QF",
    "QSEQ",
    "TPPI",
    "STATES",
    "STATES-TPPI",
    "ECHO-ANTIECHO",
]


def _resolve_encoding(raw_encoding: list) -> tuple[str, ...]:
    """Convert raw encoding list (may contain ints) to canonical strings."""
    resolved = []
    for e in raw_encoding:
        if isinstance(e, int):
            if 0 <= e < len(_FNMODE_TO_CANONICAL):
                resolved.append(_FNMODE_TO_CANONICAL[e])
            else:
                resolved.append(f"unknown({e})")
        else:
            resolved.append(str(e))
    return tuple(resolved)


def extract_topspin_metadata(meta) -> NMRMetadata:
    """
    Extract :class:`NMRMetadata` from a TopSpin metadata object.

    Parameters
    ----------
    meta : Meta or None
        The ``dataset.meta`` object produced by the TopSpin reader.

    Returns
    -------
    NMRMetadata
        Canonical NMR metadata.  Fields for which the Bruker metadata
        is absent or unrecognisable are set to ``None`` or ``'unknown'``.
    """
    if meta is None or len(meta) == 0:
        return NMRMetadata(ndim=0, domains=())

    # --- dimensionality ---
    ndim = getattr(meta, "ndim", None)
    if ndim is None or ndim == 0:
        # Bruker reader stores ndim=None; infer from isfreq, iscomplex, etc.
        isfreq = getattr(meta, "isfreq", None)
        iscomplex = getattr(meta, "iscomplex", None)
        encoding = getattr(meta, "encoding", None)
        for candidate in (isfreq, iscomplex, encoding):
            if candidate is not None and len(candidate) > 0:
                ndim = len(candidate)
                break
        else:
            ndim = 0

    # --- domains ---
    isfreq = getattr(meta, "isfreq", None)
    if isfreq is not None:
        domains = tuple("frequency" if f else "time" for f in isfreq)
    else:
        domains = tuple("unknown" for _ in range(ndim))

    # --- encoding ---
    raw_enc = getattr(meta, "encoding", None)
    encoding = _resolve_encoding(raw_enc) if raw_enc is not None else None

    # --- nuclei ---
    raw_nuc = getattr(meta, "nuc1", None)
    nuclei = tuple(raw_nuc) if raw_nuc else None

    # --- pulse program ---
    pulse_program = getattr(meta, "pulprog", None)

    # --- datatype ---
    datatype = getattr(meta, "datatype", None)

    # --- iscomplex ---
    raw_ic = getattr(meta, "iscomplex", None)
    iscomplex = tuple(raw_ic) if raw_ic else None

    # --- spectral width (Hz) ---
    raw_sw = getattr(meta, "sw_h", None)
    sw_hz = None
    if raw_sw is not None:
        sw_hz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sw
        )

    # --- spectrometer frequency (MHz) ---
    raw_sfo = getattr(meta, "sfo1", None)
    sfo_mhz = None
    if raw_sfo is not None:
        sfo_mhz = tuple(
            float(v.magnitude) if hasattr(v, "magnitude") else (float(v) if v else None)
            for v in raw_sfo
        )

    # --- source kind ---
    source_kind = infer_source_kind(ndim, domains, datatype)

    return NMRMetadata(
        ndim=ndim,
        domains=domains,
        encoding=encoding,
        nuclei=nuclei,
        pulse_program=pulse_program,
        source_kind=source_kind,
        datatype=datatype,
        iscomplex=iscomplex,
        spectral_width_hz=sw_hz,
        spectrometer_freq_mhz=sfo_mhz,
    )
