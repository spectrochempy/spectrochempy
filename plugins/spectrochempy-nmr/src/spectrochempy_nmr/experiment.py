"""
NMR Experiment model for SpectroChemPy.

Provides ``scp.nmr.Experiment``, an NMR-specific scientific interpretation
layer built on top of an existing ``NDDataset``.

All vendor-specific metadata interpretation is centralised in
:mod:`spectrochempy_nmr.nmr_metadata`.  This module consumes only the
canonical :class:`~spectrochempy_nmr.nmr_metadata.NMRMetadata` object and
never references Bruker-specific field names (``nuc1``, ``pulprog``,
``FnMODE``, ``datatype``, etc.) directly.
"""

from __future__ import annotations

import re
from numbers import Real
from typing import TYPE_CHECKING
from typing import cast

import numpy as np

from spectrochempy.core.units import Quantity
from spectrochempy.core.units import ur

if TYPE_CHECKING:
    from spectrochempy.core.dataset.nddataset import NDDataset


_APODIZATION_ALLOWED_PARAMS = {
    "em": ("lb",),
    "gm": ("lb", "gb"),
    "sp": ("ssb", "pow"),
}


class _DefaultSentinel:
    """Private default marker with a stable signature representation."""

    def __init__(self, rendered: str):
        self._rendered = rendered

    def __repr__(self) -> str:
        return self._rendered


_UNSET_NONE = _DefaultSentinel("None")
_UNSET_ZERO = _DefaultSentinel("0.0")
_DEFAULT_APODIZATION = cast(str | None, _UNSET_NONE)
_DEFAULT_LB = cast(float | Quantity | None, _UNSET_NONE)
_DEFAULT_GB = cast(float | Quantity | None, _UNSET_NONE)
_DEFAULT_SSB = cast(float | None, _UNSET_NONE)
_DEFAULT_POW = cast(float | None, _UNSET_NONE)
_DEFAULT_SIZE = cast(int | None, _UNSET_NONE)
_DEFAULT_PHASE = cast(str | None, _UNSET_NONE)
_DEFAULT_PHC0 = cast(float, _UNSET_ZERO)
_DEFAULT_PHC1 = cast(float, _UNSET_ZERO)


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


class ExperimentValidation:
    """Structured validation report for an NMR Experiment."""

    def __init__(self):
        self._errors: list[str] = []
        self._warnings: list[str] = []
        self._info: list[str] = []

    def add_error(self, msg: str) -> None:
        self._errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self._warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self._info.append(msg)

    @property
    def errors(self) -> list[str]:
        return list(self._errors)

    @property
    def warnings(self) -> list[str]:
        return list(self._warnings)

    @property
    def info(self) -> list[str]:
        return list(self._info)

    @property
    def is_valid(self) -> bool:
        return len(self._errors) == 0

    def __repr__(self) -> str:
        lines = []
        if self._errors:
            lines.append("Errors:")
            for e in self._errors:
                lines.append(f"  - {e}")
        if self._warnings:
            lines.append("Warnings:")
            for w in self._warnings:
                lines.append(f"  - {w}")
        if self._info:
            lines.append("Info:")
            for i in self._info:
                lines.append(f"  - {i}")
        return "\n".join(lines) if lines else "Validation passed."


# ---------------------------------------------------------------------------
# Experiment class
# ---------------------------------------------------------------------------


class Experiment:
    """
    NMR-specific scientific interpretation of an NDDataset.

    Wraps an existing dataset and provides NMR-specific classification,
    validation, and state-aware processing orchestration.  Does **not**
    copy, subclass, or mutate the underlying dataset.

    The current public processing workflow is intentionally limited to
    validated 1D experiments.  Multi-dimensional datasets may still be
    classified and inspected, but their processing remains outside the
    public supported scope until the scientific characterization work is
    complete.

    Parameters
    ----------
    dataset : NDDataset or list of NDDataset
        The NMR dataset (or list of datasets for pseudo-2D experiments)
        to interpret.

    Examples
    --------
    >>> import spectrochempy as scp
    >>> fid = scp.nmr.read(path)
    >>> experiment = scp.nmr.Experiment(fid)
    >>> experiment.summary()
    >>> spectrum = experiment.process(
    ...     apodization="em", lb=10.0, phase="manual", phc0=45.0
    ... )
    """

    def __init__(self, dataset):
        from spectrochempy.core.dataset.nddataset import NDDataset  # noqa: PLC0415

        # Accept a single NDDataset or a list of them
        if isinstance(dataset, NDDataset):
            self._datasets = [dataset]
            self._dataset = dataset
        elif isinstance(dataset, (list, tuple)):
            if not dataset:
                msg = "Experiment requires at least one dataset."
                raise ValueError(msg)
            for i, ds in enumerate(dataset):
                if not isinstance(ds, NDDataset):
                    msg = (
                        f"All items must be NDDataset instances, "
                        f"got {type(ds).__name__} at index {i}."
                    )
                    raise TypeError(msg)
            self._datasets = list(dataset)
            self._dataset = dataset[0]
        else:
            msg = (
                f"Experiment requires an NDDataset or a list of NDDatasets, "
                f"got {type(dataset).__name__}."
            )
            raise TypeError(msg)

        self._classification = self._classify()

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    @property
    def dataset(self) -> NDDataset:
        """The primary source dataset (first dataset for lists)."""
        return self._dataset

    @property
    def datasets(self) -> list[NDDataset]:
        """All datasets (useful for pseudo-2D series)."""
        return list(self._datasets)

    @property
    def is_multi_dataset(self) -> bool:
        """Whether this Experiment wraps multiple datasets."""
        return len(self._datasets) > 1

    # ------------------------------------------------------------------
    # Classification helpers
    # ------------------------------------------------------------------

    def _meta(self):
        """Return the metadata object of the primary dataset."""
        return self._dataset.meta

    def _has_meta(self) -> bool:
        """Return True if the dataset has usable NMR metadata."""
        meta = self._meta()
        return meta is not None and len(meta) > 0

    def _classify(self) -> dict:
        """
        Build a classification dict from canonical NMR metadata.

        All vendor-specific extraction happens in
        :func:`spectrochempy_nmr.nmr_metadata.extract_nmr_metadata`.
        This method consumes only the resulting
        :class:`~spectrochempy_nmr.nmr_metadata.NMRMetadata` fields.
        """
        from .nmr_metadata import NMRMetadata  # noqa: PLC0415
        from .nmr_metadata import extract_nmr_metadata  # noqa: PLC0415

        meta = self._meta()
        nmr_meta: NMRMetadata = extract_nmr_metadata(meta)

        # Fallback: when metadata is empty, infer basic shape from the
        # dataset itself.  This handles non-NMR or metadata-less datasets.
        if nmr_meta.ndim == 0 and self._dataset.ndim > 0:
            ndim = self._dataset.ndim
            domains = tuple("unknown" for _ in range(ndim))
            nmr_meta = NMRMetadata(
                ndim=ndim,
                domains=domains,
                encoding=None,
                nuclei=None,
                pulse_program=None,
                source_kind="unknown",
                datatype=None,
                iscomplex=None,
                spectral_width_hz=None,
                spectrometer_freq_mhz=None,
            )

        # Store for later use by validate() and properties.
        self._nmr_meta = nmr_meta

        cls: dict = {}
        cls["ndim"] = nmr_meta.ndim
        cls["domains"] = nmr_meta.domains
        cls["encoding"] = nmr_meta.encoding
        cls["nuclei"] = nmr_meta.nuclei
        cls["pulse_program"] = nmr_meta.pulse_program
        cls["source_kind"] = nmr_meta.source_kind
        cls["datatype"] = nmr_meta.datatype
        cls["iscomplex"] = nmr_meta.iscomplex

        # Summarised domain (vendor-neutral helper).
        from .nmr_metadata import summarise_domain  # noqa: PLC0415

        cls["domain"] = summarise_domain(nmr_meta.domains)

        return cls

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def ndim(self) -> int:
        """Number of data dimensions."""
        return self._classification["ndim"]

    @property
    def domains(self) -> tuple[str, ...]:
        """Per-dimension domain: ``'time'`` or ``'frequency'``."""
        return self._classification["domains"]

    @property
    def domain(self) -> str:
        """Summarized domain: ``'time'``, ``'frequency'``, ``'mixed'``, or ``'unknown'``."""
        return self._classification["domain"]

    @property
    def encoding(self) -> tuple[str, ...] | None:
        """Per-dimension quadrature encoding."""
        return self._classification["encoding"]

    @property
    def nuclei(self) -> tuple[str, ...] | None:
        """Observed nucleus per dimension."""
        return self._classification["nuclei"]

    @property
    def experiment_type(self) -> str | None:
        """Best-guess experiment type from pulse program (may be ``None``)."""
        return self._classification.get("pulse_program")

    @property
    def source_kind(self) -> str:
        """
        Source data classification.

        One of: ``'fid'``, ``'ser'``, ``'processed_1d'``,
        ``'processed_2d'``, ``'partially_processed'``, ``'unknown'``.
        """
        return self._classification["source_kind"]

    @property
    def datatype(self) -> str | None:
        """Reader-reported datatype (``'FID'``, ``'SER'``, ``'1D'``, ``'2D'``)."""
        return self._classification.get("datatype")

    # ---- Boolean state flags ----

    @property
    def is_time_domain(self) -> bool:
        """True if all dimensions are in time domain."""
        return self.domain == "time"

    @property
    def is_frequency_domain(self) -> bool:
        """True if all dimensions are in frequency domain."""
        return self.domain == "frequency"

    @property
    def is_mixed_domain(self) -> bool:
        """True if dimensions span both time and frequency domains."""
        return self.domain == "mixed"

    @property
    def is_raw(self) -> bool:
        """True if the data appears to be raw (unprocessed) time-domain."""
        return self.source_kind in ("fid", "ser")

    @property
    def is_processed(self) -> bool:
        """True if the data appears to be fully processed frequency-domain."""
        return self.source_kind in ("processed_1d", "processed_2d")

    @property
    def is_processable(self) -> bool:
        """
        True if the data can be meaningfully processed further.

        Time-domain data is processable (FFT, apodization, etc.).
        Frequency-domain data is processable (phasing, baseline, etc.).
        Mixed-domain and unknown data are not processable in this PR.
        """
        return self.domain in ("time", "frequency")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> ExperimentValidation:
        """
        Validate NMR-specific requirements of the dataset.

        Uses canonical :class:`~spectrochempy_nmr.nmr_metadata.NMRMetadata`
        fields — no Bruker-specific field names are referenced.

        Returns
        -------
        ExperimentValidation
            Report with ``errors``, ``warnings``, and ``info`` lists.
        """
        report = ExperimentValidation()

        if not self._has_meta():
            report.add_error("Dataset has no metadata — cannot interpret as NMR data.")
            return report

        nmr_meta = self._nmr_meta

        # --- Check essential metadata for time-domain ---
        if self.is_time_domain:
            sw = nmr_meta.spectral_width_hz
            if sw is None or not any(v is not None for v in sw):
                report.add_error(
                    "Missing spectral width — cannot construct frequency axis."
                )

            sfo = nmr_meta.spectrometer_freq_mhz
            if sfo is None or not any(v is not None for v in sfo):
                report.add_error(
                    "Missing spectrometer frequency — cannot construct ppm axis."
                )

            encoding = self.encoding
            if encoding is not None:
                unsupported = {"QSEQ"}
                for e in encoding:
                    if e in unsupported:
                        report.add_error(
                            f"Encoding '{e}' is not supported by the FFT pipeline."
                        )
            else:
                report.add_warning("No encoding information available — FFT may fail.")

            if self.nuclei is None:
                report.add_warning("No nucleus information available.")

        # --- Warnings for missing optional info ---
        if self.nuclei is None:
            report.add_warning("Nucleus unknown — frequency axis labeling unavailable.")

        if self.experiment_type is None:
            report.add_warning(
                "Pulse program unknown — experiment type cannot be inferred."
            )

        # --- Info ---
        if self.source_kind == "fid":
            report.add_info("Raw 1D FID detected.")
        elif self.source_kind == "ser":
            report.add_info("Raw 2D SER detected.")
            report.add_warning(
                "Multi-dimensional NMR processing is not part of the current "
                "public supported workflow."
            )
        elif self.source_kind == "processed_1d":
            report.add_info("Processed 1D spectrum detected — no FFT required.")
        elif self.source_kind == "processed_2d":
            report.add_info("Processed 2D spectrum detected — no FFT required.")
            report.add_warning(
                "Multi-dimensional NMR processing is not part of the current "
                "public supported workflow."
            )
        elif self.source_kind == "partially_processed":
            report.add_info("Partially processed multi-dimensional data detected.")
            report.add_warning(
                "Multi-dimensional NMR processing is not part of the current "
                "public supported workflow."
            )

        return report

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(
        self,
        *,
        apodization: str | None = _DEFAULT_APODIZATION,
        lb: float | Quantity | None = _DEFAULT_LB,
        gb: float | Quantity | None = _DEFAULT_GB,
        ssb: float | None = _DEFAULT_SSB,
        pow: float | None = _DEFAULT_POW,
        size: int | None = _DEFAULT_SIZE,
        phase: str | None = _DEFAULT_PHASE,
        phc0: float = _DEFAULT_PHC0,
        phc1: float = _DEFAULT_PHC1,
    ) -> NDDataset:
        """
        State-aware NMR processing.

        Applies only operations that are scientifically appropriate for the
        current data domain.  Never modifies the source dataset.

        The supported public processing workflow currently covers validated
        1D experiments only.

        Parameters
        ----------
        apodization : str, optional
            Apodization function name (``'em'``, ``'gm'``, ``'sp'``).
            Only accepted for time-domain data. Frequency-domain datasets
            reject explicit apodization requests.
        lb : float or Quantity, optional
            Explicit line-broadening parameter for ``'em'`` and Lorentzian
            term for ``'gm'``. If omitted, the selected core apodization
            function uses its own default.
        gb : float or Quantity, optional
            Explicit Gaussian broadening parameter for ``'gm'``. If omitted,
            the selected core apodization function uses its own default.
        ssb : float, optional
            Explicit sine-bell shift parameter for ``'sp'``. Must be positive
            when provided. If omitted, the selected core apodization function
            uses its own default.
        pow : float, optional
            Explicit exponent parameter for ``'sp'``. Only ``1`` and ``2`` are
            accepted by the public API. If omitted, the selected core
            apodization function uses its own default.
        size : int, optional
            Zero-fill target size.  Only applied to time-domain data.
        phase : str, optional
            ``'manual'`` to apply explicit ``phc0``/``phc1``,
            ``'metadata'`` to apply the dataset's current phase metadata via
            ``pk()``, or ``None`` for no phasing. ``'metadata'`` uses the
            dataset's own ``meta.phc0`` / ``meta.phc1`` state when present; it
            does not replay ``vendor_profile`` or TopSpin ``procs`` values.
        phc0 : float
            Zero-order phase correction in degrees (manual mode).
        phc1 : float
            First-order phase correction in degrees (manual mode).

        Returns
        -------
        NDDataset
            Processed dataset (copy of the source).

        Raises
        ------
        RuntimeError
            If the data domain does not support the requested operations.
        NotImplementedError
            If the dataset is multi-dimensional and therefore outside the
            current public supported processing scope.
        ValueError
            If an apodization parameter combination is incompatible with the
            selected apodization mode.
        """

        ds = self._dataset
        explicit_arguments = {
            "apodization": apodization is not _UNSET_NONE,
            "lb": lb is not _UNSET_NONE,
            "gb": gb is not _UNSET_NONE,
            "ssb": ssb is not _UNSET_NONE,
            "pow": pow is not _UNSET_NONE,
            "size": size is not _UNSET_NONE,
            "phase": phase is not _UNSET_NONE,
            "phc0": phc0 is not _UNSET_ZERO,
            "phc1": phc1 is not _UNSET_ZERO,
        }

        apodization = None if apodization is _UNSET_NONE else apodization
        lb = None if lb is _UNSET_NONE else lb
        gb = None if gb is _UNSET_NONE else gb
        ssb = None if ssb is _UNSET_NONE else ssb
        pow = None if pow is _UNSET_NONE else pow
        size = None if size is _UNSET_NONE else size
        phase = None if phase is _UNSET_NONE else phase
        phc0 = 0.0 if phc0 is _UNSET_ZERO else phc0
        phc1 = 0.0 if phc1 is _UNSET_ZERO else phc1

        if self.ndim > 1:
            msg = (
                "Public NMR processing currently supports only validated 1D "
                "experiments. Multi-dimensional NMR processing remains out of "
                "public scope pending further scientific characterization."
            )
            raise NotImplementedError(msg)

        if self.is_time_domain:
            return self._process_time_domain(
                ds,
                apodization=apodization,
                lb=lb,
                gb=gb,
                ssb=ssb,
                pow=pow,
                size=size,
                phase=phase,
                phc0=phc0,
                phc1=phc1,
                explicit_arguments=explicit_arguments,
            )
        if self.is_frequency_domain:
            self._reject_frequency_domain_apodization_requests(
                apodization,
                lb=lb,
                gb=gb,
                ssb=ssb,
                pow=pow,
            )
            return self._process_frequency_domain(
                ds,
                phase=phase,
                phc0=phc0,
                phc1=phc1,
                explicit_arguments=explicit_arguments,
            )
        msg = (
            f"Cannot process data in '{self.domain}' domain. "
            f"Current state: {' × '.join(self.domains)}"
        )
        raise RuntimeError(msg)

    def _reject_frequency_domain_apodization_requests(
        self,
        apodization: str | None,
        *,
        lb: float | Quantity | None,
        gb: float | Quantity | None,
        ssb: float | None,
        pow: float | None,
    ) -> None:
        """Reject explicit apodization requests on already transformed spectra."""
        provided = {
            "apodization": apodization,
            "lb": lb,
            "gb": gb,
            "ssb": ssb,
            "pow": pow,
        }
        non_null = [name for name, value in provided.items() if value is not None]
        if not non_null:
            return

        names = ", ".join(non_null)
        msg = (
            "Frequency-domain datasets cannot accept apodization requests in "
            f"Experiment.process(). Received: {names}. Use apodization only on "
            "time-domain FIDs, or call process() without apodization arguments "
            "to preserve the existing frequency-domain workflow."
        )
        raise RuntimeError(msg)

    def _process_time_domain(
        self,
        ds: NDDataset,
        *,
        apodization: str | None,
        lb: float | Quantity | None,
        gb: float | Quantity | None,
        ssb: float | None,
        pow: float | None,
        size: int | None,
        phase: str | None,
        phc0: float,
        phc1: float,
        explicit_arguments: dict[str, bool],
    ) -> NDDataset:
        """Process time-domain data: apodize → zero-fill → FFT → phase."""
        work = ds.copy()

        # 1. Apodization
        apodization_kwargs = self._validate_apodization_arguments(
            apodization,
            lb=lb,
            gb=gb,
            ssb=ssb,
            pow=pow,
        )
        requested = self._build_requested_trace(
            explicit_arguments,
            apodization=apodization,
            apodization_kwargs=apodization_kwargs,
            size=size,
            phase=phase,
            phc0=phc0,
            phc1=phc1,
        )
        applied: dict[str, object] = {}
        if apodization is not None:
            work = self._apply_apodization(work, apodization, **apodization_kwargs)
            applied["apodization"] = apodization.lower()
            applied.update(self._normalize_trace_mapping(apodization_kwargs))

        # 2. Zero-filling / FFT sizing
        if size is not None:
            initial_size = int(work.shape[-1])
            from spectrochempy.processing.fft.zero_filling import (  # noqa: PLC0415
                zf_size,
            )

            work = zf_size(work, size=size)
            if int(work.shape[-1]) != initial_size:
                applied["zero_filling"] = {"size": int(work.shape[-1])}

        # 3. FFT
        work = work.fft()
        applied["fft"] = True

        # 3b. Encoding-specific intermediate phase convention adjustments
        work = self._apply_default_post_fft_phase(work)

        # 4. Phase correction
        if phase is not None:
            work = self._apply_phase(work, phase, phc0=phc0, phc1=phc1)
            applied["phase"] = self._build_applied_phase_trace(
                phase,
                phc0=phc0,
                phc1=phc1,
            )

        # 5. Calibrate the final spectral axis using canonical NMR metadata.
        calibrated = self._calibrate_1d_spectral_axis(work)
        if self._axis_calibration_changed_axis(work, calibrated):
            applied["axis_calibration"] = str(calibrated.coord(0).units)
        return self._attach_scp_processing_trace(
            calibrated,
            requested=requested,
            applied=applied,
        )

    def _process_frequency_domain(
        self,
        ds: NDDataset,
        *,
        phase: str | None,
        phc0: float,
        phc1: float,
        explicit_arguments: dict[str, bool],
    ) -> NDDataset:
        """Process frequency-domain data: phase only (no re-FFT)."""
        work = ds.copy()
        requested = self._build_requested_trace(
            explicit_arguments,
            apodization=None,
            apodization_kwargs={},
            size=None,
            phase=phase,
            phc0=phc0,
            phc1=phc1,
        )
        applied: dict[str, object] = {}

        if phase is not None:
            work = self._apply_phase(work, phase, phc0=phc0, phc1=phc1)
            applied["phase"] = self._build_applied_phase_trace(
                phase,
                phc0=phc0,
                phc1=phc1,
            )

        return self._attach_scp_processing_trace(
            work,
            requested=requested,
            applied=applied,
        )

    def _build_requested_trace(
        self,
        explicit_arguments: dict[str, bool],
        *,
        apodization: str | None,
        apodization_kwargs: dict[str, float | Quantity],
        size: int | None,
        phase: str | None,
        phc0: float,
        phc1: float,
    ) -> dict[str, object]:
        """Record only the arguments explicitly provided by the user."""
        requested: dict[str, object] = {}
        if explicit_arguments["apodization"]:
            requested["apodization"] = apodization
        for name, value in apodization_kwargs.items():
            if explicit_arguments[name]:
                requested[name] = self._normalize_trace_value(name, value)
        if explicit_arguments["size"]:
            requested["size"] = int(size) if size is not None else None
        if explicit_arguments["phase"]:
            requested["phase"] = phase
        if explicit_arguments["phc0"]:
            requested["phc0"] = self._normalize_trace_value("phc0", phc0)
        if explicit_arguments["phc1"]:
            requested["phc1"] = self._normalize_trace_value("phc1", phc1)
        return requested

    def _normalize_trace_mapping(
        self,
        mapping: dict[str, float | Quantity],
    ) -> dict[str, object]:
        """Normalize trace values to a persistence-friendly public shape."""
        return {
            name: self._normalize_trace_value(name, value)
            for name, value in mapping.items()
        }

    def _normalize_trace_value(
        self,
        name: str,
        value: float | int | str | Quantity | None,
    ) -> object:
        """Normalize trace values with the public units expected by the contract."""
        if name in {"lb", "gb"} and value is not None:
            return self._as_hz_quantity(value)
        if name in {"phc0", "phc1"} and value is not None:
            return float(value) * ur.deg
        if name == "size" and value is not None:
            return int(value)
        return value

    def _as_hz_quantity(self, value: float | Quantity) -> Quantity:
        """Represent frequency-like processing values as quantities in Hz."""
        if isinstance(value, Quantity):
            return value.to(ur.Hz)
        return float(value) * ur.Hz

    def _build_applied_phase_trace(
        self,
        mode: str,
        *,
        phc0: float,
        phc1: float,
    ) -> dict[str, object]:
        """Record only the phase information actually consumed by the phase step."""
        phase_trace: dict[str, object] = {"mode": mode}
        if mode == "manual":
            phase_trace["phc0"] = self._normalize_trace_value("phc0", phc0)
            phase_trace["phc1"] = self._normalize_trace_value("phc1", phc1)
        return phase_trace

    def _axis_calibration_changed_axis(
        self, before: NDDataset, after: NDDataset
    ) -> bool:
        """Return True when the final calibration step changed the public axis."""
        before_coord = before.coord(0)
        after_coord = after.coord(0)
        if str(before_coord.units) != str(after_coord.units):
            return True
        if before_coord.title != after_coord.title:
            return True
        return not np.array_equal(
            np.asarray(before_coord.data), np.asarray(after_coord.data)
        )

    def _attach_scp_processing_trace(
        self,
        ds: NDDataset,
        *,
        requested: dict[str, object],
        applied: dict[str, object],
    ) -> NDDataset:
        """Attach the SpectroChemPy processing trace to the result dataset only."""
        readonly = ds.meta.readonly
        ds.meta.readonly = False

        existing = getattr(ds.meta, "nmr_processing", None)
        nmr_processing = dict(existing) if existing is not None else {}
        nmr_processing["observed_state"] = {
            "processing_history": "spectrochempy_process_recorded"
        }
        nmr_processing["scp_processing"] = {
            "requested": requested,
            "applied": applied,
        }
        ds.meta["nmr_processing"] = nmr_processing
        ds.meta.readonly = readonly
        return ds

    def _validate_apodization_arguments(
        self,
        apodization: str | None,
        *,
        lb: float | Quantity | None,
        gb: float | Quantity | None,
        ssb: float | None,
        pow: float | None,
    ) -> dict[str, float | Quantity]:
        """Validate the explicit public apodization contract."""
        provided = {
            "lb": lb,
            "gb": gb,
            "ssb": ssb,
            "pow": pow,
        }
        non_null = {
            name: value for name, value in provided.items() if value is not None
        }

        if apodization is None:
            if non_null:
                names = ", ".join(sorted(non_null))
                msg = (
                    "Explicit apodization parameters require an apodization mode. "
                    f"Received parameters without apodization: {names}."
                )
                raise ValueError(msg)
            return {}

        func_name = apodization.lower()
        if func_name not in _APODIZATION_ALLOWED_PARAMS:
            msg = (
                f"Unknown apodization function: {func_name!r}. Use 'em', 'gm', or 'sp'."
            )
            raise ValueError(msg)

        allowed = set(_APODIZATION_ALLOWED_PARAMS[func_name])
        invalid = sorted(name for name in non_null if name not in allowed)
        if invalid:
            names = ", ".join(invalid)
            msg = (
                f"Apodization {func_name!r} does not accept parameter(s): {names}. "
                f"Allowed parameters: {', '.join(_APODIZATION_ALLOWED_PARAMS[func_name])}."
            )
            raise ValueError(msg)

        validated: dict[str, float | Quantity] = {}
        for name in _APODIZATION_ALLOWED_PARAMS[func_name]:
            value = provided[name]
            if value is None:
                continue
            validated[name] = self._validate_apodization_parameter(
                func_name, name, value
            )
        return validated

    def _validate_apodization_parameter(
        self,
        func_name: str,
        param_name: str,
        value: float | Quantity,
    ) -> float | Quantity:
        """Validate one explicit apodization parameter."""
        if param_name in {"lb", "gb"}:
            return self._validate_frequency_like_parameter(func_name, param_name, value)
        if param_name == "ssb":
            if isinstance(value, Quantity):
                msg = f"Parameter 'ssb' for apodization {func_name!r} must be a plain scalar."
                raise TypeError(msg)
            if not isinstance(value, Real):
                msg = f"Parameter 'ssb' for apodization {func_name!r} must be a real scalar."
                raise TypeError(msg)
            magnitude = float(value)
            if not np.isfinite(magnitude):
                msg = f"Parameter 'ssb' for apodization {func_name!r} must be finite."
                raise ValueError(msg)
            if magnitude <= 0.0:
                msg = "Parameter 'ssb' for apodization 'sp' must be strictly positive."
                raise ValueError(msg)
            return magnitude
        if param_name == "pow":
            if isinstance(value, Quantity):
                msg = f"Parameter 'pow' for apodization {func_name!r} must be a plain scalar."
                raise TypeError(msg)
            if not isinstance(value, Real):
                msg = f"Parameter 'pow' for apodization {func_name!r} must be a real scalar."
                raise TypeError(msg)
            magnitude = float(value)
            if not np.isfinite(magnitude):
                msg = f"Parameter 'pow' for apodization {func_name!r} must be finite."
                raise ValueError(msg)
            if magnitude not in (1.0, 2.0):
                msg = "Parameter 'pow' for apodization 'sp' must be 1 or 2."
                raise ValueError(msg)
            return int(magnitude)

        msg = f"Unsupported apodization parameter {param_name!r}."
        raise ValueError(msg)

    def _validate_frequency_like_parameter(
        self,
        func_name: str,
        param_name: str,
        value: float | Quantity,
    ) -> float | Quantity:
        """Validate frequency-like explicit apodization parameters."""
        if isinstance(value, Quantity):
            if value.dimensionality != ur.Hz.dimensionality:
                msg = (
                    f"Parameter {param_name!r} for apodization {func_name!r} must "
                    "have frequency units compatible with Hz."
                )
                raise ValueError(msg)
            magnitude = float(value.magnitude)
        else:
            if not isinstance(value, Real):
                msg = (
                    f"Parameter {param_name!r} for apodization {func_name!r} must "
                    "be a real scalar or a Quantity."
                )
                raise TypeError(msg)
            magnitude = float(value)

        if not np.isfinite(magnitude):
            msg = f"Parameter {param_name!r} for apodization {func_name!r} must be finite."
            raise ValueError(msg)
        return value

    def _apply_apodization(self, ds: NDDataset, func_name: str, **kwargs) -> NDDataset:
        """Apply an apodization function by name."""
        func_name = func_name.lower()
        from spectrochempy.processing.fft.apodization import em  # noqa: PLC0415
        from spectrochempy.processing.fft.apodization import gm  # noqa: PLC0415
        from spectrochempy.processing.fft.apodization import sp  # noqa: PLC0415

        apodizers = {
            "em": em,
            "gm": gm,
            "sp": sp,
        }
        if func_name not in apodizers:
            msg = (
                f"Unknown apodization function: {func_name!r}. Use 'em', 'gm', or 'sp'."
            )
            raise ValueError(msg)
        return apodizers[func_name](ds, **kwargs)

    def _apply_phase(
        self, ds: NDDataset, mode: str, *, phc0: float, phc1: float
    ) -> NDDataset:
        """Apply phase correction."""
        from spectrochempy.processing.fft.phasing import pk  # noqa: PLC0415

        # Ensure phased metadata exists — processed data from readers
        # may have phased=None, which pk() cannot handle.
        # The reader sets meta.readonly=True; unlock before patching.
        work = ds.copy()
        work.meta.readonly = False
        if getattr(work.meta, "phased", None) is None:
            work.meta["phased"] = [False] * work.ndim
        if getattr(work.meta, "phc0", None) is None:
            work.meta["phc0"] = [0.0] * work.ndim
        if getattr(work.meta, "phc1", None) is None:
            work.meta["phc1"] = [0.0] * work.ndim
        if getattr(work.meta, "exptc", None) is None:
            work.meta["exptc"] = [0.0] * work.ndim
        if getattr(work.meta, "pivot", None) is None:
            work.meta["pivot"] = [0.0] * work.ndim

        if mode == "manual":
            return pk(work, phc0=phc0, phc1=phc1, rel=True)
        if mode == "metadata":
            return pk(work)
        msg = f"Unknown phase mode: {mode!r}. Use 'manual' or 'metadata'."
        raise ValueError(msg)

    def _apply_default_post_fft_phase(self, ds: NDDataset) -> NDDataset:
        """
        Apply encoding-specific convention fixes after the first FFT pass.

        For 2D Bruker Echo-Antiecho data, a -90° zero-order phase on the
        direct-dimension spectrum provides the correct intermediate convention
        before the second transform along F1. Without this step, the final real
        spectrum remains in quadrature relative to the TopSpin processed
        reference even though the magnitude peak is correctly positioned.
        """
        encoding = self.encoding or ()
        if self.ndim >= 2 and "ECHO-ANTIECHO" in encoding:
            return self._apply_phase(ds, "manual", phc0=-90.0, phc1=0.0)
        return ds

    def _calibrate_1d_spectral_axis(self, ds: NDDataset) -> NDDataset:
        """
        Normalize the final 1D frequency axis to ppm when possible.

        Some readers preserve enough information for ``fft()`` to create a
        frequency-domain axis in Hz but not to complete the final ppm
        calibration automatically.  The canonical NMR metadata already carries
        the spectrometer frequency and nucleus information, so finalize the
        public 1D output here in a vendor-independent way.
        """
        if ds.ndim != 1:
            return ds

        coord = ds.coord(0)
        sfo = self._nmr_meta.spectrometer_freq_mhz
        sw_hz = self._nmr_meta.spectral_width_hz
        nuclei = self._nmr_meta.nuclei
        if not sfo or sfo[0] is None:
            return ds

        work = ds.copy()
        work.meta.readonly = False
        coord = work.coord(0)
        from spectrochempy.core.units import ur  # noqa: PLC0415

        origin = getattr(self.dataset, "origin", None)
        if (
            origin in {"agilent", "jeol", "tecmag", "simpson"}
            and self.source_kind == "fid"
            and sw_hz
            and sw_hz[0] is not None
            and coord.size > 1
        ):
            offset_ppm = 0.0
            raw_offset = getattr(self.dataset.meta, "offset", None)
            if raw_offset and raw_offset[0] is not None:
                offset_ppm = float(raw_offset[0])

            ppm_width = float(sw_hz[0]) / float(sfo[0])
            sizem = max(coord.size - 1, 1)
            delta_ppm = -ppm_width / sizem
            first_ppm = offset_ppm - delta_ppm * sizem / 2.0
            ppm_axis = np.arange(coord.size, dtype=float) * delta_ppm + first_ppm

            from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415

            newcoord = Coord(ppm_axis, units="ppm")
            newcoord.meta["acquisition_frequency"] = float(sfo[0]) * ur.MHz
            work.set_coordset(newcoord)
            coord = newcoord
        else:
            if str(coord.units) == "ppm":
                return work
            coord.meta["acquisition_frequency"] = float(sfo[0]) * ur.MHz
            coord.ito("ppm")

        if nuclei and nuclei[0]:
            nucleus = str(nuclei[0])
            match = re.match(r"([^a-zA-Z]+)([a-zA-Z]+)", nucleus)
            nucleus_label = rf"$^{{{match[1]}}}{match[2]}$" if match else nucleus
            coord.title = rf"$\delta\ {nucleus_label}$"

        return work

    # ------------------------------------------------------------------
    # Summary and representation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a concise human-readable summary of the experiment."""
        lines = ["NMR Experiment"]
        lines.append(f"  dimensions: {self.ndim}")
        lines.append(f"  source kind: {self.source_kind}")
        lines.append(f"  domain: {' × '.join(self.domains)}")
        if self.nuclei:
            lines.append(f"  nuclei: {' × '.join(self.nuclei)}")
        if self.encoding:
            lines.append(f"  encoding: {' × '.join(self.encoding)}")
        lines.append(f"  processable: {'yes' if self.is_processable else 'no'}")
        if self.ndim >= 2:
            lines.append("  public processing: 1D only")
        return "\n".join(lines)

    def __repr__(self) -> str:
        kind = self.source_kind
        doms = " × ".join(self.domains)
        nucs = " × ".join(self.nuclei) if self.nuclei else "?"
        return (
            f"Experiment(kind={kind!r}, domain={doms!r}, "
            f"nuclei={nucs!r}, ndim={self.ndim})"
        )
