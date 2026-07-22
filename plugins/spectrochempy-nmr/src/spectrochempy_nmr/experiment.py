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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from spectrochempy.core.dataset.nddataset import NDDataset


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
    >>> spectrum = experiment.process(lb=10.0, phase="manual", phc0=45.0)
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
        apodization: str | None = None,
        lb: float = 1.0,
        size: int | None = None,
        phase: str | None = None,
        phc0: float = 0.0,
        phc1: float = 0.0,
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
            Only applied to time-domain data.  Ignored for frequency-domain.
        lb : float
            Line broadening in Hz (for ``apodization='em'``).  Default 1.0.
        size : int, optional
            Zero-fill target size.  Only applied to time-domain data.
        phase : str, optional
            ``'manual'`` to apply ``phc0``/``'phc1``, ``'metadata'`` to use
            stored phase values, or ``None`` for no phasing.
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
        """

        ds = self._dataset

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
                size=size,
                phase=phase,
                phc0=phc0,
                phc1=phc1,
            )
        if self.is_frequency_domain:
            return self._process_frequency_domain(
                ds,
                phase=phase,
                phc0=phc0,
                phc1=phc1,
            )
        msg = (
            f"Cannot process data in '{self.domain}' domain. "
            f"Current state: {' × '.join(self.domains)}"
        )
        raise RuntimeError(msg)

    def _process_time_domain(
        self,
        ds: NDDataset,
        *,
        apodization: str | None,
        lb: float,
        size: int | None,
        phase: str | None,
        phc0: float,
        phc1: float,
    ) -> NDDataset:
        """Process time-domain data: apodize → zero-fill → FFT → phase."""
        work = ds.copy()

        # 1. Apodization
        if apodization is not None:
            work = self._apply_apodization(work, apodization, lb=lb)

        # 2. Zero-filling / FFT sizing
        if size is not None:
            from spectrochempy.processing.fft.zero_filling import (  # noqa: PLC0415
                zf_size,
            )

            work = zf_size(work, size=size)

        # 3. FFT
        work = work.fft()

        # 3b. Encoding-specific intermediate phase convention adjustments
        work = self._apply_default_post_fft_phase(work)

        # 4. Phase correction
        if phase is not None:
            work = self._apply_phase(work, phase, phc0=phc0, phc1=phc1)

        # 5. Calibrate the final spectral axis using canonical NMR metadata.
        return self._calibrate_1d_spectral_axis(work)

    def _process_frequency_domain(
        self,
        ds: NDDataset,
        *,
        phase: str | None,
        phc0: float,
        phc1: float,
    ) -> NDDataset:
        """Process frequency-domain data: phase only (no re-FFT)."""
        work = ds.copy()

        if phase is not None:
            work = self._apply_phase(work, phase, phc0=phc0, phc1=phc1)

        return work

    def _apply_apodization(
        self, ds: NDDataset, func_name: str, *, lb: float
    ) -> NDDataset:
        """Apply an apodization function by name."""
        func_name = func_name.lower()
        if func_name == "em":
            from spectrochempy.processing.fft.apodization import em  # noqa: PLC0415

            return em(ds, lb=lb)
        if func_name == "gm":
            from spectrochempy.processing.fft.apodization import gm  # noqa: PLC0415

            return gm(ds, lb=lb)
        if func_name == "sp":
            from spectrochempy.processing.fft.apodization import sp  # noqa: PLC0415

            return sp(ds)
        msg = f"Unknown apodization function: {func_name!r}. Use 'em', 'gm', or 'sp'."
        raise ValueError(msg)

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
        if str(coord.units) == "ppm":
            return ds

        sfo = self._nmr_meta.spectrometer_freq_mhz
        nuclei = self._nmr_meta.nuclei
        if not sfo or sfo[0] is None:
            return ds

        work = ds.copy()
        work.meta.readonly = False
        coord = work.coord(0)
        from spectrochempy.core.units import ur  # noqa: PLC0415

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
