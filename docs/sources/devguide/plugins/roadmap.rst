.. _plugin-dev-roadmap:

=======================
Plugin architecture map
=======================

The long-term direction is a small domain-neutral core with official plugins
for domain-specific readers, processing conventions, simulations, and optional
numeric backends.

Completed decoupling
====================

* NMR unit contexts live in the NMR plugin.
* Hypercomplex/quaternion support lives in ``spectrochempy-hypercomplex``.
* TensorLy-backed tensor decompositions live in ``spectrochempy-tensor``.
* 2D NMR FFT encodings are delegated through ``fft.encoding``.
* Directory-to-file resolution for plugin formats is delegated through
  ``importer.resolve_directory_target``.
* Extensionless filetype inference is delegated through
  ``importer.infer_filetype_key``.
* Remote parent-directory download rules are delegated through
  ``importer.remote_download_target``.
* Five NMR readers are registered: TopSpin, Agilent, JEOL, TecMag, SIMPSON.
* The ``Experiment`` class provides the high-level NMR scientific API
  (classification, validation, state-aware processing).

NMR high-level API direction
============================

The ``Experiment`` class is the chosen high-level NMR API layer::

    dataset = scp.nmr.read(path)
    experiment = scp.nmr.Experiment(dataset)
    spectrum = experiment.process(lb=10.0, phase="manual", phc0=45.0)

Dataset accessors such as ``dataset.nmr.phase(...)`` or
``dataset.nmr.apodize(...)`` are **not** part of the planned API. Low-level
operations already exist on ``NDDataset`` (``dataset.em(...)``,
``dataset.gm(...)``, ``dataset.pk(...)``).  The NMR-specific orchestration
and scientific interpretation belong exclusively to ``Experiment``.

This avoids two concurrent high-level APIs with unclear responsibility
boundaries.

NMR roadmap (priority order)
============================

1. **Cross-vendor Experiment contract** — Complete canonical metadata
   extraction for every supported reader.  Currently
   ``Experiment._classify()`` dispatches only to the TopSpin extractor,
   which silently misclassifies non-Bruker datasets.  The same canonical
   ``NMRMetadata`` fields must carry the same meaning across all five
   readers.

2. **2D NMR processing** — Implement full 2D transformation (quaternion →
   complex for DQD/QSIM, indirect-dimension FFT).  This is the major
   functional limitation today.

3. **Phasing boundary clarification** — Decide whether ``pk``/``pk_exp``
   metadata orchestration migrates from core to the NMR plugin, or whether
   the generic numerical kernel stays in core while Experiment provides the
   NMR-specific workflow.

4. **QSEQ encoding** — Implement when real-world data justifies it.

5. **Multi-dataset workflows** — T1/T2 relaxation, DOSY diffusion,
   kinetics, and chemometric processing.  These are the main scientific
   differentiators for SpectroChemPy NMR.

Remaining boundaries
====================

``processing/fft/phasing.py`` still contains NMR metadata conventions around
``phc0``, ``phc1``, ``pivot`` and ``exptc``. The numerical phase kernel is
generic, but the public metadata update behavior should be clarified as part
of the phasing boundary decision (item 3 above).

``processing/fft/shift.py`` is retained in core because its implementation is
generic signal processing. NMR-specific aliases or defaults can be added in the
NMR plugin if needed.

Documentation direction
=======================

User documentation should describe installation, workflows, examples, and
official plugin behavior. Developer documentation should describe hooks,
registries, handlers, accessors, tests, packaging, and migration plans.
