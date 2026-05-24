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
* 2D NMR FFT encodings are delegated through ``fft.encoding``.
* Directory-to-file resolution for plugin formats is delegated through
  ``importer.resolve_directory_target``.
* Extensionless filetype inference is delegated through
  ``importer.infer_filetype_key``.
* Remote parent-directory download rules are delegated through
  ``importer.remote_download_target``.

Remaining boundaries
====================

``processing/fft/phasing.py`` still contains NMR metadata conventions around
``phc0``, ``phc1``, ``pivot`` and ``exptc``. The numerical phase kernel is
generic, but the public metadata update behavior should move only when the NMR
plugin provides replacement ``pk`` and ``pk_exp`` accessors.

``processing/fft/shift.py`` is retained in core because its implementation is
generic signal processing. NMR-specific aliases or defaults can be added in the
NMR plugin if needed.

Documentation direction
=======================

User documentation should describe installation, workflows, examples, and
official plugin behavior. Developer documentation should describe hooks,
registries, handlers, accessors, tests, packaging, and migration plans.
