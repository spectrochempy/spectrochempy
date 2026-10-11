.. _hypercomplex-plugin:

================================
Hypercomplex (quaternion) plugin
================================

Introduction
============

The ``spectrochempy-hypercomplex`` plugin extends SpectroChemPy with
quaternion/hypercomplex data support. It is designed for scientific
domains that need complex numbers in more than one dimension, most
commonly **phase-sensitive 2D NMR**.

.. _hypercomplex-why-plugin:

Why a plugin?
=============

Hypercomplex data is powerful but niche. Keeping it in the core would:

* add a heavy optional dependency (``numpy-quaternion``) to every
  SpectroChemPy installation;
* embed NMR-specific assumptions into generic dataset infrastructure;
* make the core harder to maintain and test for the majority of users
  who only need ordinary complex numbers.

By extracting hypercomplex support into an official plugin:

* the core stays lightweight and domain-neutral;
* NMR users can install the hypercomplex *representation* used in
  phase-sensitive 2D NMR on demand;
* the hypercomplex backend can evolve independently;
* other scientific domains can reuse the same mechanism if needed.

.. _hypercomplex-install:

Installation
============

Install the plugin directly or through the NMR extra:

.. code-block:: bash

    python -m pip install spectrochempy-hypercomplex

    # or, together with the NMR plugin
    python -m pip install "spectrochempy[nmr]" spectrochempy-hypercomplex

The plugin is discovered automatically once installed. No explicit loading
step is required.

Recommended API
===============

The recommended public API is the ``dataset.hyper`` accessor:

.. code-block:: python

    dataset.hyper.set_quaternion(inplace=True)
    rr = dataset.hyper.RR
    ri = dataset.hyper.component("RI")

.. _hypercomplex-concepts:

Ordinary complex vs hypercomplex
=================================

An ordinary complex dataset stores one complex number per point:

.. code-block:: python

    import spectrochempy as scp

    c = scp.NDDataset([1+2j, 3+4j])

A **hypercomplex** dataset stores *two* complex numbers per point,
typically written as a quaternion:

.. math::

    q = w + x\,i + y\,j + z\,k

In 2D NMR this corresponds to four real arrays: RR, RI, IR, II
(Real-Real, Real-Imaginary, Imaginary-Real, Imaginary-Imaginary).

The core SpectroChemPy package understands ordinary complex data natively.
Hypercomplex data requires the plugin. Once installed, the plugin provides the
``dataset.hyper`` accessor and enables quaternion-aware math and display on
such datasets.

.. _hypercomplex-api:

API Reference
=============

The public API reference for the hypercomplex plugin is listed in
:doc:`/reference/plugins`.

.. _hypercomplex-nmr-example:

Examples
========

After reading a 2D TopSpin dataset, the NMR reader stores the two quadrature
pairs as a hypercomplex (quaternion) array when both the NMR and hypercomplex
plugins are installed. This is a *representation*: the four components
``RR``, ``RI``, ``IR``, ``II`` are kept explicitly instead of being collapsed
into a single complex spectrum.

.. code-block:: python

    import spectrochempy as scp

    # Requires spectrochempy-nmr and spectrochempy-hypercomplex
    dataset = scp.nmr.read("path/to/ser", expno=1)

    # The NMR reader may already have called set_quaternion;
    # if not, you can do it explicitly:
    if not dataset.hyper.is_quaternion:
        dataset.hyper.set_quaternion(inplace=True)

    # Extract one component (a plain numpy array); rebuild a dataset
    # with the acquisition coordinates for display.
    rr = dataset.hyper.RR
    rr = scp.NDDataset(rr, coordset=dataset.coordset, dims=dataset.dims)
    rr.plot(method="map")

The ``dataset.hyper.RR``, ``RI``, ``IR`` and ``II`` accessors extract the
corresponding real components, and ``component()`` selects them by name. This
reading-and-representation scope is validated for 1D and 2D hypercomplex NMR
data, including raw ``ser`` time-domain data and vendor-processed 2D spectra.

.. _hypercomplex-future:

Limitations and scope
=====================

The hypercomplex plugin is intentionally narrow at this stage:

* it supports quaternion data in ``NDDataset``;
* it provides the numeric hooks needed by the core math framework;
* it enables representation and component extraction for phase-sensitive 2D NMR.

Public NMR *processing* remains deliberately limited to validated 1D
experiments: ``scp.nmr.Experiment(...).process()`` raises
``NotImplementedError`` for datasets with more than one dimension. The
low-level transforms used to characterize multi-dimensional encodings (such as
``fft()`` and the apodization functions) exist at the array level, but they do
not form a supported public 2D processing chain, and a single ``dataset.fft()``
call is **not** a complete 2D processing recipe. Pseudo-2D series — a list of
ordinary 1D spectra sharing a secondary coordinate (for example a relaxation or
kinetics series) — must also be kept distinct from a genuine 2D experiment in
which both dimensions are Fourier-transformed.
