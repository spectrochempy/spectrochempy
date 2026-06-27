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
* NMR users can install full 2D support on demand;
* the hypercomplex backend can evolve independently;
* other scientific domains can reuse the same mechanism if needed.

.. _hypercomplex-install:

Installation
============

Install the plugin directly or through the NMR extra:

.. code-block:: bash

    pip install spectrochempy-hypercomplex

    # or, together with the NMR plugin
    pip install spectrochempy[nmr,hypercomplex]

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
    import numpy as np

    c = scp.NDDataset(np.array([1+2j, 3+4j]))

A **hypercomplex** dataset stores *two* complex numbers per point,
typically written as a quaternion:

.. math::

    q = w + x\,i + y\,j + z\,k

In 2D NMR this corresponds to four real arrays: RR, RI, IR, II
(Real-Real, Real-Imaginary, Imaginary-Real, Imaginary-Imaginary).

The core SpectroChemPy package understands ordinary complex data natively.
Hypercomplex data requires the plugin. Once installed, the plugin provides the
``dataset.hyper`` accessor and enables supported math and 2D NMR operations on
quaternion arrays.

.. _hypercomplex-api:

API Reference
=============

All hypercomplex operations are accessed through ``dataset.hyper``:

.. list-table::
   :header-rows: 1

   * - Operation
     - Example
   * - Convert to quaternion
     - ``dataset.hyper.set_quaternion(inplace=True)``
   * - Check type
     - ``dataset.hyper.is_quaternion``
   * - Extract RR component
     - ``dataset.hyper.RR``
   * - Extract RI component
     - ``dataset.hyper.component("RI")``
   * - Extract IR component
     - ``dataset.hyper.IR``
   * - Extract II component
     - ``dataset.hyper.II``

.. _hypercomplex-nmr-example:

Examples
========

After reading a 2D TopSpin dataset, the NMR plugin can optionally
convert the data to hypercomplex form (this happens automatically when
both the NMR and hypercomplex plugins are installed):

.. code-block:: python

    import spectrochempy as scp

    # Requires spectrochempy-nmr and spectrochempy-hypercomplex
    dataset = scp.nmr.read_topspin("path/to/ser", expno=1)

    # The NMR reader may already have called set_quaternion;
    # if not, you can do it explicitly:
    if not dataset.hyper.is_quaternion:
        dataset.hyper.set_quaternion(inplace=True)

    # Now fft() understands STATES / TPPI / ECHO-ANTIECHO encodings
    spectrum = dataset.fft()

    # Extract a component for display
    rr = spectrum.hyper.RR
    rr.plot(method="map")

.. _hypercomplex-future:

Limitations and scope
=====================

The hypercomplex plugin is intentionally narrow at this stage:

* it supports quaternion data in ``NDDataset``;
* it provides the numeric hooks needed by the core math framework;
* it enables 2D NMR phase-sensitive workflows.
