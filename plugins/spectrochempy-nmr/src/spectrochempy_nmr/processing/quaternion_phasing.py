"""
Quaternion-aware phasing for 2D NMR hypercomplex data.

The built-in ``pk()`` returns a 1-D complex apodization vector and applies
it via ``dataset *= apod``.  This does not work for quaternion arrays
(quaternion × complex is not defined element-wise).

This module provides a ``pk.execute`` handler that:
  1. Decomposes the quaternion into two complex subspectra
     (``fr = RR + j*RI``, ``fi = IR + j*II``).
  2. Multiplies each subspectro by the phase apodization independently.
  3. Rebuilds the quaternion from the phased subspectra.

The handler is registered by the NMR plugin and dispatched from the
``_phase_method`` decorator in ``phasing.py``.
"""

from __future__ import annotations


def quaternion_pk_handler(dataset, **kwargs):
    """
    Phase quaternion data by applying correction to each subspectro.

    Modifies the dataset **in place** (data array replaced, metadata
    updated by the caller).

    Parameters
    ----------
    dataset : NDDataset
        The dataset with quaternion data.  The target axis must already
        be swapped to position -1 (done by the ``_phase_method`` decorator).
    apod : ndarray
        1-D complex apodization vector from the ``pk`` kernel.
    **kwargs
        Forwarded from ``pk()``; unused here.

    Returns
    -------
    NDDataset
        The same dataset, with phased data.
    """
    from spectrochempy_nmr.processing.hypercomplex import _extract_quaternion_components
    from spectrochempy_nmr.processing.hypercomplex import _rebuild_quaternion

    apod = kwargs.get("apod")
    if apod is None:
        return None

    RR, RI, IR, II = _extract_quaternion_components(dataset.data)
    fr = RR + 1j * RI
    fi = IR + 1j * II

    fr_phased = fr * apod
    fi_phased = fi * apod

    dataset.data = _rebuild_quaternion(fr_phased, fi_phased)
    return dataset
