.. _plugin-numeric-backends:

================
Numeric backends
================

The core numeric path handles ordinary NumPy arrays and ordinary complex
numbers. Plugins can provide additional execution branches for specialized
array types without adding those dependencies to the core.

A plugin can register:

* ``ndmath.execution_branch`` to identify a plugin-owned numeric branch;
* ``ndmath.execute`` to run an operation for that branch;
* ``ndmath.numpy_method.<name>`` to override selected NumPy-like methods;
* ``fft.encoding`` for domain-specific FFT acquisition encodings.
* ``fft.validate_input`` to reject unsupported FFT/IFFT requests before
  preparation can modify the dataset.

Example:

.. code-block:: python

    def register_handlers(self) -> dict:
        return {
            "ndmath.execution_branch": execution_branch,
            "ndmath.execute": execute,
        }

Handlers should return ``None`` when they do not recognize the data so the
core default path remains available.

FFT input validation
--------------------

``fft.validate_input(dataset, *, dim, inv, size, sizeff, **kwargs)`` receives
the original dataset and a resolved dimension name. It runs after the core's
domain checks, before copying, swapping axes, effective-size zeroing,
zero-filling, truncation, numerical transformation or coordinate updates.
The original keyword options, including ``inplace``, are available in
``kwargs``. This applies to method and functional FFT/IFFT entry points.

Validators must treat the dataset as read-only. Return ``None`` to accept
the request and allow other registered validators to run, or raise an
exception to reject it. Exceptions propagate to the caller before preparation;
this is not a rollback mechanism for changes made by a validator itself.
With no registered validator, the ordinary core path is unchanged.

Plugins can detect this contract using ``FFT_PREVALIDATION_API >= 1`` in
``spectrochempy.processing.fft.fft``. Older cores do not expose that marker;
plugins requiring prevalidation must refuse the affected workflow on them.
