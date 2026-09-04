.. _plugin-numeric-backends:

================
Numeric backends
================

The core numeric path handles ordinary NumPy arrays and ordinary complex
numbers. Plugins can provide additional execution branches for specialized
array types without adding those dependencies to the core.

The hypercomplex plugin uses this pattern for quaternion arrays. A plugin can
register:

* ``ndmath.execution_branch`` to identify a plugin-owned numeric branch;
* ``ndmath.execute`` to run an operation for that branch;
* ``ndmath.numpy_method.<name>`` to override selected NumPy-like methods;
* ``fft.encoding`` for domain-specific FFT acquisition encodings.

Example:

.. code-block:: python

    def register_handlers(self) -> dict:
        return {
            "ndmath.execution_branch": execution_branch,
            "ndmath.execute": execute,
        }

Handlers should return ``None`` when they do not recognize the data so the
core default path remains available.
