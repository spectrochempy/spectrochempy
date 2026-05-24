.. _plugin-accessors:

================
Plugin accessors
================

Accessors expose plugin operations on existing SpectroChemPy objects. For an
``NDDataset``, use an accessor only when the dataset is a real input to the
operation:

.. code-block:: python

    result = dataset.iris.kernel_matrix(kernel_type="langmuir")
    dataset.hyper.set_quaternion(inplace=True)

Do not use accessors for readers, constructors, or simulations that create
objects independently. Those belong under package-level namespaces:

.. code-block:: python

    dataset = scp.nmr.read_topspin("path/to/fid")
    reactor = scp.cantera.PFR

Accessor contributions are declared with ``register_accessors()``. Prefer a
namespace and a short operation name:

.. code-block:: python

    def register_accessors(self) -> list[dict]:
        return [
            {
                "namespace": "domain",
                "name": "operation",
                "func": operation,
                "description": "Run a domain operation",
            },
        ]

Legacy flat names may be declared with ``legacy_names`` during a transition,
but new documentation should show only the namespaced form.
