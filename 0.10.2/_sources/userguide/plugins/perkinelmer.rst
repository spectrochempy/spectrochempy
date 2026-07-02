.. _perkinelmer-plugin:

==================
PerkinElmer plugin
==================

The ``spectrochempy-perkinelmer`` plugin provides a reader for PerkinElmer
``.sp`` binary IR files.

Install it with:

.. code-block:: bash

    pip install spectrochempy[perkinelmer]

Use the namespaced API ``scp.perkinelmer``:

.. code-block:: python

    import spectrochempy as scp

    dataset = scp.perkinelmer.read("path/to/file.sp")

Compatibility aliases are also available:

.. code-block:: python

    dataset = scp.read_perkinelmer("path/to/file.sp")
    dataset = scp.read_sp("path/to/file.sp")

Limitations
===========

- Only single-spectrum ``.sp`` files are supported.
- The ``.prf`` format is not supported.
- Metadata extraction depends on the presence of standard PerkinElmer blocks;
  files with incomplete metadata will still load but with reduced ``meta``
  information.
