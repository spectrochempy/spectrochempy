.. _userguide.provenance:

Experimental structured provenance
==================================

SpectroChemPy provides an experimental, opt-in substrate for machine-readable
operation records under :mod:`spectrochempy.provenance`. It is separate from
the human-readable
:attr:`NDDataset.history <spectrochempy.NDDataset.history>`, whose representation and behavior are
unchanged.

This substrate is under active development for a post-1.0 provenance campaign
and is assembled on the experimental ``develop`` branch. It is not part of the
SpectroChemPy 1.0 release and its contract may still evolve.

P1 establishes immutable references and records, an append-only ledger, and a
context-local capture mechanism. P2 adds the first runtime instrumentation:
out-of-place ``NDDataset`` slicing and ``NDDataset.transpose`` create one
operation record each while a capture context is active. Every other scientific
operation remains uninstrumented and creates no record.

Captured slicing and transpose
------------------------------

Capture is opt-in. Outside an active :class:`~spectrochempy.provenance.ProvenanceCapture`,
slicing and transpose behave exactly as before and create no record:

.. code-block:: python

   import numpy as np

   import spectrochempy as scp

   dataset = scp.NDDataset(np.arange(24.0).reshape(4, 6))

   with scp.provenance.ProvenanceCapture() as capture:
       selection = dataset[:, 1:3]
       transposed = selection.transpose()

   assert len(capture.ledger) == 2

The first record references the source dataset state and creates a new object
and initial state for the selection; the second references the selection state
as its input and creates another new object. Objects are tracked transiently
through weak references and are released when the context closes. The ledger
stores only immutable records and opaque ledger-local references, never the
datasets, arrays, or their lineage graphs.

The selection is captured as a JSON-safe structural description (``slice``,
``ellipsis``, and index values), and transpose records the requested dimension
order. Slicing and transpose never mutate the source, and the textual
:attr:`~spectrochempy.NDDataset.history` produced by the operation is
unchanged. This first slice covers only single-source out-of-place operations;
in-place operations, arithmetic, concatenation, estimators, and readers are not
instrumented.

Explicit record construction
----------------------------

Records can also be constructed and appended explicitly for development and
schema evaluation:

.. code-block:: python

   from datetime import UTC, datetime

   import spectrochempy as scp

   provenance = scp.provenance
   source_object = provenance.ObjectRef("object-000001")
   source_state = provenance.StateRef("state-000001", source_object, version=0)
   result_object = provenance.ObjectRef("object-000002")
   result_state = provenance.StateRef("state-000002", result_object, version=0)

   record = provenance.OperationRecord(
       id="op-000001",
       operation_id="org.example.preprocessing.center",
       category="transform",
       implementation="example.preprocessing.center",
       provider_name="example-provider",
       provider_version="1.0",
       started_at=datetime.now(UTC),
       inputs=(provenance.ReferenceLink("source", source_state),),
       outputs=(provenance.ReferenceLink("result", result_state),),
       requested_parameters={"dim": "x"},
   )

   with provenance.ProvenanceCapture() as capture:
       capture.ledger.append(record)

   assert capture.ledger.operation_records == (record,)

The innermost active context is selected when contexts are nested. Leaving a
context, including through an exception, restores the previous context. A
ledger exposes records as immutable snapshots and defines their execution
order by append position; sequence numbers are not part of schema ``0.1``.

Safety and compatibility
------------------------

Parameter values use a conservative JSON-compatible normalizer. It accepts
bounded scalar and collection values, normalizes NumPy scalars, represents
non-finite floats explicitly, sorts mappings and unordered collections
deterministically, and reduces paths to their basename. It never falls back to
an arbitrary object's ``repr``. Explicit record construction validates
strictly by default; best-effort construction can use ``strict=False`` to
store visible ``unsupported`` or ``omitted`` markers and mark capture as
partial.

This API is experimental: its ``0.1`` operation-record schema may evolve
during the 0.x provenance campaign. It currently provides no manifest export,
replay, pipeline generation, persistence integration, figure provenance, or
claim of complete computational reproducibility. Provenance values are passive
data and must not be treated as trusted executable instructions.
