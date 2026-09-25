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
operation record each while a capture context is active. The next bounded slice
adds direct calls to ``CenterTransformer.fit()`` and
``CenterTransformer.transform()``. Other scientific operations remain
uninstrumented unless this page states otherwise.

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

Each object state carries a numeric fingerprint when it is used again as an
input. The fingerprint covers the data **values** (a digest over the numeric
buffer), the array **shape**, and the **dtype** only: it does not cover the
mask, the coordinates, the units, or the metadata. Consequently, an in-place
mutation of ``data`` between two recorded operations is detected (the next
record references the object's current observed state and is captured as
``partial`` with an explicit ``unrecorded_state_change`` omission), while a
change confined to the mask, coordinates, units, or metadata is not detected.
Verified chains keep a ``complete`` capture and their input-to-output reference
links. Only the stored digest is bounded: computing it reads the whole data
buffer and may create a contiguous copy of it.

Parameter descriptions are prepared inside the capture protection, so a
failure in capture never raises into scientific code: the operation still
returns normally and the unavailable parameter is recorded as an omission with
a ``partial`` capture. Failed operations record the bounded parameters that
were requested under ``parameters.requested`` together with the structured
failure.

A :class:`~spectrochempy.provenance.ProvenanceCapture` instance is single-use.
Once its context closes — normally or through an exception — it cannot be
reopened or resumed, and it is no longer active in tasks that inherit the
context. Create a new instance for each capture session.

The selection is captured as a JSON-safe structural description (``slice``,
``ellipsis``, and index values), and transpose records the requested dimension
order. Slicing and transpose never mutate the source, and the textual
:attr:`~spectrochempy.NDDataset.history` produced by the operation is
unchanged. This first slice covers only single-source out-of-place operations;
in-place operations, arithmetic, concatenation, estimators, and readers are not
instrumented. In particular, in-place slicing
(``dataset[:, ..., INPLACE]``) and in-place transpose are outside the capture
scope and create no record.

Direct CenterTransformer fit and transform
--------------------------------------------

Direct calls to :meth:`CenterTransformer.fit
<spectrochempy.CenterTransformer.fit>` and
:meth:`CenterTransformer.transform <spectrochempy.CenterTransformer.transform>`
record the lifecycle of one transformer identity:

.. code-block:: python

   transformer = scp.CenterTransformer(dim="y")

   with scp.provenance.ProvenanceCapture() as capture:
       calibration = dataset[:8]
       transformer.fit(calibration)
       centered = transformer.transform(calibration)

   selection_record, fit_record, transform_record = capture.ledger.operation_records
   assert fit_record.inputs[1].reference == selection_record.outputs[0].reference
   assert transform_record.inputs[0].reference == fit_record.outputs[0].reference

The fit record references the transformer's prior state and the calibration
dataset, then advances the same transformer object to a fitted state. The
transform record references that fitted state and its source dataset, and
creates a new dataset object/state. ``fit()`` and ``transform()`` accept no
configuration arguments at their call boundaries, so their
``parameters.requested`` mappings are empty. ``parameters.resolved`` records
the current ``dim`` configuration together with the resolved axis and dimension
name.

The fitted-state summary reports only bounded facts about ``mean_``: dtype,
shape, size, and masked count. Mean values are explicitly marked ``omitted``
and are not stored in the ledger, so a successful fit record is ``partial``.
This is passive trace evidence, not persistence of a fitted transformer and not
replay.

Within an active capture, continuity checks digest the transformer's current
configuration, fitted flag, compatibility signature, and learned mean,
including its mask. The stored digest is bounded, but calculating it reads the
learned arrays and compatibility-coordinate arrays in full and may create
contiguous copies. Direct configuration or learned-state changes therefore
start a new observed state and make the next record ``partial`` with an
``unrecorded_state_change`` omission. Unsupported runtime state is treated as
unverifiable rather than complete. These checks do not extend the P2 dataset
fingerprint, whose mask, coordinate, unit, and metadata limitations remain
unchanged.

A failed refit records whether the transformer's state is known to be changed,
unchanged, or unknown. In particular, the existing scientific behavior is
preserved: a failed refit may leave a previously fitted transformer unfitted.
No rollback is added for provenance. Capture failure never masks the scientific
result or exception; if a successful fit cannot be recorded, later operations
do not claim verified continuity with that missing fit.

Only direct ``fit()`` and ``transform()`` calls are in scope. The following
paths create no new Center records:

- ``fit_transform()`` and ``inverse_transform()``;
- the procedural ``center()`` adapter;
- subclasses of ``CenterTransformer``, whose additional learned state requires
  its own explicit capture contract;
- Center steps invoked internally by ``Pipeline`` or ``cross_validate``;
- all other preprocessors and estimators.

Suppression is local to the current execution context and to the two Center
operation identifiers. It composes with nested capture contexts and does not
suppress the existing P2 slice or transpose records.

Demonstrator: interactive selection followed by transpose
---------------------------------------------------------

A common interactive workflow selects a region in the assistant and then
transposes it. The region bounds are chosen at runtime and are not present in
the originating script; with capture active they are still retained in the
ledger, together with the link to the transposed result:

.. code-block:: python

   import numpy as np

   import spectrochempy as scp

   dataset = scp.NDDataset(np.arange(24.0).reshape(4, 6))


   def apply_viewer_region_then_transpose(data, x_bounds, y_bounds):
       selection = data[slice(*x_bounds), slice(*y_bounds)]
       return selection.transpose()


   with scp.provenance.ProvenanceCapture() as capture:
       result = apply_viewer_region_then_transpose(dataset, (1, 3), (0, 2))

   slice_record, transpose_record = capture.ledger.operation_records
   assert slice_record.to_dict()["parameters"]["requested"]["values"]["selection"] == [
       {"type": "slice", "start": 1, "stop": 3, "step": None},
       {"type": "slice", "start": 0, "stop": 2, "step": None},
   ]
   assert transpose_record.inputs[0].reference.id == slice_record.outputs[0].reference.id

The applied selection is recorded even though
``apply_viewer_region_then_transpose`` contains no region bounds: only the
runtime arguments carried the interactive decision. The second record
references the first record's output, so the trace retains both the selection
that was actually applied and its link to the transposed result. This
demonstrates the passive trace value of P2; it does not provide replay, and the
records are data, not executable instructions.

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
