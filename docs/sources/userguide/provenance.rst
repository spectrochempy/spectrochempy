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
operation record each while a capture context is active. P5 adds direct calls
to ``CenterTransformer.fit()`` and ``CenterTransformer.transform()``. P3 adds
four out-of-place binary operations between two ``NDDataset`` operands. P4
adds ordered ``concatenate`` and ``stack`` assembly. P6 adds a direct
``PCA.fit()`` and ``PCA.transform()`` cycle, with bounded evidence about the
fitted estimator. Other scientific operations remain uninstrumented unless
this page states otherwise.

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
in-place operations, concatenation, estimators, readers, and arithmetic beyond
the bounded P3 cases described below are not instrumented. In particular,
in-place slicing
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

Binary arithmetic between datasets
----------------------------------

Out-of-place addition, subtraction, multiplication, and true division between
two ``NDDataset`` operands create one ``combine`` record. The operands keep
their mathematical order through the explicit ``left`` and ``right`` roles,
and the new dataset has the ``result`` role:

.. code-block:: python

   left = scp.NDDataset(np.arange(6.0).reshape(2, 3))
   right = scp.NDDataset(np.ones((2, 3)))

   with scp.provenance.ProvenanceCapture() as capture:
       difference = left - right

   record = capture.ledger.operation_records[0]
   assert [link.role for link in record.inputs] == ["left", "right"]
   assert [link.role for link in record.outputs] == ["result"]

The same boundary applies to direct ``numpy.add``, ``numpy.subtract``,
``numpy.multiply``, and ``numpy.true_divide`` calls with two datasets. Operator
dispatch and direct NumPy-ufunc dispatch each emit exactly one record; delegated
numeric work does not emit a second record. For ``dataset - dataset``, both
roles intentionally point to the same object and state rather than inventing a
second identity.

The operands are represented by input references rather than parameters.
Operator records therefore have an empty ``parameters.requested`` mapping.
Direct ufunc records retain bounded, normalized explicitly supplied ufunc
keywords. ``parameters.resolved`` identifies the public dispatch route and the
result shape and dimensions. This does not change how broadcasting, units,
coordinates, masks, titles, or textual history are calculated.

The existing numeric continuity fingerprint is used independently for both
inputs. It covers numeric data values, shape, and dtype, but not masks,
coordinates, units, or metadata. A successful result whose record cannot be
prepared or appended is invalidated in the transient registry, so a following
captured operation reports partial continuity. Scientific exceptions remain
unchanged; failed records have ordered inputs and no output.

This P3 slice deliberately excludes:

- in-place operations and every ufunc call supplying ``out=``;
- scalar, quantity, coordinate, and external-array operands;
- unary operations, powers, comparisons, reductions, and other ufunc methods;
- persistence, replay, and ``Project`` ownership;
- arithmetic executed inside an uninstrumented ``Pipeline`` step or estimator
  fit/predict invoked by ``cross_validate``.

The operation record stores only opaque references and bounded descriptors. It
does not retain either dataset or a scientific buffer. Fingerprint calculation
still reads both complete numeric input buffers and may make contiguous copies,
so capture cost grows with input size even though retained record size stays
bounded.


Ordered concatenate and stack assembly
--------------------------------------

Direct :func:`~spectrochempy.concatenate` and
:func:`~spectrochempy.stack` calls over supported `NDDataset` inputs create
one ``combine`` record. The same boundary is used when these functions are
called as dataset methods. Every normalized input position is retained in
order through roles ``source[0]``, ``source[1]``, and so on:

.. code-block:: python

   with scp.provenance.ProvenanceCapture() as capture:
       assembled = scp.concatenate([left, right, left], dims="y")

   record = capture.ledger.operation_records[0]
   assert [link.role for link in record.inputs] == [
       "source[0]",
       "source[1]",
       "source[2]",
   ]
   assert record.inputs[0].reference == record.inputs[2].reference

A repeated object is observed and fingerprinted once at that public boundary;
each of its positions reuses the same state reference. The complete ordered
input list remains in the record: reference storage therefore grows linearly
with the number of inputs and is not silently truncated.

For ``concatenate``, ``parameters.requested`` contains only explicitly
supplied ``dims``, ``dim``, and ``axis`` arguments. The resolved
mapping records the effective axis and dimension after the existing alias and
default rules. Normal concatenation reports ``existing_dimension``; the
already-supported 1D ``axis=1`` promotion reports ``new_dimension``
instead. For ``stack``, an omitted ``axis`` is absent from requested
parameters while the resolved axis is ``0``; stacking always reports the new
dimension it creates.

One public call creates one P4 record. In particular, ``stack`` suppresses
the internal ``concatenate`` boundary used by its implementation, and
function/method exposure does not duplicate capture. P4 operations executed
inside the existing ``Pipeline`` and ``cross_validate`` composite boundaries
are suppressed until those composites have their own record contract. Existing
``numpy.concatenate`` and ``numpy.stack`` calls still follow NumPy's
array-conversion path, return ``ndarray``, and create no P4 record; this
slice adds no NumPy dispatch support.

Scientific failures keep their original exception and a failed record has no
output. Capture preparation or record-append failure cannot change a successful
scientific result; that result is invalidated in the transient registry so a
later captured operation reports partial continuity. Values, shapes,
coordinates, units, masks, metadata, titles, and textual histories continue to
come solely from the existing assembly implementation.

P4 does not capture unsupported non-dataset inputs or introduce new iterable,
empty-input, axis, unit, or coordinate behavior. It adds no in-place assembly,
persistence, manifest, replay, or ``Project`` ownership. As in P2 and P3,
the numeric fingerprint covers data, shape, and dtype but not masks,
coordinates, units, or metadata. Its retained digest is bounded, while
fingerprint work scales with the total input data read.


Direct PCA fit and score transform
----------------------------------

Direct :meth:`PCA.fit <spectrochempy.PCA.fit>` and
:meth:`PCA.transform <spectrochempy.PCA.transform>` calls record one estimator
identity across its fitted states and link that estimator to calibration,
source, and score datasets:

.. code-block:: python

   model = scp.PCA(n_components=3, svd_solver="full")

   with scp.provenance.ProvenanceCapture() as capture:
       calibration = dataset[:8]
       model.fit(calibration)
       scores = model.transform(calibration, n_components=2)

   selection_record, fit_record, transform_record = capture.ledger.operation_records
   assert fit_record.inputs[1].reference == selection_record.outputs[0].reference
   assert transform_record.inputs[0].reference == fit_record.outputs[0].reference
   assert transform_record.inputs[1].reference == fit_record.inputs[1].reference

The fit inputs use the roles ``estimator`` and ``calibration``. Its
``fitted_estimator`` output advances the same estimator object to a new state.
The transform inputs use ``estimator`` and ``source`` and its ``result`` output
is a new dataset state. This lets an existing P2, P3, P4, or P5 dataset state
flow into PCA and lets later instrumented operations continue from the score
dataset.

PCA configuration at the call boundary is kept under
``parameters.requested.configuration``. Explicit transform arguments, such as
``n_components``, are kept separately under ``parameters.requested.call``.
``parameters.resolved`` reports the effective solver, fitted component count,
preprocessing flags, input geometry, transform component count, and result
geometry. This separation is significant when ``svd_solver="auto"`` is
resolved by the backend or when transform returns fewer than all fitted
components.
The configuration snapshot covers ``n_components``, ``svd_solver``, ``whiten``,
``tol``, ``iterated_power``, ``n_oversamples``,
``power_iteration_normalizer``, ``random_state``, ``scaled``, and
``standardized``. ``log_level`` is diagnostic rather than scientific. The
current private ``warm_start`` flag does not participate in PCA backend
construction, fit, or direct transform and is not included. A ``RandomState``
instance is described without its internal values in the record, while its
state participates in the transient continuity fingerprint.


The fit summary stores only bounded scalar facts and array descriptors for the
learned PCA state. Component, mean, variance, variance-ratio, and singular-value
values are explicitly omitted; the fitted scikit-learn model is not serialized.
The fit record is therefore ``partial``. Continuity nevertheless fingerprints
that configuration, fitted status, the wrapper's resolved component count,
and backend solver, components, mean, explained variances, variance ratios,
singular values, noise variance, sample count, and feature count. Computing
the fingerprint reads the complete learned arrays and may make contiguous
copies, while the immutable ledger retains neither the arrays nor the model.

An unrecorded change to relevant PCA configuration or learned backend state
breaks verified continuity and makes the next PCA record ``partial``. A failed
initial fit records no output and reports whether the estimator stayed
unchanged. A failed refit preserves PCA's existing behavior: because fitting
invalidates managed fitted state before backend work, the old fit cannot be
silently reused. The failure record reports the observed state effect, and a
subsequent transform still raises the normal not-fitted error. No rollback is
introduced.

Capture preparation or append failure never changes a successful scientific
operation or masks its exception. If a successful fit cannot be recorded, the
estimator identity is invalidated. If a successful transform cannot be
recorded, its result is invalidated so a later captured operation cannot claim
complete continuity through the missing boundary.

P6 is deliberately limited to exact ``PCA`` instances and direct
``NDDataset`` arguments. It does not capture:

- ``fit_transform()``, ``inverse_transform()``, or scores/loadings accessors;
- subclasses of ``PCA`` or other estimators;
- PCA steps invoked internally by ``Pipeline`` or other composites;
- persistence, manifest, replay, or ``Project`` ownership.

The exclusion of subclasses prevents a derived estimator with additional
configuration or learned state from being attributed to the base PCA contract.
Composite suppression is local to the PCA operation identifiers and does not
suppress existing dataset records outside the composite boundary.

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
