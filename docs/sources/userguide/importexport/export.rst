Export and share data
=====================

Choose a format based on what the recipient needs to recover. Use a native
file to resume work in SpectroChemPy, CSV for numeric columns in a spreadsheet,
JCAMP-DX for infrared exchange, or a report to share a readable summary.

.. _export-native:

Save a dataset for SpectroChemPy
--------------------------------

Save a dataset with its coordinates, units, mask, metadata, and recorded
history using SpectroChemPy's native ``.scp`` format.

.. code-block:: python

   filename = dataset.save_as("spectrum.scp")
   restored = scp.NDDataset.load(filename)

For projects, use the native ``.pscp`` format. Keep the default loader settings
for files from unknown sources. Enable ``allow_unsafe_legacy`` only for a
legacy file you trust that requires pickle-based loading.

See the :doc:`native round trip
</gettingstarted/examples/gallery/auto_examples_core/c_importer/plot_roundtrip_scp>`.

.. _export-csv:

Exchange a spectrum with a spreadsheet
--------------------------------------

Use CSV for one spectrum. The writer exports a signal column and includes a
coordinate column when the dataset has a coordinate. Masked samples are
written as missing values (``NaN``). The header can carry titles and units as
text. Use the native format to preserve the dataset's full metadata.

.. code-block:: python

   filename = spectrum.write_csv("spectrum.csv", delimiter=",")
   restored = scp.read_csv(filename, csv_delimiter=",")

In Excel, import with **Data > From Text/CSV** and select the same delimiter.
If you save an edited copy, keep the coordinate and signal columns in their
original order. Check the units after reading the edited file and restore
them explicitly when needed.

See the :doc:`CSV round trip
</gettingstarted/examples/gallery/auto_examples_core/c_importer/plot_roundtrip_csv>`.

.. _export-jcamp:

Exchange infrared spectra with JCAMP-DX
---------------------------------------

Use JCAMP-DX to exchange supported real infrared spectra. The writer needs a
spectral coordinate and a ``y`` coordinate identifying the spectra, including
for a single spectrum. Use a regularly spaced spectral axis to preserve
the grid on readback. For a linked set, all spectra must share that grid.
The reader reconstructs the axis at uniform spacing from ``FIRSTX``,
``LASTX``, and ``NPOINTS``. For example, an irregular input axis
``[1000, 1001, 1004]`` can be read back as ``[1000, 1002, 1004]``.
Use the native format if you need to keep the irregular grid.

Convert coordinates and signal explicitly to supported units before export.
Even on a regular grid, compare reread values with numerical tolerances
because the text encoding can change the last digits.

.. code-block:: python

   filename = spectra.write_jcamp("spectra.jdx")
   restored = scp.jcamp.read(filename)

See the :doc:`JCAMP-DX tutorial <importJCAMPDX>` for supported units and
readback checks. The :doc:`infrared round trip
</gettingstarted/examples/gallery/auto_examples_core/c_importer/plot_roundtrip_jcamp>`
uses a single absorbance spectrum.

.. _export-markdown-report:

Share a Markdown report
-----------------------

Use ``dataset.to_markdown_report()`` to summarize a dataset and its recorded
history. The method returns Markdown text. Figure references are explicit.
When using relative paths, keep the figures and report together.

.. code-block:: python

   from pathlib import Path

   report = dataset.to_markdown_report(title="Processed IR spectrum")
   Path("spectrum-report.md").write_text(report, encoding="utf-8")

See the :doc:`report example
</gettingstarted/examples/gallery/auto_examples_processing/transformations/plot_markdown_report>`
for saving a plot beside the report.
