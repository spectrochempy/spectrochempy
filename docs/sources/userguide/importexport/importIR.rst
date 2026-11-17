Import IR Data
==============

This tutorial demonstrates how to import infrared spectroscopy data in
SpectroChemPy.

.. code:: ipython3

    import spectrochempy as scp


Supported IR File Formats
-------------------------

The core package provides these IR readers:

======== ================ ==============================
Format   Function         Description
======== ================ ==============================
OMNIC    ``read_omnic()`` Thermo Scientific (.spa, .spg)
OPUS     ``read_opus()``  Bruker OPUS (.0, …)
JCAMP-DX ``read_jcamp()`` Standard IR exchange format
======== ================ ==============================


Two optional plugins cover PerkinElmer spectra and Carroucell experiment
directories. Use the reader for your instrument when the same extension
could refer to files from different vendors.

.. _import-perkinelmer-ir:

PerkinElmer infrared spectra
----------------------------

Install the :doc:`PerkinElmer plugin </userguide/plugins/perkinelmer>` and use
its reader for single-spectrum PerkinElmer binary ``.sp`` files.

.. code-block:: bash

   python -m pip install spectrochempy-perkinelmer

.. code-block:: python

   dataset = scp.perkinelmer.read("irdata/perkinelmer/spectra.sp")
   _ = dataset.plot()

The current reader exposes wavelength in nanometres and signal in
absorbance. Check ``dataset.x.units`` and ``dataset.units`` before comparing
it with a spectrum recorded on another axis. The reader does not support
``.prf`` files or ``.sp`` files containing several spectra.

Instrument, detector, source, analyst, acquisition date, and accumulation
count are available when the relevant blocks are present in the file.
Use ``meta.get()`` to handle absent fields when reporting or checking
acquisition conditions.

.. code-block:: python

   instrument = dataset.meta.get("instrument_model", "unknown")
   detector = dataset.meta.get("detector", "unknown")
   accumulations = dataset.meta.get("accumulations")

See the :doc:`short PerkinElmer example
</gettingstarted/examples/gallery/auto_examples_core/c_importer/plot_read_perkinelmer>`
for reading and plotting the sample spectrum.

.. _import-carroucell-ir:

Carroucell experiment directories
---------------------------------

Install the :doc:`Carroucell plugin </userguide/plugins/carroucell>` when your
experiment produced a directory of OMNIC ``.spa`` spectra from several
sample holders.

.. code-block:: bash

   python -m pip install spectrochempy-carroucell

The reader groups filenames by the part before the first underscore. For
example, ``3-SILICA_1.SPA`` and ``3-SILICA_2.SPA`` belong to the same series.
Keep the numbered spectra together in the directory. The spectra within a
series are ordered by acquisition time. The ``spectra=(1, 3)`` argument
selects file numbers 1 through 3 inclusive. By default, files with
``BCKG`` in the name are excluded. Pass ``discardbg=False`` to include them
as readings.

With ``spectra``, the reader requires an integer in the segment immediately
after the first underscore in every ``.spa`` filename. It checks background
names before applying ``discardbg``. A name such as ``BG1-BCKG_1.SPA`` works
with the filter. If the directory contains names such as
``9_BCKG_001BG.SPA``, omit ``spectra`` to read the sample series while
excluding the background files.

For the example directory below, the result is a collection of datasets.
The reader can also return a single ``NDDataset``, for example when the
directory contains a single sample series. When processing multiple
directories, check the return type before iterating over sample series.

.. code-block:: python

   series = scp.carroucell.read("irdata/carroucell_samp", spectra=(1, 3))
   for sample in series:
       print(sample.name, sample.shape)

The reader can attach temperatures from a compatible ``.xls`` log placed
beside the spectra. Provide exactly one log. The reader uses its first
sheet, starting at row 10, with timestamps in column A and temperatures in
column E. Store timestamps as text in ``dd/mm/yy HH:MM:SS`` format (for
example, ``16/11/26 14:05:00``). The reader skips Excel date cells and
timestamps it cannot parse.

``delta_clocks`` is the thermocouple time minus the spectrometer time, in
seconds. The reader adds it to the spectrum timestamps, then keeps log
readings within the corrected acquisition interval. Provide at least two
valid readings with distinct timestamps in chronological order within
that interval. The reader interpolates between those readings and
extrapolates at the ends if needed. The current temperature-label code
expects 50 spectra per sample series. Read the complete series when
importing a log, then verify the offset and the resulting temperature labels
before using them quantitatively.

See the :doc:`short Carroucell example
</gettingstarted/examples/gallery/auto_examples_core/c_importer/plot_read_carroucell>`
to read the example directory and plot one sample series.

Detailed Tutorials
------------------

.. toctree::
   :maxdepth: 3

   importOMNIC
   importOPUS

See :doc:`importJCAMPDX` for infrared JCAMP-DX and other supported techniques.
