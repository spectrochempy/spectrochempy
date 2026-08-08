.. _contributing.reader:

Adding a Reader
===============

Import of spectroscopic data with their metadata from various file formats is a key feature of SpectroChemPy. Data
import is handled through the generic ``read()`` function (in ``spectrochempy.core.readers.importer``).

Reader functions are package-level APIs because they create datasets from
external data. New readers should therefore be exposed as
``scp.read_xxx(...)`` or, for plugin readers, ``scp.<plugin>.read_xxx(...)``.
Do not add reader methods to ``NDDataset`` or dataset accessors such as
``dataset.read_xxx(...)`` or ``dataset.<plugin>.read_xxx(...)``.

This guide describes the steps to add a specific reader, using the example of reading Grams/Thermo .spc files.

.. contents:: Contents
   :local:
   :depth: 2

Step 1: Add Tests and Sample Files
----------------------------------

Following :ref:`Test-Driven Development <contributing.tdd>`, start by writing tests and providing sample files:

1. Create test file in ``tests/test_core/test_readers/test_xxx.py``
2. Add sample files in ``spectrochempy_data/testdata/xxx_data/``
3. Write basic test case:

.. code-block:: python

    def test_read_spc():
        path = "spc_data/BENZENE.SPC"
        dataset = scp.read_spc(path)
        assert dataset.shape == (1, 1842)
        assert isinstance(dataset, scp.NDDataset)

For local testing, configure the data directory:

.. code-block:: python

    scp.preferences.datadir = Path("path/to/testdata")

Step 2: Register the File Format
--------------------------------

Add format details in ``spectrochempy/core/readers/importer.py``:

.. code-block:: python

    FILETYPES = [
        // ...existing code...
        ("galactic", "GRAMS/Thermo Galactic files (*.spc)"),
    ]

    ALIAS = [
        // ...existing code...
        ("galactic", "spc"),
    ]

Step 3: Create the Reader Module
--------------------------------

Create ``spectrochempy/core/readers/reader_xxx.py``:

.. code-block:: python

    # Basic structure for reader_spc.py
    from spectrochempy.core.dataset.nddataset import NDDataset
    from spectrochempy.core.readers.importer import _importer_method, Importer

    __all__ = ["read_spc"]
    def read_spc(*paths, **kwargs):
        """Read Thermo Galactic .spc file(s).

        Parameters
        ----------
        *paths : str or Path
            Path(s) to .spc file(s)
        **kwargs
            Additional import options

        Returns
        -------
        NDDataset or list of NDDataset
            Loaded spectral data
        """
        kwargs["filetypes"] = ["GRAMS/Thermo Galactic files (*.spc)"]
        kwargs["protocol"] = ["spc"]
        importer = Importer()
        return importer(*paths, **kwargs)

    @_importer_method
    def _read_spc(*args, **kwargs):
        """Internal reader implementation."""
        dataset, filename = args
        // ...implementation details...
        return dataset

Step 4: Data Format Guidelines
------------------------------

When implementing the reader:

1. Always return 2D datasets, even for 1D spectra
2. Set coordinates, metadata, and provenance on the semantic destination
3. Include relevant units and descriptions
4. Keep parser-only temporary state out of the returned dataset

Reader semantic normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the existing dataset fields for shared meanings instead of inventing
reader-specific conventions:

* signal identity:
  ``dataset.name``, ``dataset.title``, ``dataset.units``,
  ``dataset.description``
* source provenance:
  ``dataset.filename``, ``dataset.origin``, ``dataset.author``
* acquisition/session time:
  ``dataset.acquisition_date``
* pointwise time or support geometry:
  ``Coord`` values
* pointwise identifiers or categories:
  ``Coord.labels``
* import and processing events:
  ``dataset.history``
* vendor-specific technical payloads:
  ``dataset.meta``
* parser-only temporary state:
  do not persist it on the returned dataset

Dual-time rule:

* ``acquisition_date`` = dataset or session provenance
* time coordinates = observation geometry
* both may coexist in the same imported dataset

Examples:

* a single acquisition start time belongs in ``dataset.acquisition_date``
* a timestamp or elapsed-time axis for each spectrum belongs in a ``Coord``
* sample IDs, acquisition names, or categorical row identifiers belong in
  ``Coord.labels``
* vendor parameter blocks belong in ``dataset.meta``

Example of proper axis setup:

.. code-block:: python

    from spectrochempy.core.dataset.coord import Coord
    from spectrochempy.core.dataset.nddataset import NDDataset

    x_coord = Coord(wavenumbers, title="wavenumber", units="cm^-1")
    y_coord = Coord(
        elapsed_seconds,
        title="elapsed time",
        units="s",
        labels=sample_ids,
    )

    dataset = NDDataset(data)
    dataset.set_coordset(y=y_coord, x=x_coord)
    dataset.name = filename.stem
    dataset.title = "absorbance"
    dataset.units = "absorbance"
    dataset.description = "Dataset imported from vendor_x"

    dataset.filename = filename
    dataset.origin = "vendor_x"
    dataset.acquisition_date = acquisition_start
    dataset.history = f"Imported from vendor_x file {filename}"

    dataset.meta.instrument_model = instrument_model
    dataset.meta.processing_mode = processing_mode

In this example:

* ``elapsed_seconds`` stays on the ``y`` coordinate because it locates each
  observation
* ``sample_ids`` stays in ``Coord.labels`` because it identifies points along
  the imported axis
* ``acquisition_start`` becomes ``dataset.acquisition_date`` because it is
  session provenance
* vendor parameters remain in ``dataset.meta`` instead of being promoted to new
  typed dataset fields

Step 5: Documentation
---------------------

1. Add docstrings following NumPy style
2. Include examples in docstrings
3. Add reader to main documentation
4. Update `whatsnew/changelog.rst`

For complete implementation examples, see existing readers in ``spectrochempy/core/readers/``.

Step 6: Semantic Reader Tests
-----------------------------

In addition to basic shape or import tests, add focused semantic tests for the
reader you introduce.

Recommended coverage:

* identity:
  ``name``, ``title``, ``units``, ``description``
* provenance:
  ``filename``, ``origin``, ``author``, ``acquisition_date`` when available
* coordinate semantics:
  coordinate values, titles, units, and time-axis meaning
* labels:
  which axis carries labels and what they identify
* retained ``Meta``:
  important vendor-specific technical payloads
* history:
  import events and vendor processing history when preserved

These tests should check semantic placement, not only raw values. A good reader
test should make it obvious why a field lives on the dataset, on a coordinate,
in labels, or in ``Meta``.
