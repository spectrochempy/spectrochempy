.. _contributing.reader:

Adding a Reader
===============

Import of spectroscopic data with their metadata from various file formats is a key feature of SpectroChemPy. Data
import is handled through the generic ``read()`` function (in ``spectrochempy.core.readers.importer``).

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
    __dataset_methods__ = __all__

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
2. Use timestamps for time axes when available
3. Include relevant metadata and units
4. Add proper description

Example of proper axis setup:

.. code-block:: python

    # Set up coordinates
    x_data = get_wavelengths(file)  # Your implementation
    x_coord = scp.Coord(x_data, title="wavelength", units="nm")

    y_data = get_timestamps(file)  # Your implementation
    y_coord = scp.Coord(y_data, title="Time", units="s",
                        labels=acquisition_dates)

    # Create dataset
    data = get_spectra(file)  # Your implementation
    dataset = NDDataset(data,
                       coords=[y_coord, x_coord],
                       title="Absorption",
                       units="absorbance")
    dataset.description = "Dataset from .spc file\n"
    dataset.history.append(f"Imported from {filename}")

Step 5: Documentation
---------------------

1. Add docstrings following NumPy style
2. Include examples in docstrings
3. Add reader to main documentation
4. Update `whatsnew/changelog.rst`

For complete implementation examples, see existing readers in ``spectrochempy/core/readers/``.
