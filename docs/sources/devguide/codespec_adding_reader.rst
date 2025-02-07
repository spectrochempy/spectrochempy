.. _contributing.reader:


=============================
Adding a Reader
=============================

Import of spectroscopic data with their meta data from various file formats is a key feature of SpectroChemPy. Data
import is made through the generic ``read()`` function (in ``core.readers.importer`` ) which handles
the determination of filetype, search locations,  etc...).

In the following, we describe the steps to add a specific reader - with the example of Grams/Thermo .spc files.

.. contents:: Table of Contents:
   :local:

1. Add a test and sample files
==============================

Contributors are encouraged to embrace Test-Driven Development (TDD) (see :ref:`contributing.tdd` )
and reader implementation should start by writing an (initially failing) automated test case to read sample file(s) and
then produce the *minimum* amount of code to successfully read this file, i.e., generate a NDDataset with a ``data``
attribute as expected from source file.

The testing of the main readers functionalities has been made in the test_importer.py, and the test of
specific readers such as ``read_spc`` should test only specifics to the reader. Add as many tests and sample files as
the types of file format types you want the reader to handle.

For consistency add your test in ``tests/test_core/test_readers/test_xxx.py`` where ``xxx`` is an alias for the
file format or reader protocol.

The sample file(s) should be grouped in a subdirectory (e.g., ``galacticdata/`` ) of the ``spectrochempy_data/testdata``
folder forked from the `spectrochempy_data repository <https://github.com/spectrochempy/spectrochempy_data/>`_.

A minimum test could be, for instance

.. sourcecode:: python

    def test_read_spc():
        A = scp.read_spc("galacticdata/BENZENE.SPC")
        assert A.shape == (1, 1842)

This will ensure that a dataset with the expected shape has been returned.

For local testing, ensure that the default ``datadir`` correctly points to your local git repo of ``spectrochempy_data`` , e.g.,

.. sourcecode:: python

    prefs.datadir = Path(path/to/testdata)

2. Complete FILETYPES and ALIAS
===============================
They are located at the top of ``core/readers/importer.py`` with the specifics of your reader. e.g.,
for galactic files:

.. sourcecode:: python

    ("galactic", "GRAMS/Thermo Galactic files (*.spc)") #  filetype description to add in FILETYPES
    ("galactic", "spc") #  alias to add in ALIAS

The alias (``spc`` ) will be used to design the specific protocol to read the files.
It must be used to name the public and private reader functions, e.g. ``scp.read_spc()`` and ``_read_spc()`` ).

3. Create the reader_xxx.py file
================================

As illustrated below for the .spc example, the minimal file should contain:

- an example - for simplicity, the same as in the test function
- the public function  ``read_spc``,
- the private function ``_read_spc`` with the appropriate code

.. sourcecode:: python

    __all__ = ["read_spc"]
    __dataset_methods__ = __all__

    # minimum import section
    import io
    from spectrochempy.core.dataset.nddataset import NDDataset
    from spectrochempy.core.readers.importer import _importer_method, Importer

    # ======================================================================================================================
    # Public function
    # ======================================================================================================================
    def read_spc(*paths, **kwargs):
        """
        Open a spc file or a list of files with extension ``.spc`` .

        Parameters
        -----------
        *paths : str, pathlib.Path object, list of str, or list of pathlib.Path objects, optional
            The data source(s) can be specified by the name or a list of name
            for the file(s) to be loaded:
        (....)

        Returns
        --------
        read_xxx
            The dataset or a list of dataset corresponding to a (set of) .xxx
            file(s).

        Example
        ---------
        >>> scp.read_spc('galacticdata/BENZENE.SPC')
        NNDDataset: [float64] unitless (shape: (y:1, x:1842))
        """

        kwargs["filetypes"] = ["GRAMS/Thermo Galactic files (*.spc)"]  #
        kwargs["protocol"] = ["spc"]
        importer = Importer()
        return importer(*paths, **kwargs)


    # ======================================================================================================================
    # Private functions
    # ======================================================================================================================

    @_importer_method
    def _read_spc(*args, **kwargs):
        dataset, filename = args
        content = kwargs.get("content", False)

        if content:
            fid = io.BytesIO(content)
        else:
            fid = open(filename, "rb")
            content = fid.read()

        # Here comes the code to generate the NDDataset from the file
        dataset = NDDataset()
        (....)

    fid.close()
    return dataset

    # ------------------------------------------------------------------
    if __name__ == "__main__":
        pass

Once the minimal code is functional (i.e. returns the dataset with the appropriate ``data`` attribute), the metadata can
be added.

3. General Guidelines for data and metadata format
===================================================

For consistency with existing readers, the following guidelines should be followed as closely as possible:

- The NDDataset should be at least bi-dimensional with a first dimension pertaining to the wavelength/frequency dimension
  and the second dimension ``y`` pertaining to the acquisition time axis, even if the dataset consists of single 1D spectrum.
  For instance

.. sourcecode:: python

    dataset = NDDataset(np.expand_dims(ndarray,  axis=0))    # a 2D dataset from a 1D ndarray

- The acquisition time axis, when relevant, should preferably use a timestamp as coordinate. The labels should at least contain:

    - the acquisition date(s), preferably as a datetime.datetime instances including the timezone (when such data are available in the source file)
    - the initial filename(s) of individual spectra when appropriate

.. sourcecode:: python

    _y = Coord(
        [timestamp],
        title="acquisition timestamp (GMT)",
        units="s",
        labels=([acqdate], [filename]),
    )

- Use whenever possible appropriate units for the data and the coordinates (see userguide/units/units.html).
- The NDDataset ``description`` should at least mention the type of file from which the data have been imported, e.g.:

.. sourcecode:: python

    dataset.description = kwargs.get("description", "Dataset from spc file.\n")

and whenever possible the information related to the instrument, acquisition parameters, etc...

4. Polish your code and make the Pull Requests
==============================================

see: :ref:`contributing_codebase`
