# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Extend NDDataset with the import method for Thermo galactic (spc) data files."""

__all__ = ["read_soc", "read_ddr", "read_sdr", "read_hdr"]
__dataset_methods__ = __all__


from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.readers.read_omnic import _read_spa
from spectrochempy.utils.docreps import _docstring

# ======================================================================================
# Public functions
# ======================================================================================
_docstring.delete_params("Importer.see_also", "read_soc")


@_docstring.dedent
def read_soc(*paths, **kwargs):
    r"""
    Read a Surface Optics Corps. file or a list of files with extension :file:`.ddr`, :file:`.hdr` or :file:`.sdr`.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the backgroung interferogram of the spa file if present or None if absent.
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also.no_read_soc)s

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.ddr *.hdr *.sdr)"]
    kwargs["protocol"] = ["soc", "ddr", "hdr", "sdr"]
    importer = Importer()
    return importer(*paths, **kwargs)


@_docstring.dedent
def read_ddr(*paths, **kwargs):
    r"""
    Open a Surface Optics Corps. file or a list of files with extension :file:`.ddr`.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the backgroung interferogram of the spa file if present or None if absent.
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also)s

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.ddr)"]
    kwargs["protocol"] = ["ddr"]
    importer = Importer()
    return importer(*paths, **kwargs)


@_docstring.dedent
def read_hdr(*paths, **kwargs):
    r"""
    Open a Surface Optics Corps. file or a list of files with extension :file:`.hdr`.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the backgroung interferogram of the spa file if present or None if absent.
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also)s

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.hdr)"]
    kwargs["protocol"] = ["hdr"]
    importer = Importer()
    return importer(*paths, **kwargs)


@_docstring.dedent
def read_sdr(*paths, **kwargs):
    r"""
    Open a Surface Optics Corps. file or a list of files with extension :file:`.sdr`.

    Parameters
    ----------
    %(Importer.parameters)s

    Returns
    -------
    %(Importer.returns)s

    Other Parameters
    ----------------
    return_ifg : str or None, optional
        Default value is None. When set to 'sample' returns the sample interferogram
        of the spa file if present or None if absent. When set to 'background' returns
        the backgroung interferogram of the spa file if present or None if absent.
    %(Importer.other_parameters)s

    See Also
    --------
    %(Importer.see_also)s

    """
    kwargs["filetypes"] = ["Surface Optics Corp. (*.sdr)"]
    kwargs["protocol"] = ["sdr"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private functions
# ======================================================================================
@_importer_method
def _read_ddr(*args, **kwargs):
    ds = _read_spa(*args, **kwargs)
    ds.history[-1] = "Imported from ddr file(s)"
    return ds


@_importer_method
def _read_hdr(*args, **kwargs):
    ds = _read_spa(*args, **kwargs)
    ds.history[-1] = "Imported from hdr file(s)"
    return ds


@_importer_method
def _read_sdr(*args, **kwargs):
    ds = _read_spa(*args, **kwargs)
    ds.history[-1] = "Imported from sdr file(s)"
    return ds
