# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Bruker file (single dimension FID or two-dimensional SER) importers.

This module provides functionality to read Bruker Topspin NMR data files.

Functions
---------
- read_topspin : Main entry point for reading Bruker files

Notes
-----
Supports FID and processed data for 1D and 2D experiments only.

"""

__all__ = ["read_topspin"]

import contextlib
import re
from datetime import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.meta import Meta
from spectrochempy_nmr.extern.nmrglue import read_fid
from spectrochempy_nmr.extern.nmrglue import read_pdata

# ======================================================================================
# Constants
# ======================================================================================
FnMODE = ["undefined", "QF", "QSEQ", "TPPI", "STATES", "STATES-TPPI", "ECHO-ANTIECHO"]
AQ_mod = ["QF", "QSIM", "QSEQ", "DQD"]

nmr_valid_meta = [
    # ACQU
    # ('amp', ''),
    ("aq_mod", ""),
    ("aqseq", ""),  # ('aunm', ''),
    # ('autopos', ''),
    ("bf1", "MHz"),
    ("bf2", "MHz"),
    ("bf3", "MHz"),
    ("bf4", "MHz"),  # ('bf5', 'MHz'),
    # ('bf6', 'MHz'),
    # ('bf7', 'MHz'),
    # ('bf8', 'MHz'),
    # ('bytorda', ''),
    # ('cfdgtyp', ''),
    # ('cfrgtyp', ''),
    # ('chemstr', ''),
    ("cnst", ""),  # ('cpdprg', ''),
    # ('cpdprg1', ''),
    # ('cpdprg2', ''),
    # ('cpdprg3', ''),
    # ('cpdprg4', ''),
    # ('cpdprg5', ''),
    # ('cpdprg6', ''),
    # ('cpdprg7', ''),
    # ('cpdprg8', ''),
    # ('cpdprgb', ''),
    # ('cpdprgt', ''),
    ("d", "s"),
    ("date", ""),
    # ('dbl', ''),
    # ('dbp', ''),
    # ('dbp07', ''),
    # ('dbpnam0', ''),
    # ('dbpnam1', ''),
    # ('dbpnam2', ''),
    # ('dbpnam3', ''),
    # ('dbpnam4', ''),
    # ('dbpnam5', ''),
    # ('dbpnam6', ''),
    # ('dbpnam7', ''),
    # ('dbpoal', ''),
    # ('dbpoffs', ''),
    ("de", "us"),  # ('decbnuc', ''),
    # ('decim', ''),
    # ('decnuc', ''),
    # ('decstat', ''),
    ("digmod", ""),  # ('digtyp', ''),
    # ('dl', ''),
    # ('dp', ''),
    # ('dp07', ''),
    # ('dpname0', ''),
    # ('dpname1', ''),
    # ('dpname2', ''),
    # ('dpname3', ''),
    # ('dpname4', ''),
    # ('dpname5', ''),
    # ('dpname6', ''),
    # ('dpname7', ''),
    # ('dpoal', ''),
    # ('dpoffs', ''),
    # ('dqdmode', ''),
    # ('dr', ''),
    # ('ds', ''),
    # ('dslist', ''),
    # ('dspfirm', ''),
    # ('dspfvs', ''),
    # ('dtypa', ''),
    # ('exp', ''),
    # ('f1list', ''),
    # ('f2list', ''),
    # ('f3list', ''),
    # ('fcuchan', ''),
    # ('file_size', ''),
    # ('fl1', ''),
    # ('fl2', ''),
    # ('fl3', ''),
    # ('fl4', ''),
    ("fnmode", ""),  # ('fov', ''),
    # ('fq1list', ''),
    # ('fq2list', ''),
    # ('fq3list', ''),
    # ('fq4list', ''),
    # ('fq5list', ''),
    # ('fq6list', ''),
    # ('fq7list', ''),
    # ('fq8list', ''),
    # ('fs', ''),
    # ('ftlpgn', ''),
    # ('fw', ''),
    # ('gp031', ''),
    # ('gpnam0', ''),
    # ('gpnam1', ''),
    # ('gpnam10', ''),
    # ('gpnam11', ''),
    # ('gpnam12', ''),
    # ('gpnam13', ''),
    # ('gpnam14', ''),
    # ('gpnam15', ''),
    # ('gpnam16', ''),
    # ('gpnam17', ''),
    # ('gpnam18', ''),
    # ('gpnam19', ''),
    # ('gpnam2', ''),
    # ('gpnam20', ''),
    # ('gpnam21', ''),
    # ('gpnam22', ''),
    # ('gpnam23', ''),
    # ('gpnam24', ''),
    # ('gpnam25', ''),
    # ('gpnam26', ''),
    # ('gpnam27', ''),
    # ('gpnam28', ''),
    # ('gpnam29', ''),
    # ('gpnam3', ''),
    # ('gpnam30', ''),
    # ('gpnam31', ''),
    # ('gpnam4', ''),
    # ('gpnam5', ''),
    # ('gpnam6', ''),
    # ('gpnam7', ''),
    # ('gpnam8', ''),
    # ('gpnam9', ''),
    # ('gpx', ''),
    # ('gpy', ''),
    # ('gpz', ''),
    # ('grdprog', ''),
    # ('hdduty', ''),
    # ('hdrate', ''),
    # ('hgain', ''),
    # ('hl1', ''),
    # ('hl2', ''),
    # ('hl3', ''),
    # ('hl4', ''),
    # ('holder', ''),
    # ('hpmod', ''),
    # ('hpprgn', ''),
    # ('in', 's'),
    # ('inp', 's'),
    # ('instrum', ''),
    # ('l', ''),
    # ('lfilter', ''),
    # ('lgain', ''),
    # ('locked', ''),
    # ('lockfld', ''),
    # ('lockgn', ''),
    # ('lockpow', ''),
    # ('lockppm', ''),
    # ('locnuc', ''),
    # ('locphas', ''),
    # ('locshft', ''),
    # ('ltime', ''),
    ("masr", "Hz"),  # ('masrlst', ''),
    # ('nbl', ''),
    ("nc", ""),
    ("ns", ""),
    ("nuc1", ""),
    ("nuc2", ""),
    ("nuc3", ""),
    ("nuc4", ""),  # ('nuc5', ''),
    # ('nuc6', ''),
    # ('nuc7', ''),
    # ('nuc8', ''),
    ("nuclei", ""),
    ("nucleus", ""),
    ("o1", "Hz"),
    ("o2", "Hz"),
    ("o3", "Hz"),
    ("o4", "Hz"),  # ('o5', 'Hz'),
    # ('o6', 'Hz'),
    # ('o7', 'Hz'),
    # ('o8', 'Hz'),
    # ('obschan', ''),
    # ('overflw', ''),
    ("p", "us"),  # ('paps', ''),
    ("parmode", ""),  # ('pcpd', ''),
    ("ph_ref", ""),
    ("phcor", ""),  # ('php', ''),
    ("pl", ""),  # ('powmod', ''),
    # ('pr', ''),
    # ('prechan', ''),
    # ('prgain', ''),
    # ('probhd', ''),
    # ('prosol', ''),
    ("pulprog", ""),
    ("pw", "W"),  # ('qnp', ''),
    # ('qs', ''),
    # ('qsb', ''),
    # ('rd', ''),
    # ('recchan', ''),
    # ('recph', ''),
    ("rg", ""),  # ('ro', ''),
    # ('routwd1', ''),
    # ('routwd2', ''),
    # ('rpuused', ''),
    # ('rsel', ''),
    # ('s', ''),
    # ('seout', ''),
    ("sfo1", "MHz"),
    ("sfo2", "MHz"),
    ("sfo3", "MHz"),
    ("sfo4", "MHz"),  # ('sfo5', 'MHz'),
    # ('sfo6', 'MHz'),
    # ('sfo7', 'MHz'),
    # ('sfo8', 'MHz'),
    # ('solvent', ''),
    # ('sp', ''),
    # ('sp07', ''),
    # ('spectr', ''),
    # ('spnam0', ''),
    # ('spnam1', ''),
    # ('spnam10', ''),
    # ('spnam11', ''),
    # ('spnam12', ''),
    # ('spnam13', ''),
    # ('spnam14', ''),
    # ('spnam15', ''),
    # ('spnam16', ''),
    # ('spnam17', ''),
    # ('spnam18', ''),
    # ('spnam19', ''),
    # ('spnam2', ''),
    # ('spnam20', ''),
    # ('spnam21', ''),
    # ('spnam22', ''),
    # ('spnam23', ''),
    # ('spnam24', ''),
    # ('spnam25', ''),
    # ('spnam26', ''),
    # ('spnam27', ''),
    # ('spnam28', ''),
    # ('spnam29', ''),
    # ('spnam3', ''),
    # ('spnam30', ''),
    # ('spnam31', ''),
    # ('spnam4', ''),
    # ('spnam5', ''),
    # ('spnam6', ''),
    # ('spnam7', ''),
    # ('spnam8', ''),
    # ('spnam9', ''),
    # ('spoal', ''),
    # ('spoffs', ''),
    # ('subnam0', ''),
    # ('subnam1', ''),
    # ('subnam2', ''),
    # ('subnam3', ''),
    # ('subnam4', ''),
    # ('subnam5', ''),
    # ('subnam6', ''),
    # ('subnam7', ''),
    # ('subnam8', ''),
    # ('subnam9', ''),
    ("sw", "ppm"),  # ('sw_h', 'Hz'),
    # ('swibox', ''),
    ("td", ""),  # ('td0', ''),
    ("te", "K"),  # ('te2', ''),
    # ('te3', ''),
    # ('teg', ''),
    # ('tl', ''),
    # ('tp', ''),
    # ('tp07', ''),
    # ('tpname0', ''),
    # ('tpname1', ''),
    # ('tpname2', ''),
    # ('tpname3', ''),
    # ('tpname4', ''),
    # ('tpname5', ''),
    # ('tpname6', ''),
    # ('tpname7', ''),
    # ('tpoal', ''),
    # ('tpoffs', ''),
    # ('tunhin', ''),
    # ('tunhout', ''),
    # ('tunxout', ''),
    # ('usera1', ''),
    # ('usera2', ''),
    # ('usera3', ''),
    # ('usera4', ''),
    # ('usera5', ''),
    # ('v9', ''),
    ("valist", ""),
    ("vclist", ""),
    ("vd", ""),
    ("vdlist", ""),
    ("vplist", ""),
    ("vtlist", ""),  # ('wbst', ''),
    # ('wbsw', ''),
    # ('ws', ''),
    # ('xgain', ''),
    # ('xl', ''),
    # ('yl', ''),
    # ('ymax_a', ''),
    # ('ymin_a', ''),
    # ('zgoptns', ''),
    # ('zl1', ''),
    # ('zl2', ''),
    # ('zl3', ''),
    # ('zl4', ''),
    # PROCS
    # ('absf1', ''),
    # ('absf2', ''),
    # ('absg', ''),
    # ('absl', ''),
    # ('acqt0', ''),
    # ('alpha', ''),
    # ('ampcoil', ''),
    # ('anavpt', ''),
    ("aqorder", ""),  # ('assfac', ''),
    # ('assfaci', ''),
    # ('assfacx', ''),
    # ('asswid', ''),
    # ('aunmp', ''),
    # ('axleft', ''),
    # ('axname', ''),
    # ('axnuc', ''),
    # ('axright', ''),
    # ('axtype', ''),
    # ('axunit', ''),
    # ('azfe', ''),
    # ('azfw', ''),
    # ('bc_mod', ''),
    # ('bcfw', ''),
    # ('bytordp', ''),
    # ('cagpars', ''),
    # ('coroffs', ''),
    # ('cy', ''),
    # ('datmod', ''),
    # ('dc', ''),
    # ('dfilt', ''),
    # ('dtypp', ''),
    # ('eretic', ''),
    # ('f1p', ''),
    # ('f2p', ''),
    # ('fcor', ''),
    # ('fntype', ''),
    # ('frqlo3', ''),
    # ('frqlo3n', ''),
    # ('ft_mod', ''),
    # ('ftsize', ''),
    # ('gamma', ''),
    # ('gb', 'Hz' ),
    # ('gpnam', ''),
    # ('grpdly', ''),
    ("inf", "us"),  # ('intbc', ''),
    # ('intscl', ''),
    # ('isen', ''),
    # ('lb', 'Hz' ),
    # ('lev0', ''),
    # ('linpstp', ''),
    # ('locsw', ''),
    # ('lpbin', ''),
    # ('maxi', ''),
    ("mc2", ""),  # ('mdd_csalg', ''),
    # ('mdd_cslambda', ''),
    # ('mdd_csniter', ''),
    # ('mdd_csnorm', ''),
    # ('mdd_cszf', ''),
    # ('mdd_mod', ''),
    # ('mddcexp', ''),
    # ('mddct_sp', ''),
    # ('mddf180', ''),
    # ('mddlambda', ''),
    # ('mddmemory', ''),
    # ('mddmerge', ''),
    # ('mddncomp', ''),
    # ('mddniter', ''),
    # ('mddnoise', ''),
    # ('mddphase', ''),
    # ('mddseed', ''),
    # ('mddsrsize', ''),
    # ('me_mod', ''),
    # ('mean', ''),
    # ('mi', ''),
    # ('mulexpno', ''),
    # ('nc_proc', ''),
    # ('ncoef', ''),
    # ('nlev', ''),
    # ('nlogch', ''),
    # ('noisf1', ''),
    # ('noisf2', ''),
    # ('novflw', ''),
    # ('nsp', ''),
    # ('nth_pi', ''),
    # ('nusamount', ''),
    # ('nusfpnz', ''),
    # ('nusjsp', ''),
    # ('nuslist', ''),
    # ('nusseed', ''),
    # ('nust2', ''),
    # ('nustd', ''),
    # ('nzp', ''),
    # ('offset', ''),
    # ('pacoil', ''),
    # ('pc', ''),
    # ('pexsel', ''),
    # ('ph_mod', ''),
    ("phc0", "deg"),
    ("phc1", "deg"),  # ('phlist', ''),
    # ('pknl', ''),
    # ('plstep', ''),
    # ('plstrt', ''),
    # ('plw', ''),
    # ('plwmax', ''),
    # ('pparmod', ''),
    # ('ppdiag', ''),
    # ('ppiptyp', ''),
    # ('ppmpnum', ''),
    # ('ppresol', ''),
    # ('pqphase', ''),
    # ('pqscale', ''),
    # ('pscal', ''),
    # ('psign', ''),
    # ('pynm', ''),
    # ('pynmp', ''),
    # ('recpre', ''),
    # ('recprfx', ''),
    # ('recsel', ''),
    ("reverse", ""),  # ('s_dev', ''),
    # ('selrec', ''),
    ("sf", "MHz"),  # ('si', ''),
    # ('sigf1', ''),
    # ('sigf2', ''),
    # ('sino', ''),
    # ('siold', ''),
    # ('solvold', ''),
    # ('spectyp', ''),
    # ('spincnt', ''),
    # ('spnam', ''),
    # ('sppex', ''),
    # ('spw', ''),
    # ('sreglst', ''),
    # ('ssb', ''),
    # ('stsi', ''),
    # ('stsr', ''),
    # ('subnam', ''),
    ("sw_p", ""),  # ('swfinal', ''),
    # ('symm', ''),
    # ('tdeff', ''),
    # ('tdoff', ''),
    # ('te1', ''),
    # ('te4', ''),
    # ('te_pidx', ''),
    # ('te_stab', ''),
    # ('ti', ''),
    # ('tilt', ''),
    # ('tm1', ''),
    # ('tm2', ''),
    # ('toplev', ''),
    # ('userp1', ''),
    # ('userp2', ''),
    # ('userp3', ''),
    # ('userp4', ''),
    # ('userp5', ''),
    # ('wdw', ''),
    # ('xdim', ''),
    # ('ymax_p', ''),
    # ('ymin_p', ''),
    # ('zgoptns', ''),
    # ('zl1', ''),
    # ('zl2', ''),
    # ('zl3', ''),
    # ('zl4', ''),
]

# ======================================================================================
# Digital filter functions
# ======================================================================================
# Extracted from nmrglue.fileio.bruker.py (BSD License)

# Table of points to frequency shift Bruker data to remove digital filter
# (Phase is 360 degrees * num_pts)
# This table is an 'un-rounded' version base on the table by
# W.M. Westler and F. Abildgaard's offline processing note, online at:
# http://www.boc.chem.uu.nl/static/local/prospectnd/dmx_digital_filters.html
# and the updated table with additional entries at:
# http://sbtools.uchc.edu/help/nmr/nmr_toolkit/bruker_dsp_table.asp

# The rounding in the above tables appear to be based on k / (2*DECIM)
# for example 2: 44.75   = 44 + 3/4
#             4: 66.625  = 66 + 5/8
#             8: 68.563 ~= 68 + 9/16 = 68.5625
# Using this the un-rounded table was created by checking possible unrounded
# fracions which would round to those in the original table.

bruker_dsp_table = {
    10: {
        2: 44.75,
        3: 33.5,
        4: 66.625,
        6: 59.083333333333333,
        8: 68.5625,
        12: 60.375,
        16: 69.53125,
        24: 61.020833333333333,
        32: 70.015625,
        48: 61.34375,
        64: 70.2578125,
        96: 61.505208333333333,
        128: 70.37890625,
        192: 61.5859375,
        256: 70.439453125,
        384: 61.626302083333333,
        512: 70.4697265625,
        768: 61.646484375,
        1024: 70.48486328125,
        1536: 61.656575520833333,
        2048: 70.492431640625,
    },
    11: {
        2: 46.0,
        3: 36.5,
        4: 48.0,
        6: 50.166666666666667,
        8: 53.25,
        12: 69.5,
        16: 72.25,
        24: 70.166666666666667,
        32: 72.75,
        48: 70.5,
        64: 73.0,
        96: 70.666666666666667,
        128: 72.5,
        192: 71.333333333333333,
        256: 72.25,
        384: 71.666666666666667,
        512: 72.125,
        768: 71.833333333333333,
        1024: 72.0625,
        1536: 71.916666666666667,
        2048: 72.03125,
    },
    12: {
        2: 46.0,
        3: 36.5,
        4: 48.0,
        6: 50.166666666666667,
        8: 53.25,
        12: 69.5,
        16: 71.625,
        24: 70.166666666666667,
        32: 72.125,
        48: 70.5,
        64: 72.375,
        96: 70.666666666666667,
        128: 72.5,
        192: 71.333333333333333,
        256: 72.25,
        384: 71.666666666666667,
        512: 72.125,
        768: 71.833333333333333,
        1024: 72.0625,
        1536: 71.916666666666667,
        2048: 72.03125,
    },
    13: {
        2: 2.75,
        3: 2.8333333333333333,
        4: 2.875,
        6: 2.9166666666666667,
        8: 2.9375,
        12: 2.9583333333333333,
        16: 2.96875,
        24: 2.9791666666666667,
        32: 2.984375,
        48: 2.9895833333333333,
        64: 2.9921875,
        96: 2.9947916666666667,
    },
}


def _remove_digital_filter(dic, data):
    """
    Remove the digital filter from Bruker data.

    nmrglue modified Digital Filter Processing.
    """
    if "acqus" not in dic:
        raise KeyError("dictionary does not contain acqus parameters")

    if "DECIM" not in dic["acqus"]:
        raise KeyError("dictionary does not contain DECIM parameter")
    decim = dic["acqus"]["DECIM"]

    if "DSPFVS" not in dic["acqus"]:
        raise KeyError("dictionary does not contain DSPFVS parameter")
    dspfvs = dic["acqus"]["DSPFVS"]

    grpdly = dic["acqus"].get("GRPDLY", 0)

    if grpdly > 0:  # use group delay value if provided (not 0 or -1)
        phase = grpdly

    # Determine the phase correction
    elif dspfvs >= 14:  # DSPFVS greater than 14 give no phase correction.
        phase = 0.0
    else:
        if dspfvs < 10:
            dspfvs = 10  # default for DQD  # loop up the phase in the table
        if dspfvs not in bruker_dsp_table:
            raise KeyError("dspfvs not in lookup table")
        if decim not in bruker_dsp_table[dspfvs]:
            raise KeyError("decim not in lookup table")
        phase = bruker_dsp_table[dspfvs][decim]
    # fft
    si = data.shape[-1]
    pdata = np.fft.fftshift(np.fft.fft(data, si, axis=-1), -1) / float(si / 2)
    pdata = (pdata.T - pdata.T[0]).T  # remove Bruker smile

    # Phasing
    si = float(pdata.shape[-1])
    ph = 2.0j * np.pi * phase * np.arange(si) / si
    pdata = pdata * np.exp(ph)

    # ifft
    data = np.fft.ifft(np.fft.ifftshift(pdata, -1), n=int(si), axis=-1) * float(si / 2)

    # remove last points * 2
    rp = 2 * (phase // 2)
    td = dic["acqus"]["TD"] // 2
    td = int(td) - int(rp)
    dic["acqus"]["TD"] = td * 2
    return data[..., :td]


# ======================================================================================
# Bruker topspin import function
# ======================================================================================


def read_topspin(*paths, **kwargs):
    r"""
    Open TopSpin Bruker NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` object objects or valid urls, optional
        The data source(s) can be specified by the name or a list of name for the
        file(s) to be loaded:

        - e.g., ( filename1, filename2, ..., kwargs )

        If the list of filenames are enclosed into brackets:

        - e.g., ( [filename1, filename2, ...], kwargs )

        The returned datasets are merged to form a single dataset,
        except if ``merge`` is set to `False`.
    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s). When several datasets are returned, the
        result is a list-like `ScpObjectList`.

    Other Parameters
    ----------------
    content : `bytes` object, optional
        Instead of passing a filename for further reading, a bytes content can be
        directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly
        useful for a web application to handle drag and drop of files into a
        Browser.
    csv_delimiter : `str`, optional, default: `~spectrochempy.preferences.csv_delimiter`
        Set the column delimiter in CSV file.
    description : `str`, optional
        A Custom description.
    directory : `~pathlib.Path` object objects or valid urls, optional
        From where to read the files.
    download_only: `bool`, optional, default: `False`
        Used only when url are specified.  If True, only downloading and saving of the
        files is performed, with no attempt to read their content.
    merge : `bool`, optional, default: `False`
        If `True` and several filenames or a ``directory`` have been provided as
        arguments, then a single `NDDataset` with merged dataset (stacked along the first
        dimension) is returned. In the case not all datasets have compatible dimensions or types/origins,
        then several NDDatasets can be returned for different groups of compatible datasets.
    origin : str, optional
        If provided it may be used to define the type of experiment: e.g., 'ir', 'raman',..
        or the origin of the data, e.g., 'omnic', 'opus', ... It is often provided by the reader
        automatically, but can be set manually.

        It is used for instance when reading directory with different types of files, for merging
        the datasets with compatible dimensions and different origin into different groups.

        It is also used when reading with the CSV protocol. In order to properly interpret CSV file
        it can be necessary to set the origin of the spectra. Up to now only ``'omnic'`` and ``'tga'``
        have been implemented.
    pattern : `str`, optional
        A pattern to filter the files to read.

        .. versionadded:: 0.7.2
    protocol : `str`, optional
        ``Protocol`` used for reading. It can be one of {``'scp'``, ``'omnic'``,
        ``'opus'``, ``'topspin'``, ``'matlab'``, ``'jcamp'``,
        ``'csv'``, ``'excel'``}. If not provided, the correct protocol
        is inferred (whenever it is possible) from the filename extension.
    read_only: `bool`, optional, default: `True`
        Used only when url are specified.  If True, saving of the
        files is performed in the current directory, or in the directory specified by
        the directory parameter.
    recursive : `bool`, optional, default: `False`
        Read also in subfolders.
    replace_existing: `bool`, optional, default: `False`
        Used only when url are specified. By default, existing files are not replaced
        so not downloaded.
    sortbydate : `bool`, optional, default: `True`
        Sort multiple filename by acquisition date.

    See Also
    --------
    read : Generic reader inferring protocol from the filename extension.
    :func:`spectrochempy.read_zip` : Read Zip archives (containing spectrochempy readable files)
    :func:`spectrochempy.read_dir` : Read an entire directory.
    :func:`spectrochempy.read_opus` : Read OPUS spectra.
    :func:`spectrochempy.read_labspec` : Read Raman LABSPEC spectra (:file:`.txt`).
    :func:`spectrochempy.read_omnic` : Read Omnic spectra (:file:`.spa`, :file:`.spg`, :file:`.srs`).
    :func:`spectrochempy.read_soc` : Read Surface Optics Corps. files (:file:`.ddr` , :file:`.hdr` or :file:`.sdr`).
    :func:`spectrochempy.read_spc` : Read Galactic files (:file:`.spc`).
    :func:`spectrochempy.read_quadera` : Read a Pfeiffer Vacuum's QUADERA mass spectrometer software file.
    :func:`spectrochempy.read_csv` : Read CSV files (:file:`.csv`).
    :func:`spectrochempy.read_matlab` : Read Matlab files (:file:`.mat`, :file:`.dso`).
    :func:`spectrochempy.read_jcamp` : Read Infrared JCAMP-DX files (:file:`.jdx`, :file:`.dx`).
    :func:`spectrochempy.carroucell.read` : Read files in a directory after a carroucell experiment.
    :func:`spectrochempy.read_wire` : Read Renishaw Wire files (:file:`.wdf`).

    Examples
    --------
    Reading a single TopSpin file

    >>> scp.nmr.read('irdata/topspin/1/pdata/1/1r')
    NDDataset: [float64] a.u. (shape: (y:1, x:16384))

    """
    kwargs["filetypes"] = ["TopSpin files (1r, 1i, *dir/*, 1r/*)"]
    kwargs["protocol"] = ["topspin"]
    importer = Importer()
    return importer(*paths, **kwargs)


def _get_files(path, typ="acqu"):
    files = []
    for i in ["", 2, 3]:
        f = path / f"{typ}{i}"
        if f.exists():
            files.append(f)
        f = path / f"{typ}{i}s"
        if f.exists():
            files.append(f)
    return files


@_importer_method
def _read_topspin(*args, **kwargs):
    dataset, path = args

    # is-it a processed dataset file (1r, 2rr ....) ?
    processed = bool(path.match("pdata/*/*"))

    # ----------------------------------------------------------------------------------
    # start reading ....
    # ----------------------------------------------------------------------------------
    parents = path.parents

    # Get data and acquisition parameters

    if not processed:
        # a fid or a ser has been selected
        f_expno = parents[0]
        expno = f_expno.name
        procno = kwargs.get("procno", "1")
        f_procno = f_expno / "pdata" / procno
        f_name = parents[1]

    else:
        # a processes spectra has been selected (1r, ....)
        f_procno = parents[0]
        procno = f_procno.name
        f_expno = parents[2]
        expno = f_expno.name
        f_name = parents[3]

    acqus_files = _get_files(f_expno, "acqu")
    procs_files = _get_files(f_procno, "proc")

    if not processed:
        dic, data = read_fid(f_expno, acqus_files=acqus_files, procs_files=procs_files)

        # Keep the raw 1D FID in nmrglue's direct-complex orientation.
        # The public 1D QSIM FFT path already applies the conjugated Fourier
        # convention needed to recover the TopSpin spectral orientation from
        # that raw signal.  Injecting an additional -90° rotation here would
        # leave the final 1D spectrum globally quadrature-shifted relative to
        # the bundled TopSpin `1r`/`1i` reference.
        #
        # Historical 2D SER work used the imported -90° direct-dimension
        # rotation. Keep that behavior there until the separate 2D campaign is
        # revisited.
        if path.name == "ser":
            data = data * np.exp(-1j * np.pi / 2.0)

        # Look the case when the reshaping was not correct
        # for example, this happens when the number
        # of accumulated row was incomplete
        if path.name in ["ser"] and data.ndim == 1:
            # we must reshape using the acqu parameters
            _acqu2 = dic.get("acqu2s", dic.get("acqu2", {}))
            _acqu = dic.get("acqus", dic.get("acqu", {}))
            td1 = _acqu2.get("TD")
            if td1 is not None:
                try:
                    data = data.reshape(td1, -1)
                except ValueError:
                    td = _acqu.get("TD")
                    if td is not None:
                        data = data.reshape(-1, td // 2)
                    else:
                        raise KeyError(
                            "Inconsistency between TD's and data size"
                        ) from None

            # reduce to td
            ntd = _acqu.get("TD", dic.get("acqus", {}).get("TD", 0)) // 2
            data = data[..., :ntd]

        # Eliminate the digital filter
        if kwargs.get("remove_digital_filter", True) and dic["acqus"]["DECIM"] > 1:
            data = _remove_digital_filter(dic, data)

    else:
        dic, datalist = read_pdata(
            f_procno,
            acqus_files=acqus_files,
            procs_files=procs_files,
            all_components=True,
        )
        if isinstance(datalist, list):
            if datalist[0].ndim == 2:
                data, dataRI, dataIR, dataII = datalist
                # make quaternion
                from spectrochempy_hypercomplex import as_quat_array  # noqa: PLC0415

                shape = data.shape
                data = as_quat_array(
                    list(
                        zip(
                            data.flatten(),
                            dataRI.flatten(),
                            dataIR.flatten(),
                            dataII.flatten(),
                            strict=False,
                        ),
                    ),
                )
                data = data.reshape(shape)

            elif datalist[0].ndim == 1:
                # make complex
                data, dataI = datalist
                data = data + dataI * 1.0j

            else:
                return None
        else:
            data = datalist

    # we now make some rearrangement of the dic to have something more user-friendly
    # we assume that all experiments have similar (important) parameters so that the experiments are compatibles

    meta = Meta()  # This is the parameter dictionary
    datatype = path.name.upper() if not processed else f"{data.ndim}D"

    keys = sorted(dic.keys())

    # we need the ndim of the data
    parmode = int(dic["acqus"].get("PARMODE", data.ndim - 1))
    if parmode + 1 != data.ndim:
        raise KeyError(
            f"The NMR data were not read properly as the PARMODE+1 parameter ({parmode + 1}) doesn't fit"
            f" the actual number of dimensions ({data.ndim})",
        )

    # Only 1D and 2D data are supported; 3D/4D would require a different
    # hypercomplex representation than the quaternion-based one used here.
    if parmode >= 2:
        raise NotImplementedError(
            "TopSpin reader supports 1D and 2D NMR data only. "
            f"Received {parmode + 1}D data."
        )

    # read the acqu and proc
    valid_keys = list(zip(*nmr_valid_meta, strict=False))[0]
    keys_units = dict(nmr_valid_meta)

    for item in keys:
        if item[:4] in ["acqu", "proc"]:
            dim = parmode
            if len(item) > 4 and item[4] in ["2", "3"]:
                dim = parmode + 1 - int(item[4])

            for key in sorted(dic[item]):
                if key.startswith("_") or key.lower() not in valid_keys:
                    continue

                value = dic[item][key]
                units = ur(keys_units[key.lower()]) if keys_units[key.lower()] else None

                if units is not None:
                    if isinstance(value, float | int):
                        value = value * units  # make a quantity
                    elif isinstance(value, list) and isinstance(value[0], float | int):
                        value = np.array(value) * units

                if key.lower() not in meta:
                    meta[key.lower()] = [None] * data.ndim

                with contextlib.suppress(TypeError, IndexError):
                    meta[key.lower()][dim] = value

        else:
            meta[item.lower()] = dic[item]

    # Warning: from now all parameter keys are lowercase.

    # correct some initial values

    meta.encoding = [0] * (parmode + 1)
    meta.iscomplex = [False] * (parmode + 1)

    if not processed:
        meta.isfreq = [False]
        meta.encoding[-1] = AQ_mod[meta.aq_mod[-1]]
        meta.iscomplex[-1] = meta.aq_mod[-1] > 0

    if datatype in ["SER"]:
        meta.isfreq.insert(0, False)

        # Explicitly read indirect dimension encoding from acqu2s.
        # The generic metadata loop places acqus FnMODE at the direct
        # dimension, but for SER data it describes indirect dimension encoding.
        _acqu2s = dic.get("acqu2s", {})
        _fnmode_val = (
            _acqu2s.get("FnMODE")
            if _acqu2s.get("FnMODE") is not None
            else _acqu2s.get("fnmode")
        )
        if _fnmode_val is None:
            _fnmode_val = meta.fnmode[-2]

        _mc2_val = None
        if meta.mc2 is not None:
            _mc2_val = (
                _acqu2s.get("MC2")
                if _acqu2s.get("MC2") is not None
                else _acqu2s.get("mc2")
            )
            if _mc2_val is None:
                _mc2_val = meta.mc2[-2]

        # For historical reasons,
        # MC2 is interpreted when the acquisition status
        # parameter FnMODE has the value undefined, i.e. 0
        if _fnmode_val == 0 and _mc2_val is not None:
            _fnmode_val = _mc2_val + 1

        if _fnmode_val is not None:
            meta.fnmode[-2] = _fnmode_val
            meta.encoding[-2] = FnMODE[_fnmode_val]
            meta.iscomplex[-2] = _fnmode_val > 1

    # correct TD, so it is the number of complex points, not the number of data
    # not for the last dimension which is already correct
    meta.tdeff = meta.td[:]
    meta.td = list(data.shape)

    # The td adjustment for complex axes (except last) assumes quaternion/hypercomplex
    # conversion which is handled by the spectrochempy-hypercomplex plugin. Without it
    # the raw data shape must be preserved so that coordinates match.
    try:
        from spectrochempy_hypercomplex import (  # noqa: PLC0415
            is_available as _hypercomplex_available,
        )
    except ModuleNotFoundError:
        _hypercomplex_available = False

    _needs_ser_quaternion = False

    for axis in range(parmode + 1):
        if meta.iscomplex[axis]:
            if axis != parmode and _hypercomplex_available:  # noqa: SIM102
                meta.td[axis] = meta.td[axis] // 2
                if datatype == "SER":
                    _needs_ser_quaternion = True
                else:
                    # Non-TopSpin readers still represent indirect hypercomplex
                    # pairing as a single complex row before quaternion packing.
                    slices = [slice(None)] * data.ndim
                    slices[axis] = slice(0, None, 2)
                    data_even = data[tuple(slices)]
                    slices[axis] = slice(1, None, 2)
                    data_odd = data[tuple(slices)]
                    data = data_even + 1j * data_odd
            meta.tdeff[axis] = meta.tdeff[axis] // 2

    meta.sw_h = [
        (meta.sw[axis].m * meta.sfo1[axis] * 1e-6).to("Hz")
        if meta.sw[axis] is not None and meta.sfo1[axis] is not None
        else None
        for axis in range(parmode + 1)
    ]

    if processed:
        meta.si = list(data.shape)
        meta.isfreq = [True] * (parmode + 1)  # at least we assume this
        # phc0 is already populated from procs by the metadata loop above;
        # only initialise if missing (e.g. no procs file found).
        if not hasattr(meta, "phc0") or meta.phc0 is None:
            meta.phc0 = [0] * data.ndim

    # For multi-dimensional and already-processed direct-complex data we keep
    # the historical imported convention adjustment.  For raw 1D FIDs, however,
    # the public QSIM FFT path already applies the required conjugated Fourier
    # convention; conjugating the raw FID here would cancel that convention and
    # produce a spectrum inconsistent with the bundled TopSpin 1r reference.
    if meta.iscomplex[-1] and (processed or parmode > 0):
        data = np.conj(data * np.exp(np.pi * 1j / 2.0))

    # normalised amplitudes to ns=1 and rg=1
    def _norm(dat):
        meta.ns = meta.get(
            "ns",
            [1] * data.ndim,
        )  # sometimes these parameters are not present
        meta.rg = meta.get("rg", [1.0] * data.ndim)
        if processed:
            # TopSpin pdata amplitudes already carry the scan averaging.
            # Dividing them again by NS makes the treated vendor reference
            # artificially smaller than the spectrum reconstructed from the raw
            # FID with the public pipeline.
            fac = float(meta.rg[-1])
        else:
            fac = float(meta.ns[-1]) * float(meta.rg[-1])
        meta.rgold = [meta.rg[-1]]
        meta.rg[-1] = 1.0
        meta.nsold = [meta.ns[-1]]  # store the old value of NS
        meta.ns[-1] = 1
        if fac > 0:
            dat /= fac
        return dat

    data = _norm(data)

    if _needs_ser_quaternion:
        from spectrochempy_hypercomplex import as_quaternion  # noqa: PLC0415

        # TopSpin SER is already complex along F2 after read_fid() and digital
        # filter removal. Build quaternion directly from the paired F1 rows so
        # the direct dimension keeps its full complex length.
        data = as_quaternion(data[0::2], data[1::2])
        meta.td = list(data.shape)

    # add some additional information in meta
    meta.expno = [int(expno)]

    # and the metadata (and make them readonly)
    meta.datatype = datatype
    meta.pathname = str(path)

    # add two parameters needed for phasing
    meta.pivot = [0] * data.ndim
    meta.exptc = [0] * data.ndim

    # make the corresponding axis
    # debug_('Create coords...')
    coords = []
    axe_range = list(range(parmode + 1))

    use_list = kwargs.pop("use_list", False)

    for axis in axe_range:
        if parmode > 0 and use_list and axis == 0:
            # we use the vd or other list to make the axis
            # this is useful for pseudo 2D data such as relaxation, etc...
            if not isinstance(use_list, str):
                use_list = "vdlist"
            use_list = f_expno / use_list
            try:
                with open(use_list) as f:
                    vd = [float(val) for val in f.readlines()]
            except (FileNotFoundError, ValueError) as exc:
                warning_(
                    f"Could not read use_list file {use_list}: {exc}", stacklevel=2
                )
                vd = []
            coord = Coord(vd, title="time", units="s")
            coords.append(coord)

        elif not meta.isfreq[axis]:
            # the axis is in time units
            dw = (1.0 / meta.sw_h[axis]).to("us")
            coordpoints = np.arange(meta.td[axis])
            coord = Coord(
                coordpoints * dw,
                title=f"F{axis + 1} acquisition time",
            )

            coord.meta["acquisition_frequency"] = meta.sfo1[axis]
            coords.append(coord)
        else:
            size = meta.si[axis]
            sizem = max(size - 1, 1)
            deltaf = -meta.sw_h[axis] / sizem
            first = meta.sfo1[axis] - meta.sf[axis] - deltaf * sizem / 2.0

            coordpoints = np.arange(size) * deltaf + first
            coord = Coord(coordpoints)
            coord.meta["acquisition_frequency"] = meta.sfo1[axis]
            coord.ito("ppm")
            if meta.nuc1 is not None:
                nuc1 = meta.nuc1[axis]
                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, nuc1)
                if m is not None:
                    mass = m[1]
                    name = m[2]
                    nucleus = "^{" + mass + "}" + name
                else:
                    nucleus = nuc1
            else:
                nucleus = ""
            coord.title = rf"$\delta\ {nucleus}$"
            coords.append(coord)

    dataset.data = data

    for axis, cplex in enumerate(meta.iscomplex[::-1]):
        if cplex and axis > 0:
            try:
                dataset.hyper.set_quaternion(inplace=True)
                # set_quaternion halves the last complex dimension.
                # Update meta.td and rebuild the corresponding coordinate.
                meta.td[-1] = dataset.data.shape[-1]
                if not meta.isfreq[-1]:
                    dw = (1.0 / meta.sw_h[-1]).to("us")
                    coordpoints = np.arange(meta.td[-1])
                    coords[-1] = Coord(
                        coordpoints * dw,
                        title=coords[-1].title,
                    )
            except AttributeError:
                warning_(
                    "2D hypercomplex NMR data requires the spectrochempy-hypercomplex "
                    "plugin. Install it with: pip install spectrochempy-hypercomplex",
                    stacklevel=2,
                )

    dataset.meta.update(meta)
    dataset.meta.readonly = True
    dataset.set_coordset(*tuple(coords))

    dataset.units = "count"
    dataset.title = "intensity"
    dataset.origin = "topspin"
    dataset.name = f"{f_name.name} expno:{expno} procno:{procno} ({datatype})"
    dataset.filename = f_name
    if dataset.meta.date is not None:
        with contextlib.suppress(ValueError, OSError, TypeError, OverflowError):
            dataset.acquisition_date = datetime.fromtimestamp(dataset.meta.date[-1])

    dataset.history = "Imported from TopSpin dataset"

    return dataset
