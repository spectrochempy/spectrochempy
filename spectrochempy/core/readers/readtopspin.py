# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

"""
Bruker file (single dimension FID or multidimensional SER) importers.
"""

__all__ = ["read_topspin", "read_bruker_nmr"]
__dataset_methods__ = __all__

import re

import numpy as np
from quaternion import as_quat_array

from nmrglue.fileio.bruker import read as read_fid, read_pdata, read_lowmem
from spectrochempy.core import debug_
from spectrochempy.core.dataset.meta import Meta
from spectrochempy.core.dataset.coord import LinearCoord
from spectrochempy.units import ur
from spectrochempy.utils.exceptions import deprecated
from spectrochempy.core.readers.importer import Importer, importermethod

# ======================================================================================================================
# Constants
# ======================================================================================================================

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
    ("date", ""),  # ('dbl', ''),
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
]

# ======================================================================================================================
# Digital filter functions
# ======================================================================================================================
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
    nmrglue modified Digital Filter Processing
    """
    if "acqus" not in dic:
        raise KeyError("dictionary does not contain acqus parameters")

    if "DECIM" not in dic["acqus"]:
        raise KeyError("dictionary does not contain DECIM parameter")
    decim = dic["acqus"]["DECIM"]

    if "DSPFVS" not in dic["acqus"]:
        raise KeyError("dictionary does not contain DSPFVS parameter")
    dspfvs = dic["acqus"]["DSPFVS"]

    if "GRPDLY" not in dic["acqus"]:
        grpdly = 0
    else:
        grpdly = dic["acqus"]["GRPDLY"]

    if grpdly > 0:  # use group delay value if provided (not 0 or -1)
        phase = grpdly

    # Determine the phase correction
    else:
        if dspfvs >= 14:  # DSPFVS greater than 14 give no phase correction.
            phase = 0.0
        else:
            if dspfvs < 11:
                dspfvs = 11  # default for DQD  # loop up the phase in the table
            if dspfvs not in bruker_dsp_table:
                raise KeyError("dspfvs not in lookup table")
            if decim not in bruker_dsp_table[dspfvs]:
                raise KeyError("decim not in lookup table")
            phase = bruker_dsp_table[dspfvs][decim]
    # fft
    si = data.shape[-1]
    pdata = np.fft.fftshift(np.fft.fft(data, si, axis=-1), -1) / float(si / 2)
    pdata = (pdata.T - pdata.T[0]).T  # TODO: this allow generally to
    # TODO: remove Bruker smiles, not so sure actually

    # Phasing
    si = float(pdata.shape[-1])
    ph = 2.0j * np.pi * phase * np.arange(si) / si
    pdata = pdata * np.exp(ph)

    # ifft
    data = np.fft.ifft(np.fft.ifftshift(pdata, -1), si, axis=-1) * float(si / 2)

    # remove last points * 2
    rp = 2 * (phase // 2)
    td = dic["acqus"]["TD"] // 2
    td = int(td) - int(rp)
    dic["acqus"]["TD"] = td * 2
    data = data[..., :td]

    # debug_('Bruker digital filter : removed %s points' % rp)

    return data


# def _scale(meta, dim=-1, reverse=None):
#     """
#     private function: Compute scale for a given axis.
#     """
#
#     # import parameter to convert units
#     sw = float(meta.sw_h[dim])
#     sfo1 = float(meta.sfo1[dim])
#     bf1 = float(meta.bf1[dim])
#     sf = float(meta.sf[dim])
#     si = max(float(meta.si[dim])-1, 1)
#     td = float(meta.td[dim])
#
#     sr = (sf - bf1) * 1.0e6
#     o1 = (sfo1 - bf1) * 1.0e6
#
#     # set the spectral parameters
#     # (si,   sw_h, bf1, -sr + o1)
#     # size,   sw, obs,      car)  (correspondance with nmrglue udic)
#
#     # derived units (these are in ppm)
#
#     fact = 2.0 if meta.fnmode[dim] in [3, 4, 5, 6] or else 1.0
#     if meta.isfreq[dim]:
#         delta = -sw * fact / (si * bf1)
#         first = (-sr + o1)/ bf1 - delta * si / 2.
#
#     if reverse is None:
#         reverse = meta.reverse
#
#     if reverse:
#         return scal()[::-1]
#     else:
#         return scal()


# ======================================================================================================================
# Bruker topspin import function
# ======================================================================================================================
def read_topspin(*paths, **kwargs):
    """
    Open Bruker TOPSPIN (NMR) dataset.

    Parameters
    ----------
    *paths : str, optional
        Paths of the Bruker directories to read.
    **kwargs : dict
        See other parameters.

    Returns
    --------
    read_topspin
        |NDDataset| or list of |NDDataset|.

    Other Parameters
    ----------------
    expno : int, optional
        experiment number.
    procno : int
        processing number.
    protocol : {'scp', 'omnic', 'opus', 'topspin', 'matlab', 'jcamp', 'csv', 'excel'}, optional
        Protocol used for reading. If not provided, the correct protocol
        is inferred (whnever it is possible) from the file name extension.
    directory : str, optional
        From where to read the specified `filename`. If not specified, read in the default ``datadir`` specified in
        SpectroChemPy Preferences.
    merge : bool, optional
        Default value is False. If True, and several filenames have been provided as arguments,
        then a single dataset with merged (stacked along the first
        dimension) is returned (default=False).
    sortbydate : bool, optional
        Sort multiple spectra by acquisition date (default=True).
    description : str, optional
        A Custom description.
    origin : {'omnic', 'tga'}, optional
        In order to properly interpret CSV file it can be necessary to set the origin of the spectra.
        Up to now only 'omnic' and 'tga' have been implemented.
    csv_delimiter : str, optional
        Set the column delimiter in CSV file.
        By default it is the one set in SpectroChemPy ``Preferences``.
    content : bytes object, optional
        Instead of passing a filename for further reading, a bytes content can be directly provided as bytes objects.
        The most convenient way is to use a dictionary. This feature is particularly useful for a GUI Dash application
        to handle drag and drop of files into a Browser.
        For exemples on how to use this feature, one can look in the ``tests/tests_readers`` directory.
    listdir : bool, optional
        If True and filename is None, all files present in the provided `directory` are returned (and merged if `merge`
        is True. It is assumed that all the files correspond to current reading protocol (default=True)
    recursive : bool, optional
        Read also in subfolders. (default=False)

    See Also
    --------
    read_topspin : Read TopSpin Bruker NMR spectra.
    read_omnic : Read Omnic spectra.
    read_opus : Read OPUS spectra.
    read_labspec : Read Raman LABSPEC spectra.
    read_spg : Read Omnic *.spg grouped spectra.
    read_spa : Read Omnic *.Spa single spectra.
    read_srs : Read Omnic series.
    read_csv : Read CSV files.
    read_zip : Read Zip files.
    read_matlab : Read Matlab files.
    """
    kwargs["filetypes"] = [
        "Bruker TOPSPIN fid's or processed data files (fid ser 1[r|i] 2[r|i]* 3[r|i]*)",
        "Compressed TOPSPIN data directories (*.zip)",
    ]
    kwargs["protocol"] = ["topspin"]

    importer = Importer()
    return importer(*paths, **kwargs)


@deprecated(
    "read_bruker_nmr reading method is deprecated and may be removed in next versions "
    "- use read_topspin instead"
)
def read_bruker_nmr(*args, **kwargs):
    return read_topspin(*args, **kwargs)


# ..............................................................................
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


@importermethod
def _read_topspin(*args, **kwargs):
    debug_("Bruker TOPSPIN import")

    dataset, path = args
    #    content = kwargs.get('content', None)

    # is-it a processed dataset (1r, 2rr ....
    processed = True if path.match("pdata/*/*") else False

    # low memory handling (lowmem) ?
    lowmem = kwargs.get("lowmem", False)  # load all in numero by default

    # ------------------------------------------------------------------------
    # start reading ....
    # ------------------------------------------------------------------------

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

        if not lowmem:
            dic, data = read_fid(
                f_expno, acqus_files=acqus_files, procs_files=procs_files
            )
        else:
            dic, data = read_lowmem(
                f_expno, acqus_files=acqus_files, procs_files=procs_files
            )

        # apply a -90 phase shift to be compatible with topspin
        data = data * np.exp(-1j * np.pi / 2.0)

        # Look the case when the reshaping was not correct
        # for example, this happen when the number
        # of accumulated row was incomplete
        if path.name in ["ser"] and data.ndim == 1:
            # we must reshape using the acqu parameters
            td1 = dic["acqu2"]["TD"]
            try:
                data = data.reshape(td1, -1)
            except ValueError:
                try:
                    td = dic["acqu"]["TD"] // 2
                    data = data.reshape(-1, td)
                except ValueError:
                    raise KeyError("Inconsistency between TD's and data size")

            # reduce to td
            ntd = dic["acqus"]["TD"] // 2
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
                shape = data.shape
                data = as_quat_array(
                    list(
                        zip(
                            data.flatten(),
                            dataRI.flatten(),
                            dataIR.flatten(),
                            dataII.flatten(),
                        )
                    )
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

    # ........................................................................................................
    # we now make some rearrangement of the dic to have something more user friendly
    # we assume that all experiments have similar (important) parameters so that the experiments are compatibles

    meta = Meta()  # This is the parameter dictionary
    datatype = path.name.upper() if not processed else f"{data.ndim}D"

    keys = sorted(dic.keys())

    # we need the ndim of the data
    parmode = int(dic["acqus"].get("PARMODE", data.ndim - 1))
    if parmode + 1 != data.ndim:
        raise KeyError(
            f"The NMR data were not read properly as the PARMODE+1 parameter ({parmode + 1}) doesn't fit"
            f" the actual number of dimensions ({data.ndim})"
        )

    # read the acqu and proc
    valid_keys = list(zip(*nmr_valid_meta))[0]
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
                    if isinstance(value, (float, int)):
                        value = value * units  # make a quantity
                    elif isinstance(value, list) and isinstance(value[0], (float, int)):
                        value = np.array(value) * units

                if key.lower() not in meta:
                    meta[key.lower()] = [None] * data.ndim

                try:
                    meta[key.lower()][dim] = value
                except Exception:
                    pass

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

        if meta.fnmode[-2] == 0:
            # For historical reasons,
            # MC2 is interpreted when the acquisition status
            # parameter FnMODE has the value undefined, i.e. 0
            if meta.mc2 is not None:
                meta.fnmode[-2] = meta.mc2[-2] + 1

        meta.encoding[-2] = FnMODE[meta.fnmode[-2]]
        meta.iscomplex[-2] = meta.fnmode[-2] > 1

        if parmode == 2:
            meta.isfreq.insert(0, False)
            if meta.fnmode[-3] == 0 and meta.mc2 is not None:
                meta.fnmode[-3] = meta.mc2[-3] + 1
            meta.encoding[-3] = FnMODE[meta.fnmode[-3]]
            meta.iscomplex[-3] = meta.fnmode[-3] > 1

    # correct TD, so it is the number of complex points, not the number of data
    # not for the last dimension which is already correct
    meta.tdeff = meta.td[:]
    meta.td = list(data.shape)

    for axis in range(parmode + 1):
        if meta.iscomplex[axis]:
            if axis != parmode:  # already done for last axis
                meta.td[axis] = meta.td[axis] // 2
            meta.tdeff[axis] = meta.tdeff[axis] // 2

    meta.sw_h = [
        (meta.sw[axis].m * meta.sfo1[axis] * 1e-6).to("Hz")
        for axis in range(parmode + 1)
    ]

    if processed:
        meta.si = [si for si in data.shape]
        meta.isfreq = [True] * (parmode + 1)  # at least we assume this
        meta.phc0 = [0] * data.ndim

    # this transformation is to make data coherent with bruker processsing
    if meta.iscomplex[-1]:
        data = np.conj(data * np.exp(np.pi * 1j / 2.0))

    # normalised amplitudes to ns=1 and rg=1
    def _norm(dat):
        meta.ns = meta.get(
            "ns", [1] * data.ndim
        )  # sometimes these parameters are not present
        meta.rg = meta.get("rg", [1.0] * data.ndim)
        fac = float(meta.ns[-1]) * float(meta.rg[-1])
        meta.rgold = [meta.rg[-1]]
        meta.rg[-1] = 1.0
        meta.nsold = [meta.ns[-1]]  # store the old value of NS
        meta.ns[-1] = 1
        dat /= fac
        return dat

    data = _norm(data)

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

    for axis in axe_range:
        if not meta.isfreq[axis]:
            # the axis is in time units
            dw = (1.0 / meta.sw_h[axis]).to("us")
            # coordpoints = np.arange(meta.td[axis])
            # coord = Coord(coordpoints * dw,
            #             title=f"F{axis + 1} acquisition time")  # TODO: use AQSEQ for >2D data
            coord = LinearCoord(
                offset=0.0,
                increment=dw,
                units="us",
                size=meta.td[axis],
                title=f"F{axis + 1} acquisition time",
            )
            coord.meta.larmor = meta.sfo1[axis]
            coords.append(coord)
        else:
            size = meta.si[axis]
            sizem = max(size - 1, 1)
            deltaf = -meta.sw_h[axis] / sizem
            first = meta.sfo1[axis] - meta.sf[axis] - deltaf * sizem / 2.0

            # coord = Coord(np.arange(size) * deltaf + first)
            coord = LinearCoord(offset=first, increment=deltaf, size=size)
            coord.meta.larmor = meta.sfo1[axis]  # needed for ppm transformation
            coord.ito("ppm")
            if meta.nuc1 is not None:
                nuc1 = meta.nuc1[axis]
                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, nuc1)
                mass = m[1]
                name = m[2]
                nucleus = "^{" + mass + "}" + name
            else:
                nucleus = ""
            coord.title = fr"$\delta\ {nucleus}$"
            coords.append(coord)

    dataset.data = data

    for axis, cplex in enumerate(meta.iscomplex[::-1]):
        if cplex and axis > 0:
            dataset.set_quaternion(inplace=True)

    dataset.meta.update(meta)
    dataset.meta.readonly = True
    dataset.set_coordset(*tuple(coords))

    dataset.title = "intensity"
    dataset.origin = "topspin"
    dataset.name = f"{f_name.name} expno:{expno} procno:{procno} ({datatype})"
    dataset.filename = f_name

    return dataset

    # list_meta.append(meta)  # list_coords.append(coords)  # list_data.append(data)

    # # store also the varpars of the series  # varpars = kargs.get('varpars')  # if isinstance(varpars, str):  #  #
    # varpars = varpars.split()  #     if len(varpars) == 1:  #         # this should be one of the dic parameters  #
    # lvarpars.append(adic.par[varpars[0]])  # store the variable  #     elif len(varpars) == 2:  #  #
    # lvarpars.append(adic.par[varpars[0]][  #                             int(varpars[1])])  # store the variable  #
    # ldates.append(adic.par.DATE)  # and the date  # elif isinstance(varpars, list):  #     # this should be a list
    # of parameters  #     p = []  #     for var in varpars:  #         p.append(adic.par[var])  #  #
    # lvarpars.append(p)  # store the variable  #     ldates.append(adic.par.DATE)  # and the date

    # store temporarily these data  # debug_('data read finished : type : %s' % datatype)  #  #  #  # if len(  #
    # list_data) == 1:  # # debug_('One experiment read. Make it the current dataset')


# def _read_topspin_dir(*args, **kwargs):
#
#     expnos = kwargs.get('expnos', False)  # List of expnos?
#     paths = None
#     if expnos:
#         paths = [os.path.join(path, str(expno)) for expno in expnos]
#     else:
#         paths = [path, ]
#
#     # start loop on the paths
#
#     list_data = []
#     list_meta = []
#     list_coords = []
#
#     for idx, path in enumerate(paths):
#
#         # debug_('Reading %d:%s' % (idx, path))
#
#         # Acquisition parameters
#
#         if os.path.isfile(os.path.join(path, "fid")):
#             datatype = 'FID'
#         elif os.path.isfile(os.path.join(path, "ser")):
#             datatype = 'SER'
#         elif expno is not None:
#             if os.path.isfile(os.path.join(path, str(expno), "fid")):
#                 datatype = 'FID'
#                 path = os.path.join(path, str(expno))
#             elif os.path.isfile(os.path.join(path, str(expno), "ser")):
#                 datatype = 'SER'
#                 path = os.path.join(path, str(expno))
#         else:
#             if not processed:
#                 warning_('No binary fid or ser found in %s.\n'
#                          'Try processed files...' % path)
#                 processed = True
#
#         if os.path.isfile(os.path.join(path, 'pdata', procno, '1r')):
#             if not datatype or processed:
#                 datatype = '1D'
#         elif os.path.isfile(os.path.join(path, 'pdata', procno, '2rr')):
#             if not datatype or processed:
#                 datatype = '2D'
#         elif os.path.isfile(os.path.join(path, 'pdata', procno, '3rrr')):
#             if not datatype or processed:
#                 datatype = '3D'
#         else:
#             if not datatype:
#                 raise KeyError(f"No Bruker binary file could be found in {path}")
#             elif processed:
#                 warning_(f"No processed Bruker binary file could be found in {path}. Use fid's.")
#
#         # we read all parameters file whatever the datatype
#         npath, par_files = _get_par_files(path, procno, processed)
#
#         if datatype in ['FID', 'SER']:
#             if not lowmem:
#                 dic, data = read(npath, acqus_files=par_files,
#                                  read_pulseprogram=False)
#             else:
#                 dic, data = read_lowmem(npath, acqus_files=par_files, read_pulseprogram=False)
#
#             data = data * np.exp(- 1j * np.pi / 2.)  # -90 phase to be compatible with topspin
#
#             # look the case when the reshaping was not correct
#             # for example, this happen when the number
#             # of accumulated row was incomplete
#             if datatype in ['SER'] and data.ndim == 1:
#                 # we must reshape using the acqu parameters
#                 td1 = dic['acqu2']['TD']
#                 try:
#                     data = data.reshape(td1, -1)
#                 except ValueError:
#                     try:
#                         td = dic['acqu']['TD'] // 2
#                         data = data.reshape(-1, td)
#                     except ValueError:
#                         raise KeyError("Inconsistency between TD's and data size")
#
#                 # reduce to td
#                 ntd = dic['acqus']['TD'] // 2
#                 data = data[..., :ntd]
#                 # necessary for agreement with bruker data and phase
#         else:
#
#             # debug_(f'Reading processed {idx}:{path}')
#
#             dic, data = read_pdata(npath, procs_files=par_files, )
#
#         # Clean dict for pdata keys
#         keys = list(dic.keys())
#         for key in keys:
#             if key.startswith('pdata'):
#                 newkey = key.split(os.path.sep)[-1]
#                 dic[newkey] = dic.pop(key)
#
#         # Eliminate the digital filter
#         if datatype in ['FID', 'SER'] and kwargs.get('remove_digital_filter',
#                                                      True):
#             data = _remove_digital_filter(dic, data)
#
#         #
#         ..............................................................................................................
#         # we now make some rearrangement of the dic
#         # to have something more user friendly
#         # we assume that all experiments have similar (important)
#         # parameters so that the experiments are
#         # compatible
#
#         meta = Meta()  # This is the parameter dictionary
#
#         keys = sorted(dic.keys())
#
#         # we need the ndim of the data
#         parmode = int(dic['acqus']['PARMODE'])
#         if parmode + 1 != data.ndim:
#             raise KeyError(f"The NMR data were not read properly as the PARMODE+1 parameter ({parmode + 1}) doesn't
#             fit"
#                            f" the actual number of dimensions ({data.ndim})")
#
#         # read the acqu and proc
#         valid_keys = list(zip(*nmr_valid_meta))[0]
#         keys_units = dict(nmr_valid_meta)
#
#         for item in keys:
#
#             if item[:4] in ['acqu', 'proc']:
#                 dim = parmode
#                 if len(item) > 4 and item[4] in ['2', '3']:
#                     dim = parmode + 1 - int(item[4])
#
#                 for key in sorted(dic[item]):
#                     if key.startswith('_') or key.lower() not in valid_keys:
#                         continue
#
#                     value = dic[item][key]
#                     units = ur(keys_units[key.lower()]) if keys_units[key.lower()] else None
#
#                     if units is not None:
#                         if isinstance(value, (float, int)):
#                             value = value * units  # make a quantity
#                         elif isinstance(value, list) and isinstance(value[0], (float, int)):
#                             value = np.array(value) * units
#
#                     if not item.endswith('s'):  # initial parameter
#
#                         if dim == parmode:
#                             meta[key.lower()] = [value, ]
#                         else:
#                             meta[key.lower()].insert(dim, value)
#
#                     else:  # status parameters (replace initial)
#                         try:
#                             meta[key.lower()][dim] = value
#                         except Exception:
#                             pass
#
#             else:
#
#                 meta[item.lower()] = dic[item]
#
#         # Warning: from now all parameter keys are lowercase.
#
#         # correct some initial values
#
#         meta.encoding = [0] * (parmode + 1)
#         meta.iscomplex = [False] * (parmode + 1)
#
#         if datatype in ['FID', 'SER']:
#             meta.isfreq = [False]
#             meta.encoding[-1] = AQ_mod[meta.aq_mod[-1]]
#             meta.iscomplex[-1] = meta.aq_mod[-1] > 0
#
#         if datatype in ['SER']:
#             meta.isfreq.insert(0, False)
#
#             if meta.fnmode[-2] == 0:
#                 # For historical reasons,
#                 # MC2 is interpreted when the acquisition status
#                 # parameter FnMODE has the value undefined, i.e. 0
#                 if meta.mc2 is not None:
#                     meta.fnmode[-2] = meta.mc2[-2] + 1
#
#             meta.encoding[-2] = FnMODE[meta.fnmode[-2]]
#             meta.iscomplex[-2] = meta.fnmode[-2] > 1
#
#             if parmode == 2:
#                 meta.isfreq.insert(0, False)
#                 if meta.fnmode[-3] == 0 and meta.mc2 is not None:
#                     meta.fnmode[-3] = meta.mc2[-3] + 1
#                 meta.encoding[-3] = FnMODE[meta.fnmode[-3]]
#                 meta.iscomplex[-3] = meta.fnmode[-3] > 1
#
#         # correct TD, so it is the number of complex points, not the number of data
#         # not for the last dimension which is already correct
#         meta.tdeff = meta.td[:]
#         meta.td = list(data.shape)
#
#         for axis in range(parmode + 1):
#             if meta.iscomplex[axis]:
#                 if axis != parmode:  # already done for last axis
#                     meta.td[axis] = meta.td[axis] // 2
#                 meta.tdeff[axis] = meta.tdeff[axis] // 2
#
#         if datatype in ['1D', '2D', '3D']:
#             meta.si = [si for si in data.shape]
#             meta.isfreq = [True] * (parmode + 1)  # at least we assume this
#
#         # this transformation is to make data coherent with bruker processsing
#         if meta.iscomplex[-1]:
#             data = np.conj(data * np.exp(np.pi * 1j / 2.))
#
#         # normalised amplitudes to ns=1 and rg=1
#         def _norm(dat):
#             fac = float(meta.ns[-1]) * float(meta.rg[-1])
#             meta.rgold = [meta.rg[-1]]
#             meta.rg[-1] = 1.
#             meta.nsold = [meta.ns[-1]]  # store the old value of NS
#             meta.ns[-1] = 1
#             dat /= fac
#             return dat
#
#         data = _norm(data)
#
#         # add some additional inforation in meta
#         meta.expno = [int(os.path.basename(path))]
#
#         # and the metadata (and make them readonly)
#         meta.datatype = datatype
#         meta.pathname = path
#         list_meta.append(meta)
#
#         # make the corresponding axis
#         # debug_('Create coords...')
#         coords = []
#         axe_range = list(range(parmode + 1))
#         for axis in axe_range:
#             if not meta.isfreq[axis]:
#                 # the axis is in time units
#                 dw = (1. / meta.sw_h[axis]).to('us')
#                 coordpoints = np.arange(meta.td[axis])
#                 coord = Coord(coordpoints * dw,
#                               title=f"F{axis + 1} acquisition time")  # TODO: use AQSEQ for >2D data
#                 coords.append(coord)
#             else:
#                 raise NotImplementedError('Not yet implemented')
#         list_coords.append(coords)
#
#         # # store also the varpars of the series
#         # varpars = kargs.get('varpars')
#         # if isinstance(varpars, str):
#         #     varpars = varpars.split()
#         #     if len(varpars) == 1:
#         #         # this should be one of the dic parameters
#         #         lvarpars.append(adic.par[varpars[0]])  # store the variable
#         #     elif len(varpars) == 2:
#         #         lvarpars.append(adic.par[varpars[0]][
#         #                             int(varpars[1])])  # store the variable
#         #     ldates.append(adic.par.DATE)  # and the date
#         # elif isinstance(varpars, list):
#         #     # this should be a list of parameters
#         #     p = []
#         #     for var in varpars:
#         #         p.append(adic.par[var])
#         #     lvarpars.append(p)  # store the variable
#         #     ldates.append(adic.par.DATE)  # and the date
#
#         # store temporarily these data
#         # debug_('data read finished : type : %s' % datatype)
#
#         list_data.append(data)
#
#     if len(list_data) == 1:
#         # debug_('One experiment read. Make it the current dataset')
#
#         dataset.data = list_data[0]
#
#         for axis, cplex in enumerate(meta.iscomplex[::-1]):
#             if cplex and axis > 0:
#                 dataset.set_quaternion(inplace=True)
#
#         dataset.meta.update(list_meta[0])
#         dataset.meta.readonly = True
#         dataset.set_coordset(*tuple(list_coords[0]))  # must be a tuple
#         dataset.title = 'intensity'
#         dataset.origin = 'bruker'
#
#     else:
#
#         # TODO: Check this -
#         # case of multiple experiments to merge
#
#         # find difference in data.shape
#
#         diff = False
#         shape = list_data[0].shape
#         for data in list_data[1:]:
#             if np.any(data.shape != shape):
#                 diff = True
#
#         if not diff:
#             # find difference in coordss
#             coords = list_coords[0]
#             for a in list_coords[1:]:
#                 if np.any(a != coords):
#                     diff = True
#
#         if not diff:
#             info_('the experiments look perfectly compatibles'
#                   ' regarding the shape and the axis!')
#
#         # find what are the differences in meta
#         meta = list_meta[0]
#
#         mkeys = set()  # will stroe the modified keys
#         from spectrochempy.utils import dict_compare
#         for i, d1 in enumerate(list_meta):
#             for j, d2 in enumerate(list_meta[i + 1:]):
#                 added, removed, modified, _ = \
#                     dict_compare(d1.to_dict(), d2.to_dict(),
#                                  check_equal_only=False)
#                 mkeys = mkeys.union(added)
#                 mkeys = mkeys.union(removed)
#                 mkeys = mkeys.union(modified)
#
#         # some keys should not vary for homogeneous data but lets the
#         # user decide
#         info_("keys which have varied : %s" % mkeys)
#
#         mkeys = list(mkeys)
#         meta_diffs = {}
#         for key in mkeys:
#             # find the key variation
#             # WARNING: restriction is that we use only the direct observation
#             # dimension to get the values (so a variation in another dimension
#             # will not be seen!
#             if isinstance(meta[key][-1], Quantity):
#                 meta_diffs[key] = [meta[key][-1].m for meta in list_meta]
#                 meta_diffs[key] = meta_diffs[key] * meta[key][-1].units
#             else:
#                 meta_diffs[key] = [meta[key][-1] for meta in list_meta]
#
#         # we can now store the meta for the datset object
#         dataset.meta.update(list_meta[-1])
#         # we additionaly set the list of variable parameters whch will indicate
#         # the orders of the labels for the new axis
#         dataset.meta.labels = mkeys
#         dataset.meta.readonly = True  # and from now no more modifications in metas
#
#         # by default and if it is possible we try to create an homogeneous
#         # NDDataset (needs same TD, SFO1, etc.. and of course same data.shape)
#         merge_method = kwargs.get('merge_method', 'create_array')
#         if merge_method == 'create_array' and not diff:
#             # we create un array with the list of array
#             newdata = np.stack(list_data, axis=0)
#
#             # store it in the current datset
#             dataset.data = newdata
#             # complexity?
#             complexity = [False] + meta.iscomplex
#             for axis, iscomplex in enumerate(complexity):
#                 if iscomplex:
#                     dataset.set_complex(axis)
#
#             # new coords
#             vkey = kwargs.get('var_key', None)
#             vkey = vkey.lower()
#
#             if vkey is not None and vkey in mkeys:
#                 ax = meta_diffs[vkey]
#                 title = vkey
#             else:
#                 ax = np.arange(newdata.shape[0])
#                 title = '-nd-'
#             axis = Coord(ax, title=title)
#
#             labels = []
#             for i in range(len(ax)):
#                 if len(mkeys) > 1:
#                     label = []
#                     for key in mkeys:
#                         label.append(meta_diffs[key][i])
#                 else:
#                     label = meta_diffs[mkeys[-1]][i]
#                 labels.append(label)
#             labels = np.array(labels, dtype=object)
#             axis.labels = labels
#
#             # now make the coords it in the current dataset
#             # only one coords of the list is taken: they are all the same
#             # in principle... if not problem above or the experiments
#             # are not compatibles
#             dataset.coords = [axis] + list_coords[-1]
#             dataset.origin = 'bruker'
#
#     return dataset
