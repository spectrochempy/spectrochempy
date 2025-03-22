# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
#
# This file was generated for SpectroChemPy version: 0.7.3.dev10+g6b5d0168b.d20250313
#
#    ###################################################################################
#    #           DO NOT MODIFY THIS FILE BECAUSE IT IS CREATED AUTOMATICALLY.          #
#    #   ANY MODIFICATION OF THIS FILE WILL BE CANCELLED AFTER THE COMMIT IN GITHUB.   #
#    ###################################################################################
#
#
# ruff: noqa
"""
SpectroChemPy API.
"""

_LAZY_IMPORTS = {
    "PLSRegression": "spectrochempy.analysis.crossdecomposition.pls",
    "polynomialbaseline": "spectrochempy.analysis.curvefitting._models",
    "gaussianmodel": "spectrochempy.analysis.curvefitting._models",
    "lorentzianmodel": "spectrochempy.analysis.curvefitting._models",
    "voigtmodel": "spectrochempy.analysis.curvefitting._models",
    "asymmetricvoigtmodel": "spectrochempy.analysis.curvefitting._models",
    "sigmoidmodel": "spectrochempy.analysis.curvefitting._models",
    "LSTSQ": "spectrochempy.analysis.curvefitting.linearregression",
    "NNLS": "spectrochempy.analysis.curvefitting.linearregression",
    "Optimize": "spectrochempy.analysis.curvefitting.optimize",
    "EFA": "spectrochempy.analysis.decomposition.efa",
    "FastICA": "spectrochempy.analysis.decomposition.fast_ica",
    "IrisKernel": "spectrochempy.analysis.decomposition.iris",
    "IRIS": "spectrochempy.analysis.decomposition.iris",
    "MCRALS": "spectrochempy.analysis.decomposition.mcrals",
    "NMF": "spectrochempy.analysis.decomposition.nmf",
    "PCA": "spectrochempy.analysis.decomposition.pca",
    "SIMPLISMA": "spectrochempy.analysis.decomposition.simplisma",
    "SVD": "spectrochempy.analysis.decomposition.svd",
    "simps": "spectrochempy.analysis.integration.integrate",
    "trapz": "spectrochempy.analysis.integration.integrate",
    "simpson": "spectrochempy.analysis.integration.integrate",
    "trapezoid": "spectrochempy.analysis.integration.integrate",
    "ActionMassKinetics": "spectrochempy.analysis.kinetic.kineticutilities",
    "PFR": "spectrochempy.analysis.kinetic.kineticutilities",
    "find_peaks": "spectrochempy.analysis.peakfinding.peakfinding",
    "DEBUG": "spectrochempy.application.application",
    "INFO": "spectrochempy.application.application",
    "WARNING": "spectrochempy.application.application",
    "ERROR": "spectrochempy.application.application",
    "CRITICAL": "spectrochempy.application.application",
    "error_": "spectrochempy.application.application",
    "info_": "spectrochempy.application.application",
    "debug_": "spectrochempy.application.application",
    "warning_": "spectrochempy.application.application",
    "NO_DISPLAY": "spectrochempy.application.application",
    "get_loglevel": "spectrochempy.application.application",
    "set_loglevel": "spectrochempy.application.application",
    "get_config_dir": "spectrochempy.application.application",
    "check_update": "spectrochempy.application.check_update",
    "name": "spectrochempy.application.info",
    "icon": "spectrochempy.application.info",
    "description": "spectrochempy.application.info",
    "version": "spectrochempy.application.info",
    "__version__": "spectrochempy.application.info",
    "release": "spectrochempy.application.info",
    "release_date": "spectrochempy.application.info",
    "copyright": "spectrochempy.application.info",
    "url": "spectrochempy.application.info",
    "authors": "spectrochempy.application.info",
    "contributors": "spectrochempy.application.info",
    "license": "spectrochempy.application.info",
    "cite": "spectrochempy.application.info",
    "long_description": "spectrochempy.application.info",
    "preferences": "spectrochempy.application.preferences",
    "load": "spectrochempy.core.dataset.arraymixins.ndio",
    "abs": "spectrochempy.core.dataset.arraymixins.ndmath",
    "absolute": "spectrochempy.core.dataset.arraymixins.ndmath",
    "all": "spectrochempy.core.dataset.arraymixins.ndmath",
    "amax": "spectrochempy.core.dataset.arraymixins.ndmath",
    "amin": "spectrochempy.core.dataset.arraymixins.ndmath",
    "any": "spectrochempy.core.dataset.arraymixins.ndmath",
    "arange": "spectrochempy.core.dataset.arraymixins.ndmath",
    "argmax": "spectrochempy.core.dataset.arraymixins.ndmath",
    "argmin": "spectrochempy.core.dataset.arraymixins.ndmath",
    "around": "spectrochempy.core.dataset.arraymixins.ndmath",
    "average": "spectrochempy.core.dataset.arraymixins.ndmath",
    "clip": "spectrochempy.core.dataset.arraymixins.ndmath",
    "conj": "spectrochempy.core.dataset.arraymixins.ndmath",
    "conjugate": "spectrochempy.core.dataset.arraymixins.ndmath",
    "coordmax": "spectrochempy.core.dataset.arraymixins.ndmath",
    "coordmin": "spectrochempy.core.dataset.arraymixins.ndmath",
    "cumsum": "spectrochempy.core.dataset.arraymixins.ndmath",
    "diag": "spectrochempy.core.dataset.arraymixins.ndmath",
    "diagonal": "spectrochempy.core.dataset.arraymixins.ndmath",
    "empty": "spectrochempy.core.dataset.arraymixins.ndmath",
    "empty_like": "spectrochempy.core.dataset.arraymixins.ndmath",
    "eye": "spectrochempy.core.dataset.arraymixins.ndmath",
    "fromfunction": "spectrochempy.core.dataset.arraymixins.ndmath",
    "fromiter": "spectrochempy.core.dataset.arraymixins.ndmath",
    "full": "spectrochempy.core.dataset.arraymixins.ndmath",
    "full_like": "spectrochempy.core.dataset.arraymixins.ndmath",
    "geomspace": "spectrochempy.core.dataset.arraymixins.ndmath",
    "identity": "spectrochempy.core.dataset.arraymixins.ndmath",
    "linspace": "spectrochempy.core.dataset.arraymixins.ndmath",
    "logspace": "spectrochempy.core.dataset.arraymixins.ndmath",
    "max": "spectrochempy.core.dataset.arraymixins.ndmath",
    "mean": "spectrochempy.core.dataset.arraymixins.ndmath",
    "min": "spectrochempy.core.dataset.arraymixins.ndmath",
    "ones": "spectrochempy.core.dataset.arraymixins.ndmath",
    "ones_like": "spectrochempy.core.dataset.arraymixins.ndmath",
    "pipe": "spectrochempy.core.dataset.arraymixins.ndmath",
    "ptp": "spectrochempy.core.dataset.arraymixins.ndmath",
    "random": "spectrochempy.core.dataset.arraymixins.ndmath",
    "round": "spectrochempy.core.dataset.arraymixins.ndmath",
    "round_": "spectrochempy.core.dataset.arraymixins.ndmath",
    "std": "spectrochempy.core.dataset.arraymixins.ndmath",
    "sum": "spectrochempy.core.dataset.arraymixins.ndmath",
    "var": "spectrochempy.core.dataset.arraymixins.ndmath",
    "zeros": "spectrochempy.core.dataset.arraymixins.ndmath",
    "zeros_like": "spectrochempy.core.dataset.arraymixins.ndmath",
    "plot": "spectrochempy.core.dataset.arraymixins.ndplot",
    "Coord": "spectrochempy.core.dataset.coord",
    "CoordSet": "spectrochempy.core.dataset.coordset",
    "NDDataset": "spectrochempy.core.dataset.nddataset",
    "multiplot": "spectrochempy.core.plotters.multiplot",
    "multiplot_map": "spectrochempy.core.plotters.multiplot",
    "multiplot_stack": "spectrochempy.core.plotters.multiplot",
    "multiplot_image": "spectrochempy.core.plotters.multiplot",
    "multiplot_lines": "spectrochempy.core.plotters.multiplot",
    "multiplot_scatter": "spectrochempy.core.plotters.multiplot",
    "multiplot_with_transposed": "spectrochempy.core.plotters.multiplot",
    "plot_with_transposed": "spectrochempy.core.plotters.multiplot",
    "plot_1D": "spectrochempy.core.plotters.plot1d",
    "plot_pen": "spectrochempy.core.plotters.plot1d",
    "plot_scatter": "spectrochempy.core.plotters.plot1d",
    "plot_bar": "spectrochempy.core.plotters.plot1d",
    "plot_multiple": "spectrochempy.core.plotters.plot1d",
    "plot_scatter_pen": "spectrochempy.core.plotters.plot1d",
    "plot_2D": "spectrochempy.core.plotters.plot2d",
    "plot_map": "spectrochempy.core.plotters.plot2d",
    "plot_stack": "spectrochempy.core.plotters.plot2d",
    "plot_image": "spectrochempy.core.plotters.plot2d",
    "plot_3D": "spectrochempy.core.plotters.plot3d",
    "plot_surface": "spectrochempy.core.plotters.plot3d",
    "plot_waterfall": "spectrochempy.core.plotters.plot3d",
    "Project": "spectrochempy.core.project.project",
    "load_iris": "spectrochempy.core.readers.download",
    "download_nist_ir": "spectrochempy.core.readers.download",
    "read": "spectrochempy.core.readers.importer",
    "read_dir": "spectrochempy.core.readers.importer",
    "read_carroucell": "spectrochempy.core.readers.read_carroucell",
    "read_csv": "spectrochempy.core.readers.read_csv",
    "read_jcamp": "spectrochempy.core.readers.read_jcamp",
    "read_labspec": "spectrochempy.core.readers.read_labspec",
    "read_matlab": "spectrochempy.core.readers.read_matlab",
    "read_mat": "spectrochempy.core.readers.read_matlab",
    "read_omnic": "spectrochempy.core.readers.read_omnic",
    "read_spg": "spectrochempy.core.readers.read_omnic",
    "read_spa": "spectrochempy.core.readers.read_omnic",
    "read_srs": "spectrochempy.core.readers.read_omnic",
    "read_opus": "spectrochempy.core.readers.read_opus",
    "read_quadera": "spectrochempy.core.readers.read_quadera",
    "read_soc": "spectrochempy.core.readers.read_soc",
    "read_ddr": "spectrochempy.core.readers.read_soc",
    "read_sdr": "spectrochempy.core.readers.read_soc",
    "read_hdr": "spectrochempy.core.readers.read_soc",
    "read_spc": "spectrochempy.core.readers.read_spc",
    "read_topspin": "spectrochempy.core.readers.read_topspin",
    "read_wdf": "spectrochempy.core.readers.read_wire",
    "read_wire": "spectrochempy.core.readers.read_wire",
    "read_zip": "spectrochempy.core.readers.read_zip",
    "Script": "spectrochempy.core.script",
    "run_script": "spectrochempy.core.script",
    "run_all_scripts": "spectrochempy.core.script",
    "Unit": "spectrochempy.core.units",
    "Quantity": "spectrochempy.core.units",
    "ur": "spectrochempy.core.units",
    "set_nmr_context": "spectrochempy.core.units",
    "DimensionalityError": "spectrochempy.core.units",
    "write": "spectrochempy.core.writers.exporter",
    "write_csv": "spectrochempy.core.writers.write_csv",
    "write_excel": "spectrochempy.core.writers.write_excel",
    "write_xls": "spectrochempy.core.writers.write_excel",
    "write_jcamp": "spectrochempy.core.writers.write_jcamp",
    "write_matlab": "spectrochempy.core.writers.write_matlab",
    "write_mat": "spectrochempy.core.writers.write_matlab",
    "SpectroChemPyMagics": "spectrochempy.ipython.magics",
    "align": "spectrochempy.processing.alignement.align",
    "BaselineCorrection": "spectrochempy.processing.baselineprocessing.baseline_deprecated",
    "ab": "spectrochempy.processing.baselineprocessing.baseline_deprecated",
    "abc": "spectrochempy.processing.baselineprocessing.baseline_deprecated",
    "Baseline": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "get_baseline": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "basc": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "detrend": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "asls": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "snip": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "rubberband": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "lls": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "lls_inv": "spectrochempy.processing.baselineprocessing.baselineprocessing",
    "em": "spectrochempy.processing.fft.apodization",
    "gm": "spectrochempy.processing.fft.apodization",
    "sp": "spectrochempy.processing.fft.apodization",
    "sine": "spectrochempy.processing.fft.apodization",
    "sinm": "spectrochempy.processing.fft.apodization",
    "qsin": "spectrochempy.processing.fft.apodization",
    "general_hamming": "spectrochempy.processing.fft.apodization",
    "hamming": "spectrochempy.processing.fft.apodization",
    "hann": "spectrochempy.processing.fft.apodization",
    "triang": "spectrochempy.processing.fft.apodization",
    "bartlett": "spectrochempy.processing.fft.apodization",
    "blackmanharris": "spectrochempy.processing.fft.apodization",
    "fft": "spectrochempy.processing.fft.fft",
    "ifft": "spectrochempy.processing.fft.fft",
    "mc": "spectrochempy.processing.fft.fft",
    "ps": "spectrochempy.processing.fft.fft",
    "ht": "spectrochempy.processing.fft.fft",
    "pk": "spectrochempy.processing.fft.phasing",
    "pk_exp": "spectrochempy.processing.fft.phasing",
    "rs": "spectrochempy.processing.fft.shift",
    "ls": "spectrochempy.processing.fft.shift",
    "roll": "spectrochempy.processing.fft.shift",
    "cs": "spectrochempy.processing.fft.shift",
    "fsh": "spectrochempy.processing.fft.shift",
    "fsh2": "spectrochempy.processing.fft.shift",
    "dc": "spectrochempy.processing.fft.shift",
    "zf_auto": "spectrochempy.processing.fft.zero_filling",
    "zf_double": "spectrochempy.processing.fft.zero_filling",
    "zf_size": "spectrochempy.processing.fft.zero_filling",
    "zf": "spectrochempy.processing.fft.zero_filling",
    "denoise": "spectrochempy.processing.filter.denoise",
    "despike": "spectrochempy.processing.filter.denoise",
    "savgol_filter": "spectrochempy.processing.filter.filter",
    "savgol": "spectrochempy.processing.filter.filter",
    "smooth": "spectrochempy.processing.filter.filter",
    "whittaker": "spectrochempy.processing.filter.filter",
    "Filter": "spectrochempy.processing.filter.filter",
    "interpolate": "spectrochempy.processing.interpolation.interpolate",
    "autosub": "spectrochempy.processing.transformation.autosub",
    "concatenate": "spectrochempy.processing.transformation.concatenate",
    "stack": "spectrochempy.processing.transformation.concatenate",
    "dot": "spectrochempy.processing.transformation.npy",
    "NBlack": "spectrochempy.utils.colors",
    "NRed": "spectrochempy.utils.colors",
    "NBlue": "spectrochempy.utils.colors",
    "NGreen": "spectrochempy.utils.colors",
    "DEFAULT_DIM_NAME": "spectrochempy.utils.constants",
    "EPSILON": "spectrochempy.utils.constants",
    "INPLACE": "spectrochempy.utils.constants",
    "MASKED": "spectrochempy.utils.constants",
    "NOMASK": "spectrochempy.utils.constants",
    "MaskedArray": "spectrochempy.utils.constants",
    "MaskedConstant": "spectrochempy.utils.constants",
    "TYPE_BOOL": "spectrochempy.utils.constants",
    "TYPE_COMPLEX": "spectrochempy.utils.constants",
    "TYPE_FLOAT": "spectrochempy.utils.constants",
    "TYPE_INTEGER": "spectrochempy.utils.constants",
    "pathclean": "spectrochempy.utils.file",
    "Meta": "spectrochempy.utils.meta",
    "show": "spectrochempy.utils.mplutils",
    "Adict": "spectrochempy.utils.objects",
    "ReadOnlyDict": "spectrochempy.utils.objects",
    "ScpObjectList": "spectrochempy.utils.objects",
    "OrderedSet": "spectrochempy.utils.objects",
    "show_versions": "spectrochempy.utils.show_versions",
    "timeit": "spectrochempy.utils.timeutils",
}
