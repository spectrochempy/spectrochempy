.. _api_reference_spectrochempy:

User API reference
==================

.. currentmodule:: spectrochempy

The |scp| API exposes many objects and functions that are described below.

To use the API, one must load it using one of the following syntax:

>>> from spectrochempy import api

>>> from spectrochempy.api import *

In the second syntax, as usual in python, access to the objects/functions 
may be simplified (*e.g.*, we can use `plot_stack` instead of 
`api.plot_stack` but there is always a risk of overwriting some variables 
already in the namespace. Therefore, the first syntax is in general 
recommended,
although that, in the examples in this documentation, we have often use the 
second one for simplicity.


Constants
---------

:CRITICAL: 50
:DEBUG: 10
:ERROR: 40
:INFO: 20
:INPLACE: INPLACE
:NBlack: (0, 0, 0)
:NBlue: (0.0, 0.7, 1.0)
:NGreen: (0.13500000000000004, 0.9, 0.36000000000000004)
:NRed: (1.0, 0.22999999999999998, 0.0)
:WARNING: 30
:authors: C. Fernandez & A. Travert @LCS
:contributors: 
:copyright: 2014-2017 - A.Travert and C.Fernandez @ LCS
:license: CeCILL-B license
:log_level: 30
:release: 0.1a7.dev35+g53fe9771
:release_date: 2017-12-18
:scpdata: ~/spectrochempy/scp_data/testdata
:url: http://www-lcs.ensicaen.fr/spectrochempy
:version: 0.1a7.dev38+gc40b4fd6.d20171225




Objects
-------

.. autosummary::
   :toctree:

    spectrochempy.api.BaselineCorrection
    spectrochempy.api.Coord
    spectrochempy.api.CoordRange
    spectrochempy.api.CoordSet
    spectrochempy.api.Fit
    spectrochempy.api.FitParameters
    spectrochempy.api.Isotopes
    spectrochempy.api.NDArray
    spectrochempy.api.NDDataset
    spectrochempy.api.NDIO
    spectrochempy.api.NDPlot
    spectrochempy.api.PCA
    spectrochempy.api.ParameterScript
    spectrochempy.api.Project
    spectrochempy.api.ProjectPreferences
    spectrochempy.api.SVD
    spectrochempy.api.Script



Functions
---------

.. currentmodule:: spectrochempy

.. autosummary::
   :toctree:

    spectrochempy.api.Lsqnonneg
    spectrochempy.api.Lstsq
    spectrochempy.api.McrAls
    spectrochempy.api._set_figure_style
    spectrochempy.api.abs
    spectrochempy.api.align
    spectrochempy.api.apodize
    spectrochempy.api.autosub
    spectrochempy.api.available_styles
    spectrochempy.api.clear_output
    spectrochempy.api.concatenate
    spectrochempy.api.conjugate
    spectrochempy.api.diag
    spectrochempy.api.display_html
    spectrochempy.api.display_javascript
    spectrochempy.api.display_jpeg
    spectrochempy.api.display_json
    spectrochempy.api.display_latex
    spectrochempy.api.display_markdown
    spectrochempy.api.display_pdf
    spectrochempy.api.display_png
    spectrochempy.api.display_pretty
    spectrochempy.api.display_svg
    spectrochempy.api.dot
    spectrochempy.api.em
    spectrochempy.api.figure
    spectrochempy.api.gm
    spectrochempy.api.interpolate
    spectrochempy.api.load
    spectrochempy.api.multiplot
    spectrochempy.api.multiplot_image
    spectrochempy.api.multiplot_lines
    spectrochempy.api.multiplot_map
    spectrochempy.api.multiplot_scatter
    spectrochempy.api.multiplot_stack
    spectrochempy.api.multiplot_with_transposed
    spectrochempy.api.optimize
    spectrochempy.api.plot
    spectrochempy.api.plot_3D
    spectrochempy.api.plot_bar
    spectrochempy.api.plot_image
    spectrochempy.api.plot_lines
    spectrochempy.api.plot_map
    spectrochempy.api.plot_multiple
    spectrochempy.api.plot_pen
    spectrochempy.api.plot_scatter
    spectrochempy.api.plot_stack
    spectrochempy.api.plot_with_transposed
    spectrochempy.api.publish_display_data
    spectrochempy.api.raises
    spectrochempy.api.read
    spectrochempy.api.read_bruker_nmr
    spectrochempy.api.read_csv
    spectrochempy.api.read_dso
    spectrochempy.api.read_jdx
    spectrochempy.api.read_omnic
    spectrochempy.api.read_spa
    spectrochempy.api.read_spg
    spectrochempy.api.read_zip
    spectrochempy.api.run_all_scripts
    spectrochempy.api.run_script
    spectrochempy.api.set_complex
    spectrochempy.api.set_matplotlib_close
    spectrochempy.api.set_matplotlib_formats
    spectrochempy.api.set_nmr_context
    spectrochempy.api.show
    spectrochempy.api.sort
    spectrochempy.api.stack
    spectrochempy.api.swapaxes
    spectrochempy.api.transpose
    spectrochempy.api.update_display
    spectrochempy.api.upload_IRIS
    spectrochempy.api.write



Preferences
-----------

.. toctree::
    :maxdepth: 2

    preferences
    
    
