SpectroChemPy config preferences
================================




.. configtrait:: Application.log_datefmt

    The date format used by logging formatters for %(asctime)s

    :trait type: Unicode
    :default: ``'%Y-%m-%d %H:%M:%S'``

.. configtrait:: Application.log_format

    The Logging format template

    :trait type: Unicode
    :default: ``'[%(name)s]%(highlevel)s %(message)s'``

.. configtrait:: Application.log_level

    Set the log level by value or name.

    :options: ``0``, ``10``, ``20``, ``30``, ``40``, ``50``, ``'DEBUG'``, ``'INFO'``, ``'WARN'``, ``'ERROR'``, ``'CRITICAL'``
    :default: ``30``

.. configtrait:: SpectroChemPy.config_dir

    Set the configuration dir location

    :trait type: Unicode

.. configtrait:: SpectroChemPy.config_file_name

    The config filename

    :trait type: Unicode

.. configtrait:: SpectroChemPy.debug

    Set DEBUG mode, with full outputs

    :trait type: Bool
    :default: ``False``

.. configtrait:: SpectroChemPy.do_not_block

    Make the plots but do not stop (for tests)

    :trait type: Bool
    :default: ``False``

.. configtrait:: SpectroChemPy.log_datefmt

    The date format used by logging formatters for %(asctime)s

    :trait type: Unicode
    :default: ``'%Y-%m-%d %H:%M:%S'``

.. configtrait:: SpectroChemPy.log_format

    The Logging format template

    :trait type: Unicode
    :default: ``'[%(name)s]%(highlevel)s %(message)s'``

.. configtrait:: SpectroChemPy.log_level

    Set the log level by value or name.

    :options: ``0``, ``10``, ``20``, ``30``, ``40``, ``50``, ``'DEBUG'``, ``'INFO'``, ``'WARN'``, ``'ERROR'``, ``'CRITICAL'``
    :default: ``30``
    :CLI option: ``--log_level``

.. configtrait:: SpectroChemPy.quiet

    Set Quiet mode, with minimal outputs

    :trait type: Bool
    :default: ``False``

.. configtrait:: SpectroChemPy.reset_config

    Should we restaure a default configuration?

    :trait type: Bool
    :default: ``False``

.. configtrait:: SpectroChemPy.startup_project

    Project to load at startup

    :trait type: Unicode
    :CLI option: ``-p``

.. configtrait:: GeneralPreferences.csv_delimiter

    CSV data delimiter

    :trait type: Unicode
    :default: ``';'``

.. configtrait:: GeneralPreferences.data

    Default data directory

    :trait type: Unicode

.. configtrait:: GeneralPreferences.show_info_on_loading

    Display info on loading?

    :trait type: Bool
    :default: ``True``

.. configtrait:: ProjectPreferences.project_directory

    Location where projects are stored by default

    :trait type: Unicode

.. configtrait:: PlotterPreferences.background_color

    Bakground color for plots

    :trait type: Unicode
    :default: ``'#EFEFEF'``

.. configtrait:: PlotterPreferences.colormap

    Default colormap for contour plots

    :trait type: Unicode
    :default: ``'jet'``

.. configtrait:: PlotterPreferences.colormap_stack

    Default colormap for stack plots

    :trait type: Unicode
    :default: ``'viridis'``

.. configtrait:: PlotterPreferences.colormap_transposed

    Default colormap for transposed stack plots

    :trait type: Unicode
    :default: ``'magma'``

.. configtrait:: PlotterPreferences.contour_alpha

    Transparency of the contours

    :trait type: Float
    :default: ``1``

.. configtrait:: PlotterPreferences.contour_start

    Fraction of the maximum for starting contour levels

    :trait type: Float
    :default: ``0.05``

.. configtrait:: PlotterPreferences.foreground_color

    Foreground color for plots

    :trait type: Unicode
    :default: ``'#000'``

.. configtrait:: PlotterPreferences.latex_preamble

    Latex preamble for matplotlib outputs

    :trait type: Unicode
    :default: ``'\\usepackage{siunitx}\\n\\sisetup{detect-all}\\n\\usepackage{t...``

.. configtrait:: PlotterPreferences.linewidth

    Default width for lines

    :trait type: Float
    :default: ``0.7``

.. configtrait:: PlotterPreferences.max_lines_in_stack

    Maximum number of lines to plot in stack plots

    :trait type: Int
    :default: ``1000``

.. configtrait:: PlotterPreferences.method_2D

    Default plot methods for 2D

    :trait type: Unicode
    :default: ``'map'``

.. configtrait:: PlotterPreferences.number_of_contours

    Number of contours

    :trait type: Int
    :default: ``50``

.. configtrait:: PlotterPreferences.number_of_x_labels

    Number of X labels

    :trait type: Int
    :default: ``5``

.. configtrait:: PlotterPreferences.number_of_y_labels

    Number of Y labels

    :trait type: Int
    :default: ``5``

.. configtrait:: PlotterPreferences.number_of_z_labels

    Number of Z labels

    :trait type: Int
    :default: ``5``

.. configtrait:: PlotterPreferences.show_projection_x

    Show projection along x

    :trait type: Bool
    :default: ``False``

.. configtrait:: PlotterPreferences.show_projection_y

    Show projection along y

    :trait type: Bool
    :default: ``False``

.. configtrait:: PlotterPreferences.show_projections

    Show all projections

    :trait type: Bool
    :default: ``False``

.. configtrait:: PlotterPreferences.style

    Basic matplotlib style to use

    :trait type: Unicode
    :default: ``'lcs'``

.. configtrait:: PlotterPreferences.use_latex

    Should we use latex for plotting labels and texts?

    :trait type: Bool
    :default: ``True``

