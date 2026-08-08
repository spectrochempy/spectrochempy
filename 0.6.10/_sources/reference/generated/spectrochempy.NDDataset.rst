spectrochempy.NDDataset
=======================

.. currentmodule:: spectrochempy


.. autoclass:: NDDataset

   

   
   
   .. rubric:: Attributes Summary


   .. autosummary::
   
      ~NDDataset.II
      ~NDDataset.IR
      ~NDDataset.RI
      ~NDDataset.RR
      ~NDDataset.T
      ~NDDataset.acquisition_date
      ~NDDataset.author
      ~NDDataset.ax
      ~NDDataset.axT
      ~NDDataset.axec
      ~NDDataset.axecT
      ~NDDataset.axex
      ~NDDataset.axey
      ~NDDataset.comment
      ~NDDataset.coordnames
      ~NDDataset.coordset
      ~NDDataset.coordtitles
      ~NDDataset.coordunits
      ~NDDataset.created
      ~NDDataset.data
      ~NDDataset.description
      ~NDDataset.dimensionless
      ~NDDataset.dims
      ~NDDataset.directory
      ~NDDataset.divider
      ~NDDataset.dtype
      ~NDDataset.fig
      ~NDDataset.fignum
      ~NDDataset.filename
      ~NDDataset.filetype
      ~NDDataset.has_complex_dims
      ~NDDataset.has_data
      ~NDDataset.has_defined_name
      ~NDDataset.has_units
      ~NDDataset.history
      ~NDDataset.id
      ~NDDataset.imag
      ~NDDataset.is_1d
      ~NDDataset.is_complex
      ~NDDataset.is_empty
      ~NDDataset.is_float
      ~NDDataset.is_integer
      ~NDDataset.is_interleaved
      ~NDDataset.is_labeled
      ~NDDataset.is_masked
      ~NDDataset.is_quaternion
      ~NDDataset.labels
      ~NDDataset.limits
      ~NDDataset.local_timezone
      ~NDDataset.m
      ~NDDataset.magnitude
      ~NDDataset.mask
      ~NDDataset.masked_data
      ~NDDataset.meta
      ~NDDataset.modeldata
      ~NDDataset.modified
      ~NDDataset.name
      ~NDDataset.ndaxes
      ~NDDataset.ndim
      ~NDDataset.origin
      ~NDDataset.parent
      ~NDDataset.preferences
      ~NDDataset.real
      ~NDDataset.roi
      ~NDDataset.shape
      ~NDDataset.size
      ~NDDataset.suffix
      ~NDDataset.timezone
      ~NDDataset.title
      ~NDDataset.umasked_data
      ~NDDataset.unitless
      ~NDDataset.units
      ~NDDataset.value
      ~NDDataset.values
   
   

   
   
   .. rubric:: Methods Summary


   .. autosummary::
   
      ~NDDataset.abs
      ~NDDataset.absolute
      ~NDDataset.add_coordset
      ~NDDataset.align
      ~NDDataset.all
      ~NDDataset.amax
      ~NDDataset.amin
      ~NDDataset.any
      ~NDDataset.arange
      ~NDDataset.argmax
      ~NDDataset.argmin
      ~NDDataset.around
      ~NDDataset.asfortranarray
      ~NDDataset.asls
      ~NDDataset.astype
      ~NDDataset.atleast_2d
      ~NDDataset.autosub
      ~NDDataset.average
      ~NDDataset.bartlett
      ~NDDataset.basc
      ~NDDataset.blackmanharris
      ~NDDataset.clip
      ~NDDataset.close_figure
      ~NDDataset.component
      ~NDDataset.concatenate
      ~NDDataset.conj
      ~NDDataset.conjugate
      ~NDDataset.coord
      ~NDDataset.coordmax
      ~NDDataset.coordmin
      ~NDDataset.copy
      ~NDDataset.cs
      ~NDDataset.cumsum
      ~NDDataset.dc
      ~NDDataset.delete_coordset
      ~NDDataset.denoise
      ~NDDataset.despike
      ~NDDataset.detrend
      ~NDDataset.diag
      ~NDDataset.diagonal
      ~NDDataset.dot
      ~NDDataset.download_nist_ir
      ~NDDataset.dump
      ~NDDataset.em
      ~NDDataset.empty
      ~NDDataset.empty_like
      ~NDDataset.eye
      ~NDDataset.fft
      ~NDDataset.find_peaks
      ~NDDataset.fromfunction
      ~NDDataset.fromiter
      ~NDDataset.fsh
      ~NDDataset.fsh2
      ~NDDataset.full
      ~NDDataset.full_like
      ~NDDataset.general_hamming
      ~NDDataset.geomspace
      ~NDDataset.get_axis
      ~NDDataset.get_baseline
      ~NDDataset.get_labels
      ~NDDataset.gm
      ~NDDataset.hamming
      ~NDDataset.hann
      ~NDDataset.ht
      ~NDDataset.identity
      ~NDDataset.ifft
      ~NDDataset.is_units_compatible
      ~NDDataset.ito
      ~NDDataset.ito_base_units
      ~NDDataset.ito_reduced_units
      ~NDDataset.linspace
      ~NDDataset.load
      ~NDDataset.load_iris
      ~NDDataset.logspace
      ~NDDataset.ls
      ~NDDataset.max
      ~NDDataset.mc
      ~NDDataset.mean
      ~NDDataset.min
      ~NDDataset.ones
      ~NDDataset.ones_like
      ~NDDataset.pipe
      ~NDDataset.pk
      ~NDDataset.pk_exp
      ~NDDataset.plot
      ~NDDataset.plot_1D
      ~NDDataset.plot_2D
      ~NDDataset.plot_3D
      ~NDDataset.plot_bar
      ~NDDataset.plot_image
      ~NDDataset.plot_map
      ~NDDataset.plot_multiple
      ~NDDataset.plot_pen
      ~NDDataset.plot_scatter
      ~NDDataset.plot_scatter_pen
      ~NDDataset.plot_stack
      ~NDDataset.plot_surface
      ~NDDataset.plot_waterfall
      ~NDDataset.ps
      ~NDDataset.ptp
      ~NDDataset.qsin
      ~NDDataset.random
      ~NDDataset.read
      ~NDDataset.read_carroucell
      ~NDDataset.read_csv
      ~NDDataset.read_ddr
      ~NDDataset.read_dir
      ~NDDataset.read_hdr
      ~NDDataset.read_jcamp
      ~NDDataset.read_labspec
      ~NDDataset.read_mat
      ~NDDataset.read_matlab
      ~NDDataset.read_omnic
      ~NDDataset.read_opus
      ~NDDataset.read_quadera
      ~NDDataset.read_sdr
      ~NDDataset.read_soc
      ~NDDataset.read_spa
      ~NDDataset.read_spc
      ~NDDataset.read_spg
      ~NDDataset.read_srs
      ~NDDataset.read_topspin
      ~NDDataset.read_wdf
      ~NDDataset.read_wire
      ~NDDataset.read_zip
      ~NDDataset.remove_masks
      ~NDDataset.roll
      ~NDDataset.round
      ~NDDataset.round_
      ~NDDataset.rs
      ~NDDataset.rubberband
      ~NDDataset.save
      ~NDDataset.save_as
      ~NDDataset.savgol
      ~NDDataset.savgol_filter
      ~NDDataset.set_complex
      ~NDDataset.set_coordset
      ~NDDataset.set_coordtitles
      ~NDDataset.set_coordunits
      ~NDDataset.set_hypercomplex
      ~NDDataset.set_quaternion
      ~NDDataset.simps
      ~NDDataset.simpson
      ~NDDataset.sine
      ~NDDataset.sinm
      ~NDDataset.smooth
      ~NDDataset.snip
      ~NDDataset.sort
      ~NDDataset.sp
      ~NDDataset.squeeze
      ~NDDataset.stack
      ~NDDataset.std
      ~NDDataset.sum
      ~NDDataset.swapaxes
      ~NDDataset.swapdims
      ~NDDataset.take
      ~NDDataset.to
      ~NDDataset.to_array
      ~NDDataset.to_base_units
      ~NDDataset.to_reduced_units
      ~NDDataset.to_xarray
      ~NDDataset.transpose
      ~NDDataset.trapezoid
      ~NDDataset.trapz
      ~NDDataset.triang
      ~NDDataset.var
      ~NDDataset.whittaker
      ~NDDataset.write
      ~NDDataset.write_csv
      ~NDDataset.write_excel
      ~NDDataset.write_jcamp
      ~NDDataset.write_mat
      ~NDDataset.write_matlab
      ~NDDataset.write_xls
      ~NDDataset.zeros
      ~NDDataset.zeros_like
      ~NDDataset.zf
      ~NDDataset.zf_auto
      ~NDDataset.zf_double
      ~NDDataset.zf_size

   
   

   
   
   .. rubric:: Attributes Documentation

   
   .. autoattribute:: II
   .. autoattribute:: IR
   .. autoattribute:: RI
   .. autoattribute:: RR
   .. autoattribute:: T
   .. autoattribute:: acquisition_date
   .. autoattribute:: author
   .. autoattribute:: ax
   .. autoattribute:: axT
   .. autoattribute:: axec
   .. autoattribute:: axecT
   .. autoattribute:: axex
   .. autoattribute:: axey
   .. autoattribute:: comment
   .. autoattribute:: coordnames
   .. autoattribute:: coordset
   .. autoattribute:: coordtitles
   .. autoattribute:: coordunits
   .. autoattribute:: created
   .. autoattribute:: data
   .. autoattribute:: description
   .. autoattribute:: dimensionless
   .. autoattribute:: dims
   .. autoattribute:: directory
   .. autoattribute:: divider
   .. autoattribute:: dtype
   .. autoattribute:: fig
   .. autoattribute:: fignum
   .. autoattribute:: filename
   .. autoattribute:: filetype
   .. autoattribute:: has_complex_dims
   .. autoattribute:: has_data
   .. autoattribute:: has_defined_name
   .. autoattribute:: has_units
   .. autoattribute:: history
   .. autoattribute:: id
   .. autoattribute:: imag
   .. autoattribute:: is_1d
   .. autoattribute:: is_complex
   .. autoattribute:: is_empty
   .. autoattribute:: is_float
   .. autoattribute:: is_integer
   .. autoattribute:: is_interleaved
   .. autoattribute:: is_labeled
   .. autoattribute:: is_masked
   .. autoattribute:: is_quaternion
   .. autoattribute:: labels
   .. autoattribute:: limits
   .. autoattribute:: local_timezone
   .. autoattribute:: m
   .. autoattribute:: magnitude
   .. autoattribute:: mask
   .. autoattribute:: masked_data
   .. autoattribute:: meta
   .. autoattribute:: modeldata
   .. autoattribute:: modified
   .. autoattribute:: name
   .. autoattribute:: ndaxes
   .. autoattribute:: ndim
   .. autoattribute:: origin
   .. autoattribute:: parent
   .. autoattribute:: preferences
   .. autoattribute:: real
   .. autoattribute:: roi
   .. autoattribute:: shape
   .. autoattribute:: size
   .. autoattribute:: suffix
   .. autoattribute:: timezone
   .. autoattribute:: title
   .. autoattribute:: umasked_data
   .. autoattribute:: unitless
   .. autoattribute:: units
   .. autoattribute:: value
   .. autoattribute:: values
   
   

   
   
   .. rubric:: Methods Documentation

   
   .. automethod:: abs
   .. automethod:: absolute
   .. automethod:: add_coordset
   .. automethod:: align
   .. automethod:: all
   .. automethod:: amax
   .. automethod:: amin
   .. automethod:: any
   .. automethod:: arange
   .. automethod:: argmax
   .. automethod:: argmin
   .. automethod:: around
   .. automethod:: asfortranarray
   .. automethod:: asls
   .. automethod:: astype
   .. automethod:: atleast_2d
   .. automethod:: autosub
   .. automethod:: average
   .. automethod:: bartlett
   .. automethod:: basc
   .. automethod:: blackmanharris
   .. automethod:: clip
   .. automethod:: close_figure
   .. automethod:: component
   .. automethod:: concatenate
   .. automethod:: conj
   .. automethod:: conjugate
   .. automethod:: coord
   .. automethod:: coordmax
   .. automethod:: coordmin
   .. automethod:: copy
   .. automethod:: cs
   .. automethod:: cumsum
   .. automethod:: dc
   .. automethod:: delete_coordset
   .. automethod:: denoise
   .. automethod:: despike
   .. automethod:: detrend
   .. automethod:: diag
   .. automethod:: diagonal
   .. automethod:: dot
   .. automethod:: download_nist_ir
   .. automethod:: dump
   .. automethod:: em
   .. automethod:: empty
   .. automethod:: empty_like
   .. automethod:: eye
   .. automethod:: fft
   .. automethod:: find_peaks
   .. automethod:: fromfunction
   .. automethod:: fromiter
   .. automethod:: fsh
   .. automethod:: fsh2
   .. automethod:: full
   .. automethod:: full_like
   .. automethod:: general_hamming
   .. automethod:: geomspace
   .. automethod:: get_axis
   .. automethod:: get_baseline
   .. automethod:: get_labels
   .. automethod:: gm
   .. automethod:: hamming
   .. automethod:: hann
   .. automethod:: ht
   .. automethod:: identity
   .. automethod:: ifft
   .. automethod:: is_units_compatible
   .. automethod:: ito
   .. automethod:: ito_base_units
   .. automethod:: ito_reduced_units
   .. automethod:: linspace
   .. automethod:: load
   .. automethod:: load_iris
   .. automethod:: logspace
   .. automethod:: ls
   .. automethod:: max
   .. automethod:: mc
   .. automethod:: mean
   .. automethod:: min
   .. automethod:: ones
   .. automethod:: ones_like
   .. automethod:: pipe
   .. automethod:: pk
   .. automethod:: pk_exp
   .. automethod:: plot
   .. automethod:: plot_1D
   .. automethod:: plot_2D
   .. automethod:: plot_3D
   .. automethod:: plot_bar
   .. automethod:: plot_image
   .. automethod:: plot_map
   .. automethod:: plot_multiple
   .. automethod:: plot_pen
   .. automethod:: plot_scatter
   .. automethod:: plot_scatter_pen
   .. automethod:: plot_stack
   .. automethod:: plot_surface
   .. automethod:: plot_waterfall
   .. automethod:: ps
   .. automethod:: ptp
   .. automethod:: qsin
   .. automethod:: random
   .. automethod:: read
   .. automethod:: read_carroucell
   .. automethod:: read_csv
   .. automethod:: read_ddr
   .. automethod:: read_dir
   .. automethod:: read_hdr
   .. automethod:: read_jcamp
   .. automethod:: read_labspec
   .. automethod:: read_mat
   .. automethod:: read_matlab
   .. automethod:: read_omnic
   .. automethod:: read_opus
   .. automethod:: read_quadera
   .. automethod:: read_sdr
   .. automethod:: read_soc
   .. automethod:: read_spa
   .. automethod:: read_spc
   .. automethod:: read_spg
   .. automethod:: read_srs
   .. automethod:: read_topspin
   .. automethod:: read_wdf
   .. automethod:: read_wire
   .. automethod:: read_zip
   .. automethod:: remove_masks
   .. automethod:: roll
   .. automethod:: round
   .. automethod:: round_
   .. automethod:: rs
   .. automethod:: rubberband
   .. automethod:: save
   .. automethod:: save_as
   .. automethod:: savgol
   .. automethod:: savgol_filter
   .. automethod:: set_complex
   .. automethod:: set_coordset
   .. automethod:: set_coordtitles
   .. automethod:: set_coordunits
   .. automethod:: set_hypercomplex
   .. automethod:: set_quaternion
   .. automethod:: simps
   .. automethod:: simpson
   .. automethod:: sine
   .. automethod:: sinm
   .. automethod:: smooth
   .. automethod:: snip
   .. automethod:: sort
   .. automethod:: sp
   .. automethod:: squeeze
   .. automethod:: stack
   .. automethod:: std
   .. automethod:: sum
   .. automethod:: swapaxes
   .. automethod:: swapdims
   .. automethod:: take
   .. automethod:: to
   .. automethod:: to_array
   .. automethod:: to_base_units
   .. automethod:: to_reduced_units
   .. automethod:: to_xarray
   .. automethod:: transpose
   .. automethod:: trapezoid
   .. automethod:: trapz
   .. automethod:: triang
   .. automethod:: var
   .. automethod:: whittaker
   .. automethod:: write
   .. automethod:: write_csv
   .. automethod:: write_excel
   .. automethod:: write_jcamp
   .. automethod:: write_mat
   .. automethod:: write_matlab
   .. automethod:: write_xls
   .. automethod:: zeros
   .. automethod:: zeros_like
   .. automethod:: zf
   .. automethod:: zf_auto
   .. automethod:: zf_double
   .. automethod:: zf_size
   
   


.. rubric:: Examples using spectrochempy.NDDataset
.. _sphx_glr_backref_spectrochempy.NDDataset:

.. minigallery:: spectrochempy.NDDataset