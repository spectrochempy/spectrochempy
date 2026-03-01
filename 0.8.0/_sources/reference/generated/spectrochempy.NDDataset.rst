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
      ~NDDataset.all
      ~NDDataset.amax
      ~NDDataset.amin
      ~NDDataset.any
      ~NDDataset.arange
      ~NDDataset.argmax
      ~NDDataset.argmin
      ~NDDataset.around
      ~NDDataset.asfortranarray
      ~NDDataset.astype
      ~NDDataset.atleast_2d
      ~NDDataset.average
      ~NDDataset.clip
      ~NDDataset.close_figure
      ~NDDataset.component
      ~NDDataset.conj
      ~NDDataset.conjugate
      ~NDDataset.coord
      ~NDDataset.coordmax
      ~NDDataset.coordmin
      ~NDDataset.copy
      ~NDDataset.cumsum
      ~NDDataset.delete_coordset
      ~NDDataset.diag
      ~NDDataset.diagonal
      ~NDDataset.dump
      ~NDDataset.empty
      ~NDDataset.empty_like
      ~NDDataset.eye
      ~NDDataset.fromfunction
      ~NDDataset.fromiter
      ~NDDataset.full
      ~NDDataset.full_like
      ~NDDataset.geomspace
      ~NDDataset.get_axis
      ~NDDataset.get_labels
      ~NDDataset.identity
      ~NDDataset.is_units_compatible
      ~NDDataset.ito
      ~NDDataset.ito_base_units
      ~NDDataset.ito_reduced_units
      ~NDDataset.linspace
      ~NDDataset.load
      ~NDDataset.loads
      ~NDDataset.logspace
      ~NDDataset.max
      ~NDDataset.mean
      ~NDDataset.min
      ~NDDataset.ones
      ~NDDataset.ones_like
      ~NDDataset.pipe
      ~NDDataset.plot
      ~NDDataset.ptp
      ~NDDataset.random
      ~NDDataset.remove_masks
      ~NDDataset.round
      ~NDDataset.round_
      ~NDDataset.save
      ~NDDataset.save_as
      ~NDDataset.set_complex
      ~NDDataset.set_coordset
      ~NDDataset.set_coordtitles
      ~NDDataset.set_coordunits
      ~NDDataset.set_hypercomplex
      ~NDDataset.set_quaternion
      ~NDDataset.sort
      ~NDDataset.squeeze
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
      ~NDDataset.var
      ~NDDataset.zeros
      ~NDDataset.zeros_like

   
   

   
   
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
   .. automethod:: all
   .. automethod:: amax
   .. automethod:: amin
   .. automethod:: any
   .. automethod:: arange
   .. automethod:: argmax
   .. automethod:: argmin
   .. automethod:: around
   .. automethod:: asfortranarray
   .. automethod:: astype
   .. automethod:: atleast_2d
   .. automethod:: average
   .. automethod:: clip
   .. automethod:: close_figure
   .. automethod:: component
   .. automethod:: conj
   .. automethod:: conjugate
   .. automethod:: coord
   .. automethod:: coordmax
   .. automethod:: coordmin
   .. automethod:: copy
   .. automethod:: cumsum
   .. automethod:: delete_coordset
   .. automethod:: diag
   .. automethod:: diagonal
   .. automethod:: dump
   .. automethod:: empty
   .. automethod:: empty_like
   .. automethod:: eye
   .. automethod:: fromfunction
   .. automethod:: fromiter
   .. automethod:: full
   .. automethod:: full_like
   .. automethod:: geomspace
   .. automethod:: get_axis
   .. automethod:: get_labels
   .. automethod:: identity
   .. automethod:: is_units_compatible
   .. automethod:: ito
   .. automethod:: ito_base_units
   .. automethod:: ito_reduced_units
   .. automethod:: linspace
   .. automethod:: load
   .. automethod:: loads
   .. automethod:: logspace
   .. automethod:: max
   .. automethod:: mean
   .. automethod:: min
   .. automethod:: ones
   .. automethod:: ones_like
   .. automethod:: pipe
   .. automethod:: plot
   .. automethod:: ptp
   .. automethod:: random
   .. automethod:: remove_masks
   .. automethod:: round
   .. automethod:: round_
   .. automethod:: save
   .. automethod:: save_as
   .. automethod:: set_complex
   .. automethod:: set_coordset
   .. automethod:: set_coordtitles
   .. automethod:: set_coordunits
   .. automethod:: set_hypercomplex
   .. automethod:: set_quaternion
   .. automethod:: sort
   .. automethod:: squeeze
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
   .. automethod:: var
   .. automethod:: zeros
   .. automethod:: zeros_like
   
   

.. rubric:: Examples using spectrochempy.NDDataset
.. _sphx_glr_backref_spectrochempy.NDDataset:

.. minigallery:: spectrochempy.NDDataset