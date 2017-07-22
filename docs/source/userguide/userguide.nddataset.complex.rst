.. _nddataset.complex:

Complex or hypercomplex NDDatasets
**********************************

|NDDataset| objects with complex data are handled differently than in
:class:`numpy.ndarray`.

Instead, complex data are stored by interlacing the real and imaginary part.
This allows the definition of data that can be complex in several axis, and *e
.g.,* allows 2D-hypercomplex array that can be transposed (useful for NMR data).

>>> from spectrochempy import *
>>> da = NDDataset([  [1.+2.j, 2.+0j], [1.3+2.j, 2.+0.5j],
...                   [1.+4.2j, 2.+3j], [5.+4.2j, 2.+3j ] ])
>>> print(da)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-19 12:53:18.589037
last modified: 2016-02-19 12:53:18.589088
  description:
   data title:
   data shape: 4 x 2(complex)
   data units: unitless
  data values:
         [[       1        2        2        0]
          [     1.3        2        2      0.5]
          [       1      4.2        2        3]
          [       5      4.2        2        3]]
--------------------------------------------------------------------------------

if the dataset is also complex in the first dimension (columns) then we
should have (note the shape description!):

>>> print(da)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-19 12:56:17.785833
last modified: 2016-02-19 12:56:17.786108
  description:
   data title:
   data shape: 2(complex) x 2(complex)
   data units: unitless
  data values:
         [[       1        2        2        0]
          [     1.3        2        2      0.5]
          [       1      4.2        2        3]
          [       5      4.2        2        3]]
--------------------------------------------------------------------------------

Complex NDDataset manipulation routine:


.. toctree::
