.. _nddataset:

N-dimensional dataset objects
*****************************

The main N-dimensional dataset |NDDataset| class
used by |scp|.

Introduction
============

|NDDataset| objects mostly behave as numpy's `~numpy.ndarray`.

However, unlike raw numpy's ndarray, the presence of optional properties such
as `uncertainty`, `mask`, `units`, `axes`, and axes `labels` make them
(hopefully) more appropriate for handling spectroscopic information, one of
the major objectives of the |scp| package.

Additional metadata can also be added to the instances of this class through the
`meta` properties.


In the following example, a minimal 1D dataset is created from a simple list:

>>> from spectrochempy import * # doctest: +SKIP
Welcome to SpectroChemPy!
-matplotlib and numpy namespaces are available under the name 'plt' and 'np' respectively
-matplotlib currently uses the 'nbagg' backend. To change this, use e.g., plt.switch_backend('qt4agg') or %matplotlib inline
...

>>> da = NDDataset([1,2,3])
>>> print(da) # doctest: +SKIP
--------------------------------------------------------------------------------
        author: christian@macbookcf.local
      created: 2016-02-18 14:37:13.826452
last modified: 2016-02-18 14:37:13.826452
  description:
   data title:
    data size: 3
  data values:
         array([       1,        2,        3])
--------------------------------------------------------------------------------

Except few addtional metadata such `author`, `created` ..., there is not much
differences with respect to a conventional `~numpy.ndarray`. For example, one
can apply :ref:`ufunc` directly to a nddataset or make basic arithmetic
operation with these objects:

>>> da2 = np.sqrt(da**3)
>>> print(da2)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 14:37:13.841598
last modified: 2016-02-18 14:37:13.841598
  description:
   data title:
    data size: 3
  data values:
         array([       1,     2.83,      5.2])
--------------------------------------------------------------------------------

But |NDDataset| can be created with much more information, as abovementioned:

>>> dx = np.random.random((10, 100, 3))
>>> axe0 = Axis(coords = np.linspace(4000., 1000., 10),
...            labels = 'a b c d e f g h i j'.split(),
...            mask = None,
...            units = "cm^-1",
...            title = 'wavelength')
>>> axe1 = Axis(coords = np.linspace(0., 60., 100),
...            labels = None,
...            mask = None,
...            units = "s",
...            title = 'time-on-stream')
>>> axe2 = Axis(coords = np.linspace(200., 300., 3),
...            labels = ['cold', 'normal', 'hot'],
...            mask = None,
...            units = "K",
...            title = 'temperature')
>>> da = NDDataset(dx,
...                   axes = [axe0, axe1, axe2],
...                   title='absorbance',
...                   units='dimensionless'
...                   )
>>> print(da)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 14:55:31.296528
last modified: 2016-02-18 14:55:31.296528
  description:
   data title: absorbance
   data shape: 10 x 100 x 3
   data units: dimensionless
  data values:
         [[[ 0.1249  0.2485  0.2801]
           [ 0.3814  0.8511  0.0214]
           ...,
           [ 0.6599  0.8625  0.4346]
           [ 0.1213  0.1462  0.0579]]
          [[ 0.2121  0.4248  0.1578]
           [ 0.0575  0.4325  0.864 ]
           ...,
           [ 0.6506  0.482   0.4579]
           [ 0.7741  0.9069  0.5646]]
          ...,
          [[ 0.7935  0.7464  0.2619]
           [ 0.2052  0.408   0.2826]
           ...,
           [ 0.734   0.914   0.1439]
           [ 0.9391  0.6966  0.9407]]
          [[ 0.9616  0.6756  0.7476]
           [ 0.352   0.7905  0.8835]
           ...,
           [ 0.4984  0.4447  0.4597]
           [ 0.0376  0.6532  0.2819]]]
       axis 0:
               title: Wavelength
         coordinates: [ 4000.      3666.6667 ...,  1333.3333  1000.    ]
               units: 1 / cm
              labels: ['a' 'b' ..., 'i' 'j']
       axis 1:
               title: Time-on-stream
         coordinates: [  0.       0.6061 ...,  59.3939  60.    ]
               units: s
       axis 2:
               title: Temperature
         coordinates: [ 200.  250.  300.]
               units: K
              labels: ['cold' 'normal' 'hot']
--------------------------------------------------------------------------------


Using NDDataset
===============

.. toctree::
   :maxdepth: 3

   userguide.nddataset.creation
   userguide.nddataset.complex
   userguide.nddataset.slicing
   userguide.nddataset.ufunc

NMR NDDataset
=============



