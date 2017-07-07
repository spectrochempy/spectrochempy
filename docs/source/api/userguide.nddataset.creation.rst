.. _nddataset.creation:

Create NDDataset objects
************************

There are many ways to create |NDDataset| objects.


1. Create from scratch
======================

Let's create a complete dataset from scratch. For this we need to define at
least an array (actually a simple list is enough, but it is generally more
convenient to create `~numpy.ndarray`).

>>> from spectrochempy import *
>>> dx = [0, 1, 2, 3]   # a simple list
>>> da = NDDataset(dx)  # the dataset is created
>>> print(da)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 20:44:11.392305
last modified: 2016-02-18 20:44:11.392971
  description:
   data title:
    data size: 4
   data units: unitless
  data values:
         [       0        1        2        3]
--------------------------------------------------------------------------------

We can add some information to this dataset:

>>> da.title = 'intensity'
>>> da.description = 'Some experimental measurements'
>>> da.units = 'dimensionless'
>>> print(da)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 20:44:11.392305
last modified: 2016-02-18 20:47:22.704622
  description: Some experimental measurements
   data title: intensity
    data size: 4
   data units: dimensionless
  data values:
         [       0        1        2        3]
--------------------------------------------------------------------------------

We can also put all needed information by passing the parameters at the
|NDDataset| instance creation. Here is a complete example of what can be done
to create such dataset.

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
>>> db = NDDataset(dx,
...                   axes = [axe0, axe1, axe2],
...                   title='absorbance',
...                   units='dimensionless'
...                   )
>>> print(db)
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

.. seealso::
       Any numpy creation function can be used to set up the initial dataset
       array.
       `numpy array creation routines <https://docs.scipy.org/doc/numpy/reference/routines.array-creation.html#routines-array-creation>`_

2. Copying existing NDDataset
=============================
To copy an existing dataset, this is as simple as:

>>> da_copy = da.copy()

or alternatively:

>>> da_copy = da[:]

Finally, it is also possible to initialize a dataset, using an existing one:


>>> dc = NDDataset(da, units='mol/g', title='concentration')
>>> print(dc)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 21:12:24.967702
last modified: 2016-02-18 21:12:24.970881
  description:
   data title: concentration
    data size: 4
   data units: mol / g
  data values:
         [       0        1        2        3]
--------------------------------------------------------------------------------

3. Importing from external dataset
==================================
Depending on the available importers, one can import data from external file.

see ... to do

.. toctree::
