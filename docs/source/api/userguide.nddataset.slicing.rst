.. _nddataset.slicing:

NDDataset object slicing
************************

|NDDataset| objects can be sliced in different ways, *e.g.,*:

1. by index, using a slice such as [3], [0:10], [:, 3:4], [..., 5:10], ...

2. by values, using a slice such as [3000.0:3500.0], [..., 300.0], ...

3. by labels, using a slice such as ['monday':'friday'], ...


As an example, let's first create a complete dataset:

>>> from spectrochempy import *
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

1. Slicing by index
===================

We can make slicing by index of this dataset very easily, in the same way as
we are can perform slicing of `~numpy.ndarray`.

>>> da_1 = da[0,1,2]
>>> print(da_1)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 17:48:21.295804
last modified: 2016-02-18 17:48:21.295804
  description:
   data title: absorbance
   data shape: 1 x 1 x 1
   data units: dimensionless
  data values:
         [[[ 0.7525]]]
       axis 0:
               title: Wavelength
         coordinates: 4000.0
               units: 1 / cm
              labels: a
       axis 1:
               title: Time-on-stream
         coordinates: 0.6060606060606061
               units: s
       axis 2:
               title: Temperature
         coordinates: 300.0
               units: K
              labels: hot
--------------------------------------------------------------------------------


2. Slicing by values
====================

Let's for example take the
data for the 4000  :math:`\mathrm{cm}^{-1}`  wavelength:

>>> da_4000 = da[4000.0]
>>> print(da_4000)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 15:17:25.950469
last modified: 2016-02-18 15:17:25.950469
  description:
   data title: absorbance
   data shape: 1 x 100 x 3
   data units: dimensionless
  data values:
         [[[ 0.8115  0.9189  0.7072]
           [ 0.203   0.2276  0.7525]
           ...,
           [ 0.9525  0.6958  0.4237]
           [ 0.1348  0.2573  0.5974]]]
       axis 0:
               title: Wavelength
         coordinates: 4000.0
               units: 1 / cm
              labels: a
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

.. note::

    Note the floating point value, *i.e.,* 4000.0, used as a slicing index. If
    one use an
    integer value (4000), then the slicing is made **by index not by value**,
    and in this particular case, an `IndexError` is issued as index 4000 does
    not exists (size along axis 0 is only 10, so that index vary between 0
    and 9!).

    >>> da_fail = da[4000]
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    ...
    IndexError: index 4000 is out of bounds for axis 0 with size 10

3. Slicing by labels
====================

The slicing can also be done by labels when the axes are labeled:

>>> da_hot = da[..., 'hot']
>>> print(da_hot)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 15:28:34.272201
last modified: 2016-02-18 15:28:34.272201
  description:
   data title: absorbance
   data shape: 10 x 100 x 1
   data units: dimensionless
  data values:
         [[[ 0.7072]
           [ 0.7525]
           ...,
           [ 0.4237]
           [ 0.5974]]
          [[ 0.4869]
           [ 0.7665]
           ...,
           [ 0.2694]
           [ 0.8294]]
          ...,
          [[ 0.5887]
           [ 0.9259]
           ...,
           [ 0.072 ]
           [ 0.4564]]
          [[ 0.5928]
           [ 0.7503]
           ...,
           [ 0.0364]
           [ 0.2001]]]
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
         coordinates: 300.0
               units: K
              labels: hot
--------------------------------------------------------------------------------

or using a slice range (and mixed slicing methods!) :

>>> da_range = da[4000.0, :,  'normal':'hot']
>>> print(da_range)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 15:36:25.223173
last modified: 2016-02-18 15:36:25.223173
  description:
   data title: absorbance
   data shape: 1 x 100 x 2
   data units: dimensionless
  data values:
         [[[ 0.9189  0.7072]
           [ 0.2276  0.7525]
           ...,
           [ 0.6958  0.4237]
           [ 0.2573  0.5974]]]
       axis 0:
               title: Wavelength
         coordinates: 4000.0
               units: 1 / cm
              labels: a
       axis 1:
               title: Time-on-stream
         coordinates: [  0.       0.6061 ...,  59.3939  60.    ]
               units: s
       axis 2:
               title: Temperature
         coordinates: [ 250.  300.]
               units: K
              labels: ['normal' 'hot']
--------------------------------------------------------------------------------

.. toctree::
