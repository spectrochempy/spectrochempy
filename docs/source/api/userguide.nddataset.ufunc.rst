.. _ufunc:

Numpy universal functions (ufunc's)
===================================
A numpy universal function (or `~numpy.ufunc` for short) is a function that
operates on `~numpy.ndarray` in an element-by-element fashion. It's
vectorized and so rather fast.

As |scp| |NDDataset| imitate the behaviour of numpy objects, many numpy
ufuncs can be applied directly.

For example, if you need all the elements of a |NDDataset| to be change to the
squared rooted values, you can use the :func:`numpy.sqrt` function:

>>> from spectrochempy import *

>>> da = NDDataset([1., 2., 3.])
>>> da_sqrt = np.sqrt(ndd)
>>> print(da_sqrt)
--------------------------------------------------------------------------------
       author: christian@macbookcf.local
      created: 2016-02-18 18:36:48.937395
last modified: 2016-02-18 18:36:48.937395
  description:
   data title:
    data size: 3
   data units: unitless
  data values:
         [ 1.      1.4142  1.7321]
--------------------------------------------------------------------------------

.. note::
       it is worth noticing that importing |scp| also import
       the numpy namespace as `np`, *i.e.*,

       >>> import numpy as np

       is automatically performed, so there is no
       need to import it again.


Ufuns with NDDataset with units
-------------------------------
When |NDDataset| have units, some restrictions apply on the use of ufuncs:

Some function functions accept only dimensionless quantities. This is the
case of:

- Power and logarithmic functions: :func:`exp`, :func:`expm1`, :func:`log10`,
  :func:`log1p`, and :func:`log`


These functions accept angle as a quantity:

- :func:`cos`, :func:`sin`, and :func:`tan`

They return floats.

These functions accept floats, integers, and dimensionless quantities:

- :func:`acos`, :func:`asin`, :func:`atan`, and :func:`atan2`

They return angles as quantities.

:func:`atan2` accepts accept floats, integers, and quantities of the same
dimension.  It returns angle as a quantity.


These functions accept floats, integers, and dimensionless quantities:

- :func:`ceil` and :func:`floor`

If the input is a float or an integer, the output is a float.  If the input is
a :class:`~natu.core.Quantity` or a :class:`~natu.core.ScalarUnit`, the result
is the same and has the same :attr:`dimension`, :attr:`display`, etc.

These functions accept floats, integers, and quantities:

- :func:`fabs` and :func:`copysign`

If the input is a float or an integer, the output is a float.  If the input is a
:class:`~natu.core.Quantity` or a :class:`~natu.core.ScalarUnit`, the result is
of the same type and has the same :attr:`dimension`, :attr:`display`, etc. (of
the first argument in the case of :func:`copysign`).

These functions also accept floats, integers, and quantities:

- :func:`ldexp`, :func:`pow`, and :func:`sqrt`

If the input is an float or an integer, the output is a float.  If the input is
a :class:`~natu.core.Quantity` or a :class:`~natu.core.ScalarUnit`, the result
is the same.  The dimensions and display units are handled according to the
power.

:func:`fmod` accepts floats, integers, and quantities of the same dimension.
The output is always a float.

These functions also accept floats, integers, and quantities of the same
dimension:

- :func:`fsum` and :func:`hypot`

The display unit (and :attr:`prefixable` attribute, if applicable) of the first
argument or entry is propagated to the output.

These functions accept floats or quantities:

- :func:`isinf`, :func:`isfinite` (only available in Python >= 3.2), and
  :func:`isnan`

Only the value of a quantity is used; dimension and display unit are ignored.
