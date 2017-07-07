.. _units:

.. ipython::

    @suppress
    In [1]: from spectrochempy.api import *
      ....: Logger.log_level('Warning')


Units
#####

|scp| can do calculations with units

Create quantities
-----------------
* to create quantity, use for instance, one of the following expression:

.. ipython::

    In [1]: quantity('10.0 cm')

.. ipython::

    In [1]: quantity(10.0, 'cm')

.. ipython::

    In [1]: quantity(10.0, ur.cm)

or may be (?) simpler,

.. ipython::

    In [1]: 10.0 * ur.cm

`ur` stands for **unit registry**, which handle many type of units
(and conversion between them)

Do arithmetics with units
-------------------------

.. ipython::

    In [1]: a = 900 * ur.km

    In [2]: b = 4.5 * ur.hours

    In [3]: a/b

Such calculations can also be done using the following syntax

.. ipython::

    # using a string expression
    In [1]: quantity("900 km / (8 hours)")


Convert between units
---------------------

.. ipython::

    In [1]: c = a/b

    In [2]: c.to('m/s')

We can make the conversion *inplace* using *ito* instead of *to*

.. ipython::

    In [2]: c.ito('m/s')

    In [2]: c

Do math operations with consistent units
----------------------------------------

.. ipython::

    In [1]: x = 10 * ur.radians

    In [2]: np.sin(x)

Consistency of the units are checked!

.. ipython::

    In [1]: x = 10 * ur.meters

    In [2]: np.sqrt(x)

but this is wrong...

.. ipython::
    :verbatim:

    In [1]: x = 10 * ur.meters

    In [2]: np.cos(x)
    ERROR - DimensionalityError: Cannot convert from 'meter' to 'radian'










