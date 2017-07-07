.. _uncertainty:

.. ipython::

    @suppress
    In [1]: from spectrochempy.api import *

Uncertainties
#############

|scp| can do calculations with uncertainties (and units).

A quantity, with an `uncertainty` is called a **measurement**\ .

Use one of the following expression to create such `measurement`:

.. ipython::

    In [1]: measurement('10.0 cm')

.. ipython::

    In [1]: quantity(10.0, 'cm')

