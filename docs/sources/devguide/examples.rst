.. _contributing.examples:

Adding your examples to the gallery
------------------------------------

If you have any nice examples to add to the SpectroChemPy gallery, they are more than welcome. They will allow other users to see the API in operation.

Examples should demonstrate the public SpectroChemPy API. Use dataset
arithmetic and methods where they express the calculation clearly. Use
``scp.Coord.linspace()`` for coordinates, ``scp.gaussian()`` for a Gaussian
profile, and ``scp.normal()`` for normal random samples. Keep NumPy for
interoperability, low-level comparisons, and operations without a suitable
public equivalent. For figures, use ``dataset.plot()``, ``scp.multiplot()``,
or ``scp.plot_multiple()``, including ``output=`` when saving a figure.
Keep Matplotlib for integrations or features outside the public plotting API.

In notebooks, assign an unused return value to ``_`` if IPython would display
it, for example ``_ = dataset.plot()``. Keep the figures and outputs that
explain the example.

There are two ways to do this:

* Use the discussion forums and attach your files, which we will take into account
  after revisions, mentioning of course the origin and the possible copyright.

* Create a pull request: For this operation, if you are a newcomer with the ``git`` version system,
  please refer to the :ref:`develguide`  section .
