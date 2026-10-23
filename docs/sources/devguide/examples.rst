.. _contributing.examples:

Adding your examples to the gallery
------------------------------------

If you have any nice examples to add to the SpectroChemPy gallery, they are more than welcome. They will allow other users to see the API in operation.

Examples should demonstrate the public SpectroChemPy API. Prefer helpers such as
``scp.Coord.linspace``, ``scp.linspace`` or ``scp.normal`` over importing NumPy
solely to create arrays. Keep NumPy when the example specifically demonstrates
interoperability or when no suitable public equivalent exists. Likewise, prefer
``dataset.plot()``, ``scp.multiplot()`` or ``scp.plot_multiple()``, including
their ``output=`` lifecycle, over direct Matplotlib calls. Keep Matplotlib when
the example demonstrates integration or needs a feature the public plotting API
does not expose.

There are two ways to do this:

* Use the discussion forums and attach your files, which we will take into account
  after revisions, mentioning of course the origin and the possible copyright.

* Create a pull request: For this operation, if you are a newcomer with the ``git`` version system,
  please refer to the :ref:`develguide`  section .
