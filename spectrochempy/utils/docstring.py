"""Docstring module of the SpectroChemPy package

We use the docrep_ package for managing our docstrings

.. _docrep: http://docrep.readthedocs.io/en/latest/

(this module is copied from psyplot)

"""

from spectrochempy.extern.docrep import DocstringProcessor, dedents, safe_modulo

__all__ = ['docstrings','dedent','dedents','indent','append_original_doc']

def dedent(func):
    """
    Dedent the docstring of a function and substitute with :attr:`params`

    Parameters
    ----------
    func: function
        function with the documentation to dedent"""
    func.__doc__ = func.__doc__ and dedents(func.__doc__)
    return func


def indent(text, num=4):
    """Indent the given string"""
    str_indent = ' ' * num
    return str_indent + ('\n' + str_indent).join(text.splitlines())


def append_original_doc(parent, num=0):
    """Return an iterator that append the docstring of the given `parent`
    function to the applied function"""
    def func(func):
        func.__doc__ = func.__doc__ and func.__doc__ + indent(
            parent.__doc__, num)
        return func
    return func


_docstrings = DocstringProcessor()

_docstrings.get_sectionsf('DocstringProcessor.get_sections')(
        dedent(DocstringProcessor.get_sections))


class SpectroChemPyDocstringProcessor(DocstringProcessor):
    """
    A :class:`docrep.DocstringProcessor` subclass with possible types section
    """

    param_like_sections = DocstringProcessor.param_like_sections + [
        'Possible types']

    @_docstrings.dedent
    def get_sections(self, s, base, sections=[
            'Parameters', 'Other Parameters', 'Returns', 'Possible types']):
        """
        Extract the specified sections out of the given string

        The same as the :meth:`docrep.DocstringProcessor.get_sections` method
        but uses the ``'Possible types'`` section by default, too

        Parameters
        ----------
        %(DocstringProcessor.get_sections.parameters)s

        Returns
        -------
        str
            The replaced string
        """
        return super(SpectroChemPyDocstringProcessor, self).get_sections(
                                                            s, base, sections)

del _docstrings

#: :class:`docrep.SpectroChemPyDocstringProcessor` instance that simplifies
#: the reuse of docstrings from between different python objects.
docstrings = SpectroChemPyDocstringProcessor()


## Set some general parameters
docstrings.get_sections(docstrings.dedents(
"""
Note
----
To be completed with useful common parameters

Parameters
----------
axis : int, optional, default: -1
    Dimension index along which the method should be applied.
inplace : bool, optional, default= `False`
    Flag to say that the method return a new object (default)
    or not (inplace=True)

Other Parameters
----------------

Returns
-------
object 
    Same object or a copy depending on the `inplace` flag.

"""
), 'generic_method', sections=['Parameters', 'Returns'])

docstrings.keep_params('generic_method.parameters', 'axis')
docstrings.keep_params('generic_method.parameters', 'inplace')
docstrings.keep_params('generic_method.parameters', 'axis', 'inplace')
docstrings.keep_params('generic_method.returns', 'object')


if __name__ == '__main__':


    # modified exemple from the doc of docrep

    @docstrings.get_sectionsf('do_something')
    @docstrings.dedent
    def do_something(a, b, c, d):
        """
        Add tree numbers

        Parameters
        ----------
        a: int
            The first number
        b: int
            The second number
        c: int
            The third number
        d: int
            Another number

        Returns
        -------
        int
            `a` + `b` + `c` + `d`

        """
        return a + b + c + d

    docstrings.delete_params('do_something.parameters', 'c', 'd')
    @docstrings.dedent
    def do_more(*args, **kwargs):
        """
        Add two numbers and multiply it by 2

        Parameters
        ----------
        %(do_something.parameters.no_c|d)s

        Returns
        -------
        int
            (`a` + `b`) * 2

        """

        return do_something(*args[:1]) * 2

    print(do_more.__doc__)