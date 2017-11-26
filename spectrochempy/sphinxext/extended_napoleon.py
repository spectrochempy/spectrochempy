"""Sphinx extension module to provide additional sections for numpy docstrings

This extension extends the :mod:`sphinx.ext.napoleon` package with an
additional *Possible types* section in order to document possible types for
descriptors.

Notes
-----
If you use this module as a sphinx extension, you should not list the
:mod:`sphinx.ext.napoleon` module in the extensions variable of your conf.py.
This module has been tested for sphinx 1.3.1.

Copied from PsyPlot - See license in ....

"""

from abc import ABCMeta, abstractmethod
from sphinx.ext.napoleon import (
    NumpyDocstring, GoogleDocstring, setup as napoleon_setup)

__all__ = []

class DocstringExtension(object):
    """Class that introduces a "Possible Types" section

    This class serves as a base class for
    :class:`sphinx.ext.napoleon.NumpyDocstring` and
    :class:`sphinx.ext.napoleon.GoogleDocstring` to introduce
    another section names *Possible types*

    Examples
    --------
    The usage is the same as for the NumpyDocstring class, but it supports
    the `Possible types` section::

        >>> from sphinx.ext.napoleon import Config

        >>> from psyplot.sphinxext.extended_napoleon import (
        ...     ExtendedNumpyDocstring)
        >>> config = Config(napoleon_use_param=True,
        ...                 napoleon_use_rtype=True)
        >>> docstring = '''
        ... Possible types
        ... --------------
        ... type1
        ...     Description of `type1`
        ... type2
        ...     Description of `type2`'''
        >>> print(ExtendedNumpyDocstring(docstring, config))
        .. rubric:: Possible types

        * *type1* --
          Description of `type1`
        * *type2* --
          Description of `type2`"""
    __metaclass__ = ABCMeta

    def _parse_possible_types_section(self, section):
        fields = self._consume_fields(prefer_type=True)
        lines = ['.. rubric:: %s' % section, '']
        multi = len(fields) > 1
        for _name, _type, _desc in fields:
            field = self._format_field(_name, _type, _desc)
            if multi:
                lines.extend(self._format_block('* ', field))
            else:
                lines.extend(field)
        return lines + ['']

    @abstractmethod
    def _parse(self):
        pass


class ExtendedNumpyDocstring(NumpyDocstring, DocstringExtension):
    """:class:`sphinx.ext.napoleon.NumpyDocstring` with more sections"""

    def _parse(self, *args, **kwargs):
        self._sections['possible types'] = self._parse_possible_types_section
        return super(ExtendedNumpyDocstring, self)._parse(*args, **kwargs)


class ExtendedGoogleDocstring(GoogleDocstring, DocstringExtension):
    """:class:`sphinx.ext.napoleon.GoogleDocstring` with more sections"""
    def _parse(self, *args, **kwargs):
        self._sections['possible types'] = self._parse_possible_types_section
        return super(ExtendedGoogleDocstring, self)._parse(*args, **kwargs)


def process_docstring(app, what, name, obj, options, lines):
    """Process the docstring for a given python object.

    Called when autodoc has read and processed a docstring. `lines` is a list
    of docstring lines that `_process_docstring` modifies in place to change
    what Sphinx outputs.

    The following settings in conf.py control what styles of docstrings will
    be parsed:

    * ``napoleon_google_docstring`` -- parse Google style docstrings
    * ``napoleon_numpy_docstring`` -- parse NumPy style docstrings

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application object representing the Sphinx process.
    what : str
        A string specifying the type of the object to which the docstring
        belongs. Valid values: "module", "class", "exception", "function",
        "method", "attribute".
    name : str
        The fully qualified name of the object.
    obj : module, class, exception, function, method, or attribute
        The object to which the docstring belongs.
    options : sphinx.ext.autodoc.Options
        The options given to the directive: an object with attributes
        inherited_members, undoc_members, show_inheritance and noindex that
        are True if the flag option of same name was given to the auto
        directive.
    lines : list of str
        The lines of the docstring, see above.

        .. note:: `lines` is modified *in place*

    Notes
    -----
    This function is (to most parts) taken from the :mod:`sphinx.ext.napoleon`
    module, sphinx version 1.3.1, and adapted to the classes defined here"""
    result_lines = lines
    if app.config.napoleon_numpy_docstring:
        docstring = ExtendedNumpyDocstring(
            result_lines, app.config, app, what, name, obj, options)
        result_lines = docstring.lines()
    if app.config.napoleon_google_docstring:
        docstring = ExtendedGoogleDocstring(
            result_lines, app.config, app, what, name, obj, options)
        result_lines = docstring.lines()

    lines[:] = result_lines[:]


def setup(app):
    """Sphinx extension setup function

    When the extension is loaded, Sphinx imports this module and executes
    the ``setup()`` function, which in turn notifies Sphinx of everything
    the extension offers.

    Parameters
    ----------
    app : sphinx.application.Sphinx
        Application object representing the Sphinx process

    Notes
    -----
    This function uses the setup function of the :mod:`sphinx.ext.napoleon`
    module"""
    from sphinx.application import Sphinx
    if not isinstance(app, Sphinx):
        return  # probably called by tests

    app.connect('autodoc-process-docstring', process_docstring)
    return napoleon_setup(app)
