"""Sphinx extension that defines a new automodule directive with autosummary

The :class:`AutoSummDirective` defined in this extension module allows the
same functionality as the automodule and autoclass directives of the
:mod:`sphinx.ext.autodoc` module but with an additional `autosummary` option.
This option puts a autosummary in the style of the
:mod:`sphinx.ext.autosummary` module at the beginning of the class or module.
The content of this autosummary is automatically determined by the results of
the automodule (or autoclass) directive.

This extension gives also the possibility to choose which data shall be shown
and to include the docstring of the ``'__call__'`` attribute.
"""
import logging
import re
import six
import sphinx
from sphinx.ext.autodoc import (
    ClassDocumenter, ModuleDocumenter, ALL, PycodeError,
    ModuleAnalyzer, bool_option, AttributeDocumenter, DataDocumenter, Options,
    prepare_docstring)
import sphinx.ext.autodoc as ad
from sphinx.ext.autosummary import Autosummary, mangle_signature
from docutils import nodes
from docutils.statemachine import ViewList

if sphinx.__version__ >= '1.7':
    from sphinx.ext.autodoc import Signature, get_documenters
    from sphinx.ext.autodoc.directive import (
        AutodocDirective, AUTODOC_DEFAULT_OPTIONS, DocumenterBridge,
        process_documenter_options)
else:
    from sphinx.ext.autodoc import (
        getargspec, formatargspec, AutoDirective as AutodocDirective,
        AutoDirective as AutodocRegistry)

if sphinx.__version__ >= '2.0':
    from sphinx.util import force_decode
else:
    from sphinx.ext.autodoc import force_decode


try:
    from cyordereddict import OrderedDict
except ImportError:
    try:
        from collections import OrderedDict
    except ImportError:
        from ordereddict import OrderedDict

if six.PY2:
    from itertools import imap as map

__version__ = '0.1.10'

__author__ = "Philipp Sommer"


sphinx_version = list(map(float, re.findall(r'\d+', sphinx.__version__)[:3]))

logger = logging.getLogger(__name__)


class AutosummaryDocumenter(object):
    """Abstract class for for extending Documenter methods

    This classed is used as a base class for Documenters in order to provide
    the necessary methods for the :class:`AutoSummDirective`."""

    #: List of functions that may filter the members
    filter_funcs = []

    #: Grouper functions
    grouper_funcs = []

    def __init__(self):
        raise NotImplementedError

    def get_grouped_documenters(self, all_members=False):
        """Method to return the member documenters

        This method is somewhat like a combination of the
        :meth:`sphinx.ext.autodoc.ModuleDocumenter.generate` method and the
        :meth:`sphinx.ext.autodoc.ModuleDocumenter.document_members` method.
        Hence it initializes this instance by importing the object, etc. and
        it finds the documenters to use for the autosummary option in the same
        style as the document_members does it.

        Returns
        -------
        dict
            dictionary whose keys are determined by the :attr:`member_sections`
            dictionary and whose values are lists of tuples. Each tuple
            consists of a documenter and a boolean to identify whether a module
            check should be made describes an attribute or not. The dictionary
            can be used in the
            :meth:`AutoSummDirective.get_items_from_documenters` method

        Notes
        -----
        If a :class:`sphinx.ext.autodoc.Documenter.member_order` value is not
        in the :attr:`member_sections` dictionary, it will be put into an
        additional `Miscallaneous` section."""
        self.parse_name()
        self.import_object()
        # If there is no real module defined, figure out which to use.
        # The real module is used in the module analyzer to look up the module
        # where the attribute documentation would actually be found in.
        # This is used for situations where you have a module that collects the
        # functions and classes of internal submodules.
        self.real_modname = None or self.get_real_modname()

        # try to also get a source code analyzer for attribute docs
        try:
            self.analyzer = ModuleAnalyzer.for_module(self.real_modname)
            # parse right now, to get PycodeErrors on parsing (results will
            # be cached anyway)
            self.analyzer.find_attr_docs()
        except PycodeError as err:
            logger.debug('[autodocsumm] module analyzer failed: %s', err)
            # no source file -- e.g. for builtin and C modules
            self.analyzer = None
            # at least add the module.__file__ as a dependency
            if hasattr(self.module, '__file__') and self.module.__file__:
                self.directive.filename_set.add(self.module.__file__)
        else:
            self.directive.filename_set.add(self.analyzer.srcname)

        self.env.temp_data['autodoc:module'] = self.modname
        if self.objpath:
            self.env.temp_data['autodoc:class'] = self.objpath[0]

        want_all = all_members or self.options.inherited_members or \
            self.options.members is ALL
        # find out which members are documentable
        members_check_module, members = self.get_object_members(want_all)

        # remove members given by exclude-members
        if self.options.exclude_members:
            members = [(membername, member) for (membername, member) in members
                       if membername not in self.options.exclude_members]

        # document non-skipped members
        memberdocumenters = []
        if sphinx_version < [1, 7]:
            registry = AutodocRegistry._registry
        else:
            registry = get_documenters(self.env.app)
        for (mname, member, isattr) in self.filter_members(members, want_all):
            classes = [cls for cls in six.itervalues(registry)
                       if cls.can_document_member(member, mname, isattr, self)]
            if not classes:
                # don't know how to document this member
                continue
            # prefer the documenter with the highest priority
            classes.sort(key=lambda cls: cls.priority)
            # give explicitly separated module name, so that members
            # of inner classes can be documented
            full_mname = self.modname + '::' + \
                '.'.join(self.objpath + [mname])

            documenter = classes[-1](self.directive, full_mname, self.indent)
            memberdocumenters.append((documenter,
                                      members_check_module and not isattr))
        documenters = OrderedDict()
        for e in memberdocumenters:
            section = self.member_sections.get(
                e[0].member_order, 'Miscallaneous')
            if self.env.app:
                e[0].parse_name()
                e[0].import_object()
                user_section = self.env.app.emit_firstresult(
                    'autodocsumm-grouper', self.objtype, e[0].object_name,
                    e[0].object, section, self.object)
                section = user_section or section
            documenters.setdefault(section, []).append(e)
        return documenters


class AutoSummModuleDocumenter(ModuleDocumenter, AutosummaryDocumenter):
    """Module documentor suitable for the :class:`AutoSummDirective`

    This class has the same functionality as the base
    :class:`sphinx.ext.autodoc.ModuleDocumenter` class but with an additional
    `autosummary` and the :meth:`get_grouped_documenters` method.
    It's priority is slightly higher than the one of the ModuleDocumenter."""

    #: slightly higher priority than
    #: :class:`sphinx.ext.autodoc.ModuleDocumenter`
    priority = ModuleDocumenter.priority + 0.1

    #: original option_spec from :class:`sphinx.ext.autodoc.ModuleDocumenter`
    #: but with additional autosummary boolean option
    option_spec = ModuleDocumenter.option_spec
    option_spec['autosummary'] = bool_option

    member_sections = OrderedDict([
        (ad.ClassDocumenter.member_order, 'Classes'),
        (ad.ExceptionDocumenter.member_order, 'Exceptions'),
        (ad.FunctionDocumenter.member_order, 'Functions'),
        (ad.DataDocumenter.member_order, 'Data'),
        ])
    """:class:`~collections.OrderedDict` that includes the autosummary sections

    This dictionary defines the sections for the autosummmary option. The
    values correspond to the :attr:`sphinx.ext.autodoc.Documenter.member_order`
    attribute that shall be used for each section."""


class AutoSummClassDocumenter(ClassDocumenter, AutosummaryDocumenter):
    """Class documentor suitable for the :class:`AutoSummDirective`

    This class has the same functionality as the base
    :class:`sphinx.ext.autodoc.ClassDocumenter` class but with an additional
    `autosummary` option to provide the ability to provide a summary of all
    methods and attributes at the beginning.
    It's priority is slightly higher than the one of the ClassDocumenter"""

    #: slightly higher priority than
    #: :class:`sphinx.ext.autodoc.ClassDocumenter`
    priority = ClassDocumenter.priority + 0.1

    #: original option_spec from :class:`sphinx.ext.autodoc.ClassDocumenter`
    #: but with additional autosummary boolean option
    option_spec = ClassDocumenter.option_spec
    option_spec['autosummary'] = bool_option

    member_sections = OrderedDict([
        (ad.MethodDocumenter.member_order, 'Methods'),
        (ad.AttributeDocumenter.member_order, 'Attributes'),
        ])
    """:class:`~collections.OrderedDict` that includes the autosummary sections

    This dictionary defines the sections for the autosummmary option. The
    values correspond to the :attr:`sphinx.ext.autodoc.Documenter.member_order`
    attribute that shall be used for each section."""


class CallableDataDocumenter(DataDocumenter):
    """:class:`sphinx.ext.autodoc.DataDocumenter` that uses the __call__ attr
    """

    priority = DataDocumenter.priority + 0.1

    def format_args(self):
        # for classes, the relevant signature is the __init__ method's
        callmeth = self.get_attr(self.object, '__call__', None)
        if callmeth is None:
            return None
        if sphinx_version < [1, 7]:
            try:
                argspec = getargspec(callmeth)
            except TypeError:
                # still not possible: happens e.g. for old-style classes
                # with __call__ in C
                return None
            if argspec[0] and argspec[0][0] in ('cls', 'self'):
                del argspec[0][0]
            if sphinx_version < [1, 4]:
                return formatargspec(*argspec)
            else:
                return formatargspec(callmeth, *argspec)
        else:
            try:
                args = Signature(callmeth).format_args()
            except TypeError:
                return None
            else:
                args = args.replace('\\', '\\\\')
                return args

    def get_doc(self, encoding=None, ignore=1):
        """Reimplemented  to include data from the call method"""
        content = self.env.config.autodata_content
        if content not in ('both', 'call') or not self.get_attr(
                self.get_attr(self.object, '__call__', None), '__doc__'):
            return super(CallableDataDocumenter, self).get_doc(
                encoding=encoding, ignore=ignore)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is both docstrings
        docstrings = []
        if content != 'call':
            docstring = self.get_attr(self.object, '__doc__', None)
            docstrings = [docstring + '\n'] if docstring else []
        calldocstring = self.get_attr(
            self.get_attr(self.object, '__call__', None), '__doc__')
        if docstrings:
            docstrings[0] += calldocstring
        else:
            docstrings.append(calldocstring + '\n')

        doc = []
        for docstring in docstrings:
            if not isinstance(docstring, six.text_type):
                docstring = force_decode(docstring, encoding)
            doc.append(prepare_docstring(docstring, ignore))

        return doc


class CallableAttributeDocumenter(AttributeDocumenter):
    """:class:`sphinx.ext.autodoc.AttributeDocumenter` that uses the __call__
    attr
    """

    priority = AttributeDocumenter.priority + 0.1

    def format_args(self):
        # for classes, the relevant signature is the __init__ method's
        callmeth = self.get_attr(self.object, '__call__', None)
        if callmeth is None:
            return None
        if sphinx_version < [1, 7]:
            try:
                argspec = getargspec(callmeth)
            except TypeError:
                # still not possible: happens e.g. for old-style classes
                # with __call__ in C
                return None
            if argspec[0] and argspec[0][0] in ('cls', 'self'):
                del argspec[0][0]
            if sphinx_version < [1, 4]:
                return formatargspec(*argspec)
            else:
                return formatargspec(callmeth, *argspec)
        else:
            try:
                args = Signature(callmeth).format_args()
            except TypeError:
                return None
            else:
                args = args.replace('\\', '\\\\')
                return args

    def get_doc(self, encoding=None, ignore=1):
        """Reimplemented  to include data from the call method"""
        content = self.env.config.autodata_content
        if content not in ('both', 'call') or not self.get_attr(
                self.get_attr(self.object, '__call__', None), '__doc__'):
            return super(CallableAttributeDocumenter, self).get_doc(
                encoding=encoding, ignore=ignore)

        # for classes, what the "docstring" is can be controlled via a
        # config value; the default is both docstrings
        docstrings = []
        if content != 'call':
            docstring = self.get_attr(self.object, '__doc__', None)
            docstrings = [docstring + '\n'] if docstring else []
        calldocstring = self.get_attr(
            self.get_attr(self.object, '__call__', None), '__doc__')
        if docstrings:
            docstrings[0] += calldocstring
        else:
            docstrings.append(calldocstring + '\n')

        doc = []
        for docstring in docstrings:
            if not isinstance(docstring, six.text_type):
                docstring = force_decode(docstring, encoding)
            doc.append(prepare_docstring(docstring, ignore))

        return doc


class AutoSummDirective(AutodocDirective, Autosummary):
    """automodule directive that makes a summary at the beginning of the module

    This directive combines the
    :class:`sphinx.ext.autodoc.directives.AutodocDirective` and
    :class:`sphinx.ext.autosummary.Autosummary` directives to put a summary of
    the specified module at the beginning of the module documentation."""

    if sphinx_version < [1, 7]:
        _default_flags = AutodocDirective._default_flags.union({'autosummary'})
    else:
        AUTODOC_DEFAULT_OPTIONS.append('autosummary')

    @property
    def autosummary_documenter(self):
        """Returns the AutosummaryDocumenter subclass that can be used"""
        try:
            return self._autosummary_documenter
        except AttributeError:
            pass
        objtype = self.name[4:]
        env = self.state.document.settings.env
        if sphinx_version < [1, 7]:
            doc_class = self._registry[objtype]
            params = self
        else:
            reporter = self.state.document.reporter
            try:
                lineno = reporter.get_source_and_line(self.lineno)[1]
            except AttributeError:
                lineno = None
            doc_class = get_documenters(self.env.app)[objtype]
            params = DocumenterBridge(
                env, reporter,
                process_documenter_options(doc_class, env.config,
                                           self.options),
                lineno)
        documenter = doc_class(params, self.arguments[0])
        if hasattr(documenter, 'get_grouped_documenters'):
            self._autosummary_documenter = documenter
            return documenter
        # in case the has been changed in the registry, we decide manually
        if objtype == 'module':
            documenter = AutoSummModuleDocumenter(params, self.arguments[0])
        elif objtype == 'class':
            documenter = AutoSummClassDocumenter(params, self.arguments[0])
        else:
            raise ValueError(
                "Could not find a valid documenter for the object type %s" % (
                    objtype))
        self._autosummary_documenter = documenter
        return documenter

    def run(self):
        """Run method for the directive"""
        doc_nodes = AutodocDirective.run(self)
        if 'autosummary' not in self.options:
            return doc_nodes
        try:
            self.env = self.state.document.settings.env
        except AttributeError:
            pass  # is set automatically with sphinx >= 1.8.0
        if sphinx_version < [2, 0]:
            self.warnings = []
            self.result = ViewList()
        documenter = self.autosummary_documenter
        grouped_documenters = documenter.get_grouped_documenters()
        summ_nodes = self.autosumm_nodes(documenter, grouped_documenters)

        dn = summ_nodes.pop(documenter.fullname)
        if self.name == 'automodule':
            doc_nodes = self.inject_summ_nodes(doc_nodes, summ_nodes)
        # insert the nodes directly after the paragraphs
        if self.name == 'autoclass':
            for node in dn[::-1]:
                self._insert_after_paragraphs(doc_nodes[1], node)
            dn = []
        elif self.name == 'automodule':
            # insert table before the documentation of the members
            istart = 2 if 'noindex' not in self.options else 0
            # if we have a title in the module, we look for the section
            if (len(doc_nodes) >= istart + 1 and
                    isinstance(doc_nodes[istart], nodes.section)):
                others = doc_nodes[istart]
                istart = 2  # skip the title
            else:
                others = doc_nodes
            found = False
            if len(others[istart:]) >= 2:
                for i in range(istart, len(others)):
                    if isinstance(others[i], sphinx.addnodes.index):
                        found = True
                        break
            if found:
                for node in dn[::-1]:
                    others.insert(i, node)
                dn = []
        return self.warnings + dn + doc_nodes

    def _insert_after_paragraphs(self, node, insertion):
        """Inserts the given `insertion` node after the paragraphs in `node`

        This method inserts the `insertion` node after the instances of
        nodes.paragraph in the given `node`.
        Usually the node of one documented class is set up like

        Name of the documented item (allways) (nodes.Element)
        Summary (sometimes) (nodes.paragraph)
        description (sometimes) (nodes.paragraph)
        Parameters section (sometimes) (nodes.rubric)

        We want to be below the description, so we loop until we
        are below all the paragraphs. IF that does not work,
        we simply put it at the end"""
        found = False
        if len(node) >= 2:
            for i in range(len(node[1])):
                if not isinstance(node[1][i], nodes.paragraph):
                    node[1].insert(i + 1, insertion)
                    found = True
                    break
        if not found:
            node.insert(1, insertion)

    def inject_summ_nodes(self, doc_nodes, summ_nodes):
        """Method to inject the autosummary nodes into the autodoc nodes

        Parameters
        ----------
        doc_nodes: list
            The list of nodes as they are generated by the
            :meth:`sphinx.ext.autodoc.AutodocDirective.run` method
        summ_nodes: dict
            The generated autosummary nodes as they are generated by the
            :meth:`autosumm_nodes` method. Note that `summ_nodes` must only
            contain the members autosummary tables!

        Returns
        -------
        doc_nodes: list
            The modified `doc_nodes`

        Notes
        -----
        `doc_nodes` are modified in place and not copied!"""
        def inject_summary(node):
            if isinstance(node, nodes.section):
                for sub in node:
                    inject_summary(sub)
                return
            if (len(node) and (isinstance(node, nodes.section) or (
                    isinstance(node[0], nodes.Element) and
                    node[0].get('module') and node[0].get('fullname')))):
                node_summ_nodes = summ_nodes.get("%s.%s" % (
                    node[0]['module'], node[0]['fullname']))
                if not node_summ_nodes:
                    return
                for summ_node in node_summ_nodes[::-1]:
                    self._insert_after_paragraphs(node, summ_node)
        for node in doc_nodes:
            inject_summary(node)
        return doc_nodes

    def autosumm_nodes(self, documenter, grouped_documenters):
        """Create the autosummary nodes based on the documenter content

        Parameters
        ----------
        documenter: sphinx.ext.autodoc.Documenter
            The base (module or class) documenter for which to generate the
            autosummary tables of its members
        grouped_documenters: dict
            The dictionary as it is returned from the
            :meth:`AutosummaryDocumenter.get_grouped_documenters` method

        Returns
        -------
        dict
            a mapping from the objects fullname to the corresponding
            autosummary tables of its members. The objects include the main
            object of the given `documenter` and the classes that are defined
            in it

        See Also
        --------
        AutosummaryDocumenter.get_grouped_documenters, inject_summ_nodes"""

        summ_nodes = {}
        this_nodes = []
        for section, documenters in six.iteritems(grouped_documenters):
            items = self.get_items_from_documenters(documenters)
            if not items:
                continue
            node = nodes.rubric()
            # create note for the section title (we could also use .. rubric
            # but that causes problems for latex documentations)
            self.state.nested_parse(
                ViewList(['**%s**' % section]), 0, node)
            this_nodes += node
            this_nodes += self.get_table(items)
            for mdocumenter, check_module in documenters:
                if (mdocumenter.objtype == 'class' and
                        not (check_module and not mdocumenter.check_module())):
                    if hasattr(mdocumenter, 'get_grouped_documenters'):
                        summ_nodes.update(self.autosumm_nodes(
                            mdocumenter, mdocumenter.get_grouped_documenters())
                            )
        summ_nodes[documenter.fullname] = this_nodes
        return summ_nodes

    def get_items_from_documenters(self, documenters):
        """Return the items needed for creating the tables

        This method creates the items that are used by the
        :meth:`sphinx.ext.autosummary.Autosummary.get_table` method by what is
        taken from the values of the
        :meth:`AutoSummModuleDocumenter.get_grouped_documenters` method.

        Returns
        -------
        list
            A list containing tuples like
            ``(name, signature, summary_string, real_name)`` that can be used
            for the :meth:`sphinx.ext.autosummary.Autosummary.get_table`
            method."""

        items = []

        max_item_chars = 50
        base_documenter = self.autosummary_documenter
        try:
            base_documenter.analyzer = ModuleAnalyzer.for_module(
                    base_documenter.real_modname)
            attr_docs = base_documenter.analyzer.find_attr_docs()
        except PycodeError as err:
            logger.debug('[autodocsumm] module analyzer failed: %s', err)
            # no source file -- e.g. for builtin and C modules
            base_documenter.analyzer = None
            attr_docs = {}
            # at least add the module.__file__ as a dependency
            if (hasattr(base_documenter.module, '__file__') and
                    base_documenter.module.__file__):
                base_documenter.directive.filename_set.add(
                    base_documenter.module.__file__)
        else:
            base_documenter.directive.filename_set.add(
                base_documenter.analyzer.srcname)

        for documenter, check_module in documenters:
            documenter.parse_name()
            documenter.import_object()
            documenter.real_modname = documenter.get_real_modname()
            real_name = documenter.fullname
            display_name = documenter.object_name
            if display_name is None:  # for instance attributes
                display_name = documenter.objpath[-1]
            if check_module and not documenter.check_module():
                continue

            # -- Grab the signature

            sig = documenter.format_signature()
            if not sig:
                sig = ''
            else:
                max_chars = max(10, max_item_chars - len(display_name))
                sig = mangle_signature(sig, max_chars=max_chars)
#                sig = sig.replace('*', r'\*')

            # -- Grab the documentation

            no_docstring = False
            if documenter.objpath:
                key = ('.'.join(documenter.objpath[:-1]),
                       documenter.objpath[-1])
                try:
                    doc = attr_docs[key]
                    no_docstring = True
                except KeyError:
                    pass
            if not no_docstring:
                documenter.add_content(None)
                doc = documenter.get_doc()
                if doc:
                    doc = doc[0]
                else:
                    continue

            while doc and not doc[0].strip():
                doc.pop(0)

            # If there's a blank line, then we can assume the first sentence /
            # paragraph has ended, so anything after shouldn't be part of the
            # summary
            for i, piece in enumerate(doc):
                if not piece.strip():
                    doc = doc[:i]
                    break

            # Try to find the "first sentence", which may span multiple lines
            m = re.search(r"^([A-Z].*?\.)(?:\s|$)", " ".join(doc).strip())
            if m:
                summary = m.group(1).strip()
            elif doc:
                summary = doc[0].strip()
            else:
                summary = ''

            items.append((display_name, sig, summary, real_name))
        return items


def dont_document_data(config, fullname):
    """Check whether the given object should be documented

    Parameters
    ----------
    config: sphinx.Options
        The configuration
    fullname: str
        The name of the object

    Returns
    -------
    bool
        Whether the data of `fullname` should be excluded or not"""
    if config.document_data is True:
        document_data = [re.compile('.*')]
    else:
        document_data = config.document_data
    if config.not_document_data is True:
        not_document_data = [re.compile('.*')]
    else:
        not_document_data = config.not_document_data
    return (
            # data should not be documented
            (any(re.match(p, fullname) for p in not_document_data)) or
            # or data is not included in what should be documented
            (not any(re.match(p, fullname) for p in document_data)))


class NoDataDataDocumenter(CallableDataDocumenter):
    """DataDocumenter that prevents the displaying of large data"""

    #: slightly higher priority as the one of the CallableDataDocumenter
    priority = CallableDataDocumenter.priority + 0.1

    def __init__(self, *args, **kwargs):
        super(NoDataDataDocumenter, self).__init__(*args, **kwargs)
        fullname = '.'.join(self.name.rsplit('::', 1))
        if hasattr(self.env, 'config') and dont_document_data(
                self.env.config, fullname):
            self.options = Options(self.options)
            self.options.annotation = ' '


class NoDataAttributeDocumenter(CallableAttributeDocumenter):
    """AttributeDocumenter that prevents the displaying of large data"""

    #: slightly higher priority as the one of the CallableAttributeDocumenter
    priority = CallableAttributeDocumenter.priority + 0.1

    def __init__(self, *args, **kwargs):
        super(NoDataAttributeDocumenter, self).__init__(*args, **kwargs)
        fullname = '.'.join(self.name.rsplit('::', 1))
        if hasattr(self.env, 'config') and dont_document_data(
                self.env.config, fullname):
            self.options = Options(self.options)
            self.options.annotation = ' '


def setup(app):
    """setup function for using this module as a sphinx extension"""
    app.setup_extension('sphinx.ext.autosummary')
    app.setup_extension('sphinx.ext.autodoc')

    # make sure to allow inheritance when registering new documenters
    if sphinx_version < [1, 7]:
        registry = AutodocRegistry._registry
    else:
        registry = get_documenters(app)
    for cls in [AutoSummClassDocumenter, AutoSummModuleDocumenter,
                CallableAttributeDocumenter, NoDataDataDocumenter,
                NoDataAttributeDocumenter]:
        if not issubclass(registry.get(cls.objtype), cls):
            try:
                # we use add_documenter because this does not add a new
                # directive
                app.add_documenter(cls)
            except AttributeError:
                app.add_autodocumenter(cls)

    # directives
    if sphinx.__version__ >= '1.8':
        app.add_directive('automodule', AutoSummDirective, override=True)
        app.add_directive('autoclass', AutoSummDirective, override=True)
    else:
        app.add_directive('automodule', AutoSummDirective)
        app.add_directive('autoclass', AutoSummDirective)

    # group event
    app.add_event('autodocsumm-grouper')

    # config value
    app.add_config_value('autodata_content', 'class', True)
    app.add_config_value('document_data', True, True)
    app.add_config_value('not_document_data', [], True)
    return {'version': sphinx.__display_version__, 'parallel_read_safe': True}
