"""autodoc extension for configurable traitlets

Borrowed from https://github.com/dask/dask-gateway/blob/master/docs/sphinxext/autodoc_traitlets.py

"""

from traitlets import TraitType, Undefined
from sphinx.ext.autodoc import DataDocumenter, ModuleDocumenter


class ConfigurableDocumenter(ModuleDocumenter):
    """Specialized Documenter subclass for traits with config=True"""

    objtype = "configurable"
    directivetype = "class"

    def get_object_members(self, want_all):
        """Add traits with .tag(config=True) to members list"""
        get_traits = (
                self.object.class_own_traits
                if self.options.inherited_members
                else self.object.class_traits
        )
        trait_members = []
        for name, trait in sorted(get_traits(config=True).items()):
            # put help in __doc__ where autodoc will look for it
            trait.__doc__ = trait.help
            trait_members.append((name, trait))
        return False, trait_members

    def add_directive_header(self, sig):
        pass

    def add_content(self, more_content):
        pass


class TraitDocumenter(DataDocumenter):
    objtype = "trait"
    directivetype = "data"
    member_order = 1
    priority = 100

    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
        return isinstance(member, TraitType)

    def format_name(self):
        name = ".".join(self.fullname.split(".")[-2:])
        return "c.%s" % name

    def add_directive_header(self, sig):
        default = self.object.default_value
        if default is Undefined:
            default = ""
        # Ensures escape sequences render properly
        default_s = repr(repr(default))[1:-1]
        val = "= {}({})".format(self.object.__class__.__name__, default_s)
        self.options.annotation = val
        self.modname = ""
        return super().add_directive_header(sig)


def setup(app):
    app.add_autodocumenter(ConfigurableDocumenter)
    app.add_autodocumenter(TraitDocumenter)