# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

import re
import ast

from traitlets import (
    HasTraits,
    Unicode,
    validate,
    TraitError,
    Instance,
    Float,
)

from spectrochempy.core.project.baseproject import AbstractProject
from spectrochempy.core import error_

__all__ = ["Script", "run_script", "run_all_scripts"]


class Script(HasTraits):

    _name = Unicode()
    _content = Unicode(allow_none=True)
    _priority = Float(min=0.0, max=100.0)
    _parent = Instance(AbstractProject, allow_none=True)

    def __init__(self, name="unamed_script", content=None, parent=None, priority=50.0):
        """
        Executable scripts.

        The scripts are used in a project.

        Parameters
        ----------
        name : str
            Name of the script. The name should be unique.
        content : str
            Content of sthe script.
        parent : instance of |Project|
            Parent project.
        priority: int
            Default=50.

        See Also
        --------
        Project: Object containing |NDDataset|s, sub-|Project|s and |Scripts|.

        Examples
        --------
        Make a script

        >>> s = "set_loglevel(INFO)"
        >>> s = "info_('Hello')"
        >>> myscript = scp.Script("print_hello_info", s)

        Execute a script

        >>> scp.run_script(myscript)
        """

        self.name = name
        self.content = content
        self.parent = parent
        self.priority = priority

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __dir__(self):
        return ["name", "content", "parent"]

    def __call__(self, *args):
        return self.execute(*args)

    def __eq__(self, other):
        if self._content == other.content:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # ------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @validate("_name")
    def _name_validate(self, proposal):
        pv = proposal["value"]
        if len(pv) < 2:
            raise TraitError("script name must have at least 2 characters")
        p = re.compile(r"^([^\W0-9]?[a-zA-Z_]+[\w]*)")
        if p.match(pv) and p.match(pv).group() == pv:
            return pv
        raise TraitError(
            "Not a valid script name : only _ letters and numbers "
            "are valids. For the fist character, numbers are "
            "not allowed"
        )

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @validate("_content")
    def _content_validate(self, proposal):

        pv = proposal["value"]
        if len(pv) < 1:  # do not allow null but None
            raise TraitError("Script content must be non Null!")
        if pv is None:
            return

        try:
            ast.parse(pv)
        except Exception:
            raise

        return pv

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------
    # ..........................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `Project`.

        Rather than isinstance(obj, Project) use object.implements('Project').
        This is useful to check type without importing the module

        Parameters
        ----------
        name: Object type name, optional
            If not None, the function return True is the object type correspond to name.
            If None the function return the object type name.

        Returns
        -------
        Bool or str
        """
        if name is None:
            return "Script"
        else:
            return name == "Script"

    def execute(self, localvars=None):
        co = (
            "from spectrochempy import *\n"
            "import spectrochempy as scp\n" + self._content
        )
        code = compile(co, "<string>", "exec")
        if localvars is None:
            # locals was not passed, try to avoid missing values for name
            # such as 'project', 'proj', 'newproj'...
            # other missing name if they correspond to the parent project
            # will be subtitued latter upon exception
            localvars = locals()
            # localvars['proj']=self.parent
            # localvars['project']=self.parent

        try:
            exec(code, globals(), localvars)
            return

        except NameError as e:
            # most of the time, a script apply to a project
            # let's try to substitute the parent to the missing name
            regex = re.compile(r"'(\w+)'")
            s = regex.search(e.args[0]).group(1)
            localvars[s] = self.parent  # lgtm[py/modification-of-locals]
            # TODO: check if this a real error or not  (need to come
            #  back on this later)
        try:
            exec(code, globals(), localvars)
        except NameError as e:
            error_(
                e + ". pass the variable `locals()` : this may solve " "this problem! "
            )


def run_script(script, localvars=None):
    """
    Execute a given project script in the current context.

    Parameters
    ----------
    script : script instance
        The script to execute.
    localvars : dict, optional
        If provided it will be used for evaluating the script. In general,
        it can be `localvrs`=``locals()``.

    Returns
    -------
    out
        Output of the script if any
    """

    return script.execute(localvars)


def run_all_scripts(project):
    """
    Execute all scripts in a project following their priority.

    Parameters
    ----------
    project : project instance
        The project in which the scripts have to be executed
    """
    # TODO: complete this run_all_script function

    project


if __name__ == "__main__":
    pass
