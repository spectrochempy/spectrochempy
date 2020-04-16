# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

import re
import ast

from traitlets import (HasTraits, Unicode, validate, TraitError, Instance,
                       Float, )

from spectrochempy.core.project.baseproject import AbstractProject

__all__ = ['Script', 'run_script', 'run_all_scripts']


class Script(HasTraits):
    """
    Executable script associated to a project

    """
    _name = Unicode
    _content = Unicode
    _priority = Float(min=0., max=100.)
    _parent = Instance(AbstractProject, allow_none=True)

    def __init__(self, name, content="", parent=None, priority=50.):
        """
        Parameters
        ----------
        name
        content
        parent
        """
        self.name = name
        self.content = content
        self.parent = parent
        self.priority = priority

    # ------------------------------------------------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------------------------------------------------

    def __dir__(self):
        return ['name', 'content', 'parent']

    def __call__(self, *args):

        return self.execute(*args)

    # ------------------------------------------------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------------------------------------------------

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @validate('_name')
    def _name_validate(self, proposal):
        pv = proposal['value']
        if len(pv) < 2:
            raise TraitError('script name must have at least 2 characters')
        p = re.compile("^([^\W0-9]?[a-zA-Z_]+[\w]*)")
        if p.match(pv) and p.match(pv).group() == pv:
            return pv
        raise TraitError('Not a valid script name : only _ letters and numbers '
                         'are valids. For the fist character, numbers are '
                         'not allowed')

    @property
    def content(self):
        return self._content

    @content.setter
    def content(self, value):
        self._content = value

    @validate('_content')
    def _content_validate(self, proposal):
        pv = proposal['value']
        if len(pv) < 1:
            raise TraitError("Script content must be non Null!")

        try:
            py_code = ast.parse(pv)
        except:
            raise

        return pv

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    # ------------------------------------------------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------------------------------------------------
    def execute(self, localvars=None):
        co = 'from spectrochempy import *\n' \
             'import spectrochempy as scp\n' + self._content
        code = compile(co, '<string>', 'exec')
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
            localvars[s] = self.parent

        try:
            exec(code, globals(), localvars)
        except NameError as e:
            error_(e + '. pass the variable `locals()` : this may solve '
                          'this problem! ')


def run_script(script, localvars=None):
    """
    Execute a given project script in the current context.

    Parameters
    ----------
    script : script instance
        The script to execute
    localvars : dict, optional
        If provided it will be used for evaluating the script. In general,
        it can be `localvrs`=``locals()``.

    Returns
    -------
    output of the script if any

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


if __name__ == '__main__':
    pass
