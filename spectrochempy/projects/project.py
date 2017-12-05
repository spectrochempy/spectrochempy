# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to provide a general
# API for displaying, processing and analysing spectrochemical data.
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
# =============================================================================




import uuid

from traitlets import (Dict, List, Bool, Instance, Unicode, HasTraits, This,
                       Any, default)
from traitlets.config.configurable import Configurable

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.pyscripts.pyscripts import PyScript
from spectrochempy.utils.meta import Meta
from .baseproject import AbstractProject

__all__ = ['Project', 'ProjectsOptions']

class ProjectsOptions(Configurable):

    default_directory = Unicode(help='location where all projects are '
                                     'strored by defauult').tag(config=True)

class Project(AbstractProject):
    """A manager for multiple projects and datasets in a main project

    """

    _id = Unicode
    _name = Unicode(allow_none=True)

    _parent = This
    _projects = Dict(This)
    _datasets = Dict(Instance(NDDataset))
    _scripts = Dict(Instance(PyScript))
    _others = Dict()
    _meta = Instance(Meta)


    # ........................................................................
    def __init__(self, *args, name=None, **meta):
        """
        Parameters
        ----------
        args: series of objects, optional.
            argument type will be interpreted correctly if they are of type
            |NDDataset|, |Project|, or other objects such as |Scripts|.
            This is optional, as they can be added later.
        name : str, optional.
            The name of the project.  If the name is not provided, it will
            be generated automatically.
        meta : any other attributes to described the project

        """
        self.parent = None
        self.name = name

        if meta:
            self.meta.update(meta)

        for obj in args:
            self._set_from_type(obj)

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    # ........................................................................
    def _set_from_type(self, obj, name=None):

        if isinstance(obj, NDDataset) :
            # add it to the _datasets dictionary
            self.add_dataset(obj, name)

        elif isinstance(obj ,type(self)):  # can not use Project here!
            self.add_project(obj, name)

        elif isinstance(obj, PyScript):
            self.add_script(obj, name)

        elif hasattr(obj, 'name'):
            self._others[obj.name]=obj

        else:
            raise ValueError('objects of type {} has no name and so '
                             'cannot be appended to the project '.format(type(
                              obj).__name__))

    # ........................................................................
    def _get_from_type(self, name):
        pass


    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    # ........................................................................
    def __getitem__(self, key):

        if not isinstance(key, str):
            raise KeyError('The key must be a string.')

        if key in self.datasets_names:
            return self._datasets[key]
        elif key in self.projects_names:
            return self._projects[key]
        elif key in self.scripts_names:
            return self._scripts[key]
        else:
            raise KeyError("This object name does not exist in this project.")

    # ........................................................................
    def __setitem__(self, key, value):

        if not isinstance(key, str) :
            raise KeyError('The key must be a string.')

        if key in self.allnames and \
                not isinstance(value, type(self[key])) :
            raise ValueError('the key exists but for a different type '
                             'of object: {}'.format(type(self[key]).__name__))

        if key in self.datasets_names:
            self._datasets[key] = value
        elif key in self.projects_names :
            self._projects[key] = value
        elif key in self.scripts_names :
            self._scripts[key] = value
        else:
            # the key does not exists
            self._set_from_type(value, name=key)

    # ........................................................................
    def __getattr__(self, item):

        if "_validate" in item or "_changed" in item:
            # this avoid infinite recursion due to the traits management
            return super().__getattribute__(item)

        elif item in self.allnames:
            # allows to access project, dataset or script by attribute
            return self[item]

        elif item in self.meta.keys():
            # return the attribute
            return self.meta[item]

        else:
            raise AttributeError("`%s` has no attribute `%s`"%(
                type(self).__name__, item))

    # ........................................................................
    def __iter__(self):
        for items in sorted(self._datasets.items()):
            yield items

    # ........................................................................
    def __str__(self):

        s = "Project {}:\n".format(self.name)

        def _listproj(s, project, ns) :
            ns += 1
            sep = "   "*ns

            for k, v in project._projects.items():
                s += "{} ⤷ {} (sub-project)\n".format(sep, k)
                s = _listproj(s, v, ns) # recursive call

            for k, v in project._datasets.items():
                s += "{} ⤷ {} (dataset)\n".format(sep, k)

            for k, v in project._scripts.items():
                s += "{} ⤷ {} (script)\n".format(sep, k)

            return s

        return _listproj(s, self, 0)

    def _repr_html_(self):

        h = self.__str__()
        h = h.replace('\n','<br/>\n')
        h = h.replace(' ','&nbsp;')

        return h

    # ------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------

    # ........................................................................
    @default('_id')
    def _id_default(self):
        return str(uuid.uuid1())  # a unique id

    # ........................................................................
    @property
    def id(self):
        """
        str -  Readonly object identifier.

        """
        return self._id

    # ..................................................
    @property
    def name(self):
        """
        str - An user friendly name for the project

        """
        return self._name

    # .........................................................................
    @name.setter
    def name(self, name):
        # property.setter for name
        if name is not None:
            self._name = name
        else:
            self.name = "Project-" + self.id.split('-')[0]

    # ........................................................................
    @property
    def parent(self):
        return self._parent

    # ........................................................................
    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            # A parent project already exists for this sub-project but the
            # entered values gives a different parent. This is not allowed,
            # as it can produce impredictable results. We will fisrt remove it
            # from the current project.
            self._parent.remove_project(self.name)
        self._parent = value

    # ........................................................................
    @default('_parent')
    def _get_parent(self):
        return None

    # ........................................................................
    @default('_meta')
    def _meta_default(self):
        return Meta()

    # ........................................................................
    @property
    def meta(self):
        return self._meta

    # ........................................................................
    @property
    def datasets_names(self) :
        lst = self._datasets.keys()
        lst = sorted(lst)
        return lst
    # ........................................................................
    @property
    def datasets(self) :
        d = []
        for name in self.datasets_names:
            d.append(self._datasets[name])
        return d

    # ........................................................................
    @property
    def projects_names(self) :
        lst = self._projects.keys()
        lst = sorted(lst)
        return lst

    # ........................................................................
    @property
    def projects(self):
        p = []
        for name in self.projects_names:
            p.append(self._projects[name])
        return p

    # ........................................................................
    @property
    def scripts_names(self) :
        lst = self._scripts.keys()
        lst = sorted(lst)
        return lst

    # ........................................................................
    property
    def scripts(self) :
        s = []
        for name in self.scripts_names:
            s.append(self._scripts[name])
        return s

    @property
    def allnames(self):
        return self.datasets_names+self.projects_names+self.scripts_names



    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    # ........................................................................
    # dataset items

    # ........................................................................
    def add_datasets(self, *args):

        for ds in args:
            self.add_dataset(ds)

    # ........................................................................
    def add_dataset(self, dataset, name=None):
        dataset.parent = self
        if name is None:
            name = dataset.name
        self._datasets[name] = dataset

    # ........................................................................
    def remove_dataset(self, name):
        self._datasets[name]._parent = None  # remove the parent info
        del self._datasets[name] # remove the object from the list of datasets

    # ........................................................................
    def remove_all_dataset(self):
        for v in self._datasets.values():
            v._parent = None
        self._datasets = {}

    # ........................................................................
    # project items

    # ........................................................................
    def add_projects(self, *args):
        """
        Add one or a series of projects to the current project.

        Parameters
        ----------
        *args : project instances


        """
        for proj in args:
            self.add_project(proj)

    # ........................................................................
    def add_project(self, proj, name=None):
        """
        Add one project to the current project.

        Parameters
        ----------
        proj : a project instance


        """
        proj.parent = self
        if name is None:
            name = proj.name
        self._projects[name] = proj

    # ........................................................................
    def remove_project(self, name):
        self._projects[name]._parent = None
        del self._projects[name]

    # ........................................................................
    def remove_all_project(self):
        for v in self._projects.values():
            v._parent = None
        self._projects = {}

    # ........................................................................
    # script items

    # ........................................................................
    def add_scripts(self, *args) :

        for sc in args :
            self.add_script(sc)

    # ........................................................................
    def add_script(self, script, name=None) :
        script.parent = self
        if name is None:
            name = script.name
        self._scripts[name] = script

    # ........................................................................
    def remove_script(self, name) :
        self._scripts[name]._parent = None
        del self._scripts[name]

    # ........................................................................
    def remove_all_script(self) :
        for v in self._scripts.values():
            v._parent = None
        self._scripts = {}



# =============================================================================
if __name__ == '__main__':
    pass