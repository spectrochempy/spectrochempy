# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["Project"]

from copy import deepcopy as cpy
import uuid
import pathlib
from functools import wraps

import dill
from traitlets import Dict, Instance, Unicode, This, default

from spectrochempy.core.dataset.nddataset import NDDataset, NDIO
from spectrochempy.core.scripts.script import Script
from spectrochempy.core.dataset.meta import Meta
from spectrochempy.core.project.baseproject import AbstractProject


# from collections import OrderedDict

# cfg = config_manager
# preferences = preferences


# ======================================================================================================================
# Project class
# ======================================================================================================================
class Project(AbstractProject, NDIO):
    _id = Unicode()
    _name = Unicode(allow_none=True)

    _parent = This()
    _projects = Dict(This)
    _datasets = Dict(Instance(NDDataset))
    _scripts = Dict(Instance(Script))
    _others = Dict()
    _meta = Instance(Meta)

    _filename = Instance(pathlib.Path, allow_none=True)
    _directory = Instance(pathlib.Path, allow_none=True)

    # ..........................................................................
    def __init__(self, *args, argnames=None, name=None, **meta):
        """
        A manager for projects, datasets and scripts.

        It can handle multiple datsets, sub-projects, and scripts in a main project.

        Parameters
        ----------
        *args : Series of objects, optional
            Argument type will be interpreted correctly if they are of type
            |NDDataset|, |Project|, or other objects such as |Script|.
            This is optional, as they can be added later.
        argnames : list, optional
            If not None, this list gives the names associated to each
            objects passed as args. It MUST be the same length that the
            number of args, or an error wil be raised.
            If None, the internal name of each object will be used instead.
        name : str, optional
            The name of the project.  If the name is not provided, it will be
            generated automatically.
        **meta : dict
            Any other attributes to described the project.

        See Also
        --------
        NDDataset : The main object containing arrays.
        Script : Executables scripts container.

        Examples
        --------

        >>> myproj = scp.Project(name='project_1')
        >>> ds = scp.NDDataset([1., 2., 3.], name='dataset_1')
        >>> myproj.add_dataset(ds)
        >>> print(myproj)
        Project project_1:
            ⤷ dataset_1 (dataset)
        """
        super().__init__()

        self.parent = None
        self.name = name

        if meta:
            self.meta.update(meta)

        for i, obj in enumerate(args):
            name = None
            if argnames:
                name = argnames[i]
            self._set_from_type(obj, name)

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    def _set_from_type(self, obj, name=None):

        if isinstance(obj, NDDataset):
            # add it to the _datasets dictionary
            self.add_dataset(obj, name)

        elif isinstance(obj, type(self)):  # can not use Project here!
            self.add_project(obj, name)

        elif isinstance(obj, Script):
            self.add_script(obj, name)

        elif hasattr(obj, "name"):
            self._others[obj.name] = obj

        else:
            raise ValueError(
                "objects of type {} has no name and so "
                "cannot be appended to the project ".format(type(obj).__name__)
            )

    # ..........................................................................
    def _get_from_type(self, name):
        pass  # TODO: ???

    # ..........................................................................
    def _repr_html_(self):

        h = self.__str__()
        h = h.replace("\n", "<br/>\n")
        h = h.replace(" ", "&nbsp;")

        return h

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    def __getitem__(self, key):

        if not isinstance(key, str):
            raise KeyError("The key must be a string.")

        if key == "No project":
            return

        if "/" in key:
            # Case of composed name (we assume not more than one level subproject
            parent = key.split("/")[0]
            if parent in self.projects_names:
                if key in self._projects[parent].datasets_names:
                    return self._projects[parent]._datasets[key]
                elif key in self._projects[parent].scripts_names:
                    return self._projects[parent]._scripts[key]
        if key in self.datasets_names:
            return self._datasets[key]
        elif key in self.projects_names:
            return self._projects[key]
        elif key in self.scripts_names:
            return self._scripts[key]
        else:
            raise KeyError(f"{key}: This object name does not exist in this project.")

    # ..........................................................................
    def __setitem__(self, key, value):

        if not isinstance(key, str):
            raise KeyError("The key must be a string.")

        if key in self.allnames and not isinstance(value, type(self[key])):
            raise ValueError(
                "the key exists but for a different type "
                "of object: {}".format(type(self[key]).__name__)
            )

        if key in self.datasets_names:
            value.parent = self
            self._datasets[key] = value
        elif key in self.projects_names:
            value.parent = self
            self._projects[key] = value
        elif key in self.scripts_names:
            value.parent = self
            self._scripts[key] = value
        else:
            # the key does not exists
            self._set_from_type(value, name=key)

    # ..........................................................................
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
            raise AttributeError(
                "`%s` has no attribute `%s`" % (type(self).__name__, item)
            )

    # ..........................................................................
    def __iter__(self):
        for items in self.allitems:
            yield items

    # ..........................................................................
    def __str__(self):

        s = "Project {}:\n".format(self.name)

        lens = len(s)

        def _listproj(s, project, ns):
            ns += 1
            sep = "   " * ns

            for k, v in project._projects.items():
                s += "{} ⤷ {} (sub-project)\n".format(sep, k)
                s = _listproj(s, v, ns)  # recursive call

            for k, v in project._datasets.items():
                s += "{} ⤷ {} (dataset)\n".format(sep, k)

            for k, v in project._scripts.items():
                s += "{} ⤷ {} (script)\n".format(sep, k)

            if len(s) == lens:
                # nothing has been found in the project
                s += "{} (empty project)\n".format(sep)

            return s.strip("\n")

        return _listproj(s, self, 0)

    def __dir__(self):
        return [
            "name",
            "meta",
            "parent",
            "datasets",
            "projects",
            "scripts",
        ]

    def __copy__(self):
        new = Project()
        # new.name = self.name + '*'
        for item in self.__dir__():
            # if item == 'name':
            #     continue
            item = "_" + item
            data = getattr(self, item)
            # if isinstance(data, (Project,NDDataset, Script)):
            #     setattr(new, item, data.copy())
            # elif item in ['_datasets', '_projects', '_scripts']:
            #
            # else:
            setattr(new, item, cpy(data))
        return new

    # ------------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------------

    # ..........................................................................
    @default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    # ..........................................................................
    @property
    def id(self):
        """
        Readonly object identifier (str).
        """
        return self._id

    # ..........................................................................
    @property
    def name(self):
        """
        An user friendly name for the project.

        The default is automatically generated (str).
        """
        return self._name

    # ..........................................................................
    @name.setter
    def name(self, name):
        # property.setter for name
        if name is not None:
            self._name = name
        else:
            self.name = "Project-" + self.id.split("-")[0]

    # ..........................................................................
    @property
    def parent(self):
        """
        Instance of the Project which is the parent (if any) of the
        current project (project).
        """
        return self._parent

    # ..........................................................................
    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            # A parent project already exists for this sub-project but the
            # entered values gives a different parent. This is not allowed,
            # as it can produce impredictable results. We will fisrt remove it
            # from the current project.
            self._parent.remove_project(self.name)
        self._parent = value

    # ..........................................................................
    @default("_parent")
    def _get_parent(self):
        return None

    # ..........................................................................
    @default("_meta")
    def _meta_default(self):
        return Meta()

    # ..........................................................................
    @property
    def meta(self):
        """
        Metadata for the project (meta).

        meta contains all attribute except the name,
        id and parent of the current project.
        """
        return self._meta

    # ..........................................................................
    @property
    def datasets_names(self):
        """
        Names of all dataset included in this project.
        (does not return those located in sub-folders) (list).
        """
        lst = list(self._datasets.keys())
        return lst

    @property
    def directory(self):
        return self._directory

    # ..........................................................................
    @property
    def datasets(self):
        """
        Datasets included in this project excluding those
        located in subprojects (list).
        """
        d = []
        for name in self.datasets_names:
            d.append(self._datasets[name])
        return d

    @datasets.setter
    def datasets(self, datasets):

        self.add_datasets(*datasets)

    # ..........................................................................
    @property
    def projects_names(self):
        """
        Names of all subprojects included in this project (list).
        """
        lst = list(self._projects.keys())
        return lst

    # ..........................................................................
    @property
    def projects(self):
        """
        Subprojects included in this project (list).
        """
        p = []
        for name in self.projects_names:
            p.append(self._projects[name])
        return p

    @projects.setter
    def projects(self, projects):

        self.add_projects(*projects)

    # ..........................................................................
    @property
    def scripts_names(self):
        """
        Names of all scripts included in this project (list).
        """
        lst = list(self._scripts.keys())
        return lst

    # ..........................................................................
    @property
    def scripts(self):
        """
        Scripts included in this project (list).
        """
        s = []
        for name in self.scripts_names:
            s.append(self._scripts[name])
        return s

    @scripts.setter
    def scripts(self, scripts):

        self.add_scripts(*scripts)

    @property
    def allnames(self):
        """
        Names of all objects contained in this project (list).
        """
        return self.datasets_names + self.projects_names + self.scripts_names

    @property
    def allitems(self):
        """
        All items contained in this project (list).
        """
        return (
            list(self._datasets.items())
            + list(self._projects.items())
            + list(self._scripts.items())
        )

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    # ..........................................................................
    def implements(self, name=None):
        """
        Utility to check if the current object implement `Project`.

        Rather than isinstance(obj, Project) use object.implements('Project').
        This is useful to check type without importing the module.
        """
        if name is None:
            return "Project"
        else:
            return name == "Project"

    def copy(self):
        """
        Make an exact copy of the current project.
        """
        return self.__copy__()

    # ------------------------------------------------------------------------
    # dataset items
    # ------------------------------------------------------------------------

    # ..........................................................................
    def add_datasets(self, *datasets):
        """
        Add several datasets to the current project.

        Parameters
        ----------
        *datasets : series of |NDDataset|
            Datasets to add to the current project.
            The name of the entries in the project will be identical to the
            names of the datasets.

        See Also
        --------
        add_dataset : Add a single dataset to the current project.

        Examples
        --------

        >>> ds1 = scp.NDDataset([1, 2, 3])
        >>> ds2 = scp.NDDataset([4, 5, 6])
        >>> ds3 = scp.NDDataset([7, 8, 9])
        >>> proj = scp.Project()
        >>> proj.add_datasets(ds1, ds2, ds3)
        """
        for ds in datasets:
            self.add_dataset(ds)

    # ..........................................................................
    def add_dataset(self, dataset, name=None):
        """
        Add a single dataset to the current project.

        Parameters
        ----------
        dataset : |NDDataset|
            Datasets to add.
            The name of the entry will be the name of the dataset, except
            if parameter `name` is defined.
        name : str, optional
            If provided the name will be used to name the entry in the project.

        See Also
        --------
        add_datasets : Add several datasets to the current project.

        Examples
        --------

        >>> ds1 = scp.NDDataset([1, 2, 3])
        >>> proj = scp.Project()
        >>> proj.add_dataset(ds1, name='Toto')
        """

        dataset.parent = self
        if name is None:
            name = dataset.name

        n = 1
        while name in self.allnames:
            # this name already exists
            name = f"{dataset.name}-{n}"
            n += 1

        dataset.name = name
        self._datasets[name] = dataset

    # ..........................................................................
    def remove_dataset(self, name):
        """
        Remove a dataset from the project.

        Parameters
        ----------
        name : str
            Name of the dataset to remove.
        """
        self._datasets[name]._parent = None  # remove the parent info
        del self._datasets[name]  # remove the object from the list of datasets

    # ..........................................................................
    def remove_all_dataset(self):
        """
        Remove all dataset from the project.
        """
        for v in self._datasets.values():
            v._parent = None
        self._datasets = {}

    # ------------------------------------------------------------------------
    # project items
    # ------------------------------------------------------------------------

    # ..........................................................................
    def add_projects(self, *projects):
        """
        Add one or a series of projects to the current project.

        Parameters
        ----------
        projects : project instances
            The projects to add to the current ones.
        """
        for proj in projects:
            self.add_project(proj)

    # ..........................................................................
    def add_project(self, proj, name=None):
        """
        Add one project to the current project.

        Parameters
        ----------
        proj : a project instance
            A project to add to the current one.
        """
        proj.parent = self
        if name is None:
            name = proj.name
        else:
            proj.name = name
        self._projects[name] = proj

    # ..........................................................................
    def remove_project(self, name):
        """
        Remove one project from the current project.

        Parameters
        ----------
        name : str
            Name of the project to remove.
        """
        self._projects[name]._parent = None
        del self._projects[name]

    # ..........................................................................
    def remove_all_project(self):
        """
        Remove all projects from the current project.
        """
        for v in self._projects.values():
            v._parent = None
        self._projects = {}

    # ------------------------------------------------------------------------
    # script items
    # ------------------------------------------------------------------------

    # ..........................................................................
    def add_scripts(self, *scripts):
        """
        Add one or a series of scripts to the current project.

        Parameters
        ----------
        scripts : |Script| instances
        """
        for sc in scripts:
            self.add_script(sc)

    # ..........................................................................
    def add_script(self, script, name=None):
        """
        Add one script to the current project.

        Parameters
        ----------
        script : a |Script| instance
        name : str
        """
        script.parent = self
        if name is None:
            name = script.name
        else:
            script.name = name
        self._scripts[name] = script

    # ..........................................................................
    def remove_script(self, name):
        self._scripts[name]._parent = None
        del self._scripts[name]

    # ..........................................................................
    def remove_all_script(self):
        for v in self._scripts.values():
            v._parent = None
        self._scripts = {}


def makescript(priority=50):
    def decorator(func):
        ss = dill.dumps(func)
        print(ss)

        @wraps(func)
        def wrapper(*args, **kwargs):
            f = func(*args, **kwargs)
            return f

        return wrapper

    return decorator


# ======================================================================================================================
if __name__ == "__main__":
    pass
