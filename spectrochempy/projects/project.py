# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


# ============================================================================
# standard library import
# ============================================================================

import os
import uuid
import warnings
import json
from io import BytesIO
from functools import wraps

# ============================================================================
# third party imports
# ============================================================================

from traitlets import (Dict, List, Bool, Instance, Unicode, This, Any, default)
#import dill

# ============================================================================
# local imports
# ============================================================================

from spectrochempy.application import app
from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.scripts.script import Script
from spectrochempy.utils import (Meta, SpectroChemPyWarning, make_zipfile,
                                 ScpFile, )

from spectrochempy.projects.baseproject import AbstractProject

__all__ = ['Project']

log = app.log
options = app
projectsoptions = app.projectsoptions


# ============================================================================
# Project class
# ============================================================================
class Project(AbstractProject):
    """A manager for multiple projects and datasets in a main project

    """

    _id = Unicode
    _name = Unicode(allow_none=True)

    _parent = This
    _projects = Dict(This)
    _datasets = Dict(Instance(NDDataset))
    _scripts = Dict(Instance(Script))
    _others = Dict()
    _meta = Instance(Meta)
    _filename = Unicode

    # ........................................................................
    def __init__(self, *args, argnames=None, name=None, **meta):
        """
        Parameters
        ----------
        args: series of objects, optional.
            argument type will be interpreted correctly if they are of type
            |NDDataset|, |Project|, or other objects such as |Script|.
            This is optional, as they can be added later.
        argnames : list, optional
            If not None, this list gives the names associated to each
            objects passed as args. It MUST be the same length that the
            number of args, or an error wil be raised.
            If None, the internal name of each object will be used instead.
        name : str, optional.
            The name of the project.  If the name is not provided, it will
            be generated automatically.
        meta : any other attributes to described the project

        """
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

    # ........................................................................
    def _set_from_type(self, obj, name=None):

        if isinstance(obj, NDDataset):
            # add it to the _datasets dictionary
            self.add_dataset(obj, name)

        elif isinstance(obj, type(self)):  # can not use Project here!
            self.add_project(obj, name)

        elif isinstance(obj, Script):
            self.add_script(obj, name)

        elif hasattr(obj, 'name'):
            self._others[obj.name] = obj

        else:
            raise ValueError('objects of type {} has no name and so '
                             'cannot be appended to the project '.format(
                type(obj).__name__))

    # ........................................................................
    def _get_from_type(self, name):
        pass
        # TODO: ???

    # ........................................................................
    def _repr_html_(self):

        h = self.__str__()
        h = h.replace('\n', '<br/>\n')
        h = h.replace(' ', '&nbsp;')

        return h

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

        if not isinstance(key, str):
            raise KeyError('The key must be a string.')

        if key in self.allnames and not isinstance(value, type(self[key])):
            raise ValueError('the key exists but for a different type '
                             'of object: {}'.format(type(self[key]).__name__))

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
            raise AttributeError(
                "`%s` has no attribute `%s`" % (type(self).__name__, item))

    # ........................................................................
    def __iter__(self):
        for items in sorted(self.allitems):
            yield items

    # ........................................................................
    def __str__(self):

        s = "Project {}:\n".format(self.name)

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

            return s

        return _listproj(s, self, 0)

    def __dir__(self):
        return ['name', 'meta', 'parent', 'datasets', 'projects', 'scripts', ]

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
        str - Readonly object identifier.

        """
        return self._id

    # ..................................................
    @property
    def name(self):
        """
        str - An user friendly name for the project. The default is
        automatically generated.

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
        """
        project - instance of the Project which is the parent (if any) of the
        current project.

        """
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
        """
        meta - instance of Meta that contains all attribute except the name,
        id and parent of the current project.

        """
        return self._meta

    # ........................................................................
    @property
    def datasets_names(self):
        """
        list - names of all dataset included in this project
        (does not return those located in sub-folders).

        """
        lst = self._datasets.keys()
        lst = sorted(lst)
        return lst

    # ........................................................................
    @property
    def datasets(self):
        """
        list - datasets included in this project excluding those
        located in subprojects.

        """
        d = []
        for name in self.datasets_names:
            d.append(self._datasets[name])
        return d

    # ........................................................................
    @property
    def projects_names(self):
        """
        list - names of all subprojects included in this project.

        """
        lst = self._projects.keys()
        lst = sorted(lst)
        return lst

    # ........................................................................
    @property
    def projects(self):
        """
        list - subprojects included in this project.

        """
        p = []
        for name in self.projects_names:
            p.append(self._projects[name])
        return p

    # ........................................................................
    @property
    def scripts_names(self):
        """
        list - names of all scripts included in this project.

        """
        lst = self._scripts.keys()
        lst = sorted(lst)
        return lst

    # ........................................................................
    @property
    def scripts(self):
        """
        list - scripts included in this project.

        """
        s = []
        for name in self.scripts_names:
            s.append(self._scripts[name])
        return s

    @property
    def allnames(self):
        """
        list - names of all objects contained in this project

        """
        return self.datasets_names + self.projects_names + self.scripts_names

    @property
    def allitems(self):
        """
        list - all items contained in this project

        """
        return list(self._datasets.items()) + list(self._projects.items()) + \
               list(self._scripts.items())

    @property
    def filename(self):
        """
        str - current filename for this project.

        """
        if self._filename:
            return os.path.basename(self._filename)
        else:
            return self.id

    @property
    def directory(self):
        """
        str - current directory for this project.

        """
        if self._filename:
            return os.path.dirname(self._filename)
        else:
            return ''

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    # ........................................................................
    # dataset items

    # ........................................................................
    def add_datasets(self, *datasets):
        """
        Add datasets to the current project.

        Parameters
        ----------
        datasets : series of |NDDataset|
            Datasets to add to the current project.
            The name of the entries in the project will be identical to the
            names of the datasets.

        Examples
        --------

        Assuming that ds1, ds2 and ds3 are already defined datasets:

        >>> proj = Project()
        >>> proj.add_datasets(ds1, ds2, ds3)

        """
        for ds in datasets:
            self.add_dataset(ds)

    # ........................................................................
    def add_dataset(self, dataset, name=None):
        """
        Add datasets to the current project.

        Parameters
        ----------
        dataset : |NDDataset|
            Datasets to add.
            The name of the entry will be the name of the dataset, except
            if parameter `name` is defined.
        name : str, optional
            If provided the name will be used to name the entry in the project.

        Examples
        --------

        Assuming that ds1 is an already defined dataset:

        >>> proj = Project()
        >>> proj.add_dataset(ds1, name='Toto')

        """

        dataset.parent = self
        if name is None:
            name = dataset.name
        self._datasets[name] = dataset

    # ........................................................................
    def remove_dataset(self, name):
        """
        Remove a dataset from the project

        Parameters
        ----------
        name : str
            Name of the dataset to remove.


        """
        self._datasets[name]._parent = None  # remove the parent info
        del self._datasets[name]  # remove the object from the list of datasets

    # ........................................................................
    def remove_all_dataset(self):
        """
        Remove all dataset from the project

        """
        for v in self._datasets.values():
            v._parent = None
        self._datasets = {}

    # ........................................................................
    # project items

    # ........................................................................
    def add_projects(self, *projects):
        """
        Add one or a series of projects to the current project.

        Parameters
        ----------
        projects : project instances


        """
        for proj in projects:
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
        """
        remove one project from the current project.

        Parameters
        ----------
        name : str
            Name of the project to remove

        """
        self._projects[name]._parent = None
        del self._projects[name]

    # ........................................................................
    def remove_all_project(self):
        """
        remove all projects from the current project.

        """
        for v in self._projects.values():
            v._parent = None
        self._projects = {}

    # ........................................................................
    # script items

    # ........................................................................
    def add_scripts(self, *scripts):
        """
         Add one or a series of scripts to the current project.

         Parameters
         ----------
         scripts : |Script| instances


         """
        for sc in scripts:
            self.add_script(sc)

    # ........................................................................
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
        self._scripts[name] = script

    # ........................................................................
    def remove_script(self, name):
        self._scripts[name]._parent = None
        del self._scripts[name]

    # ........................................................................
    def remove_all_script(self):
        for v in self._scripts.values():
            v._parent = None
        self._scripts = {}

    # ........................................................................
    def save(self, filename=None, directory=None, overwrite_data=True,
             **kwargs):
        """
        Save the current project
        (default extension: ``.pscp`` ).

        Parameters
        ----------
        filename : str
            The filename of the file where to save the current dataset
        directory : str, optional.
            If the destination path is not given, the project will be saved in
            the default location defined in the configuration options.
        overwrite_data : bool
            If True the default, everything is saved, even if the data
            already exists (overwrite. If False, only other object or
            attributes can be changed. This allow to keep the original data
            intact. Processing step that may change this data can be saved in
            scripts.

        See Also
        --------
        load

        """

        directory = kwargs.get("directory", projectsoptions.projects_directory)

        if not filename:
            # the current file name or default filename (project name)
            filename = self.name
            if self.directory:
                directory = self.directory

        if not os.path.exists(directory):
            raise IOError("directory doesn't exists!")

        if not filename.endswith('.pscp'):
            filename = filename + '.pscp'

        if os.path.isdir(directory):
            filename = os.path.expanduser(os.path.join(directory, filename))
        else:
            warnings.warn('Provided directory is a file, '
                          'so we use its parent directory',
                          SpectroChemPyWarning)
            filename = os.path.join(os.path.dirname(directory), filename)


        global savedproj, keepdata
        keepdata = False
        if not overwrite_data and os.path.exists(filename):
            # We need to check if the file already exists, as we want not to
            # change the original data
            keepdata = True
            # get the original project
            savedproj = Project.load(filename)

        # Imports deferred for startup time improvement
        import zipfile
        import tempfile

        compression = zipfile.ZIP_DEFLATED
        zipf = make_zipfile(filename, mode="w", compression=compression)

        # Stage arrays in a temporary file on disk, before writing to zip.
        fd, tmpfile = tempfile.mkstemp(suffix='-spectrochempy.ds.scp')
        os.close(fd)

        pars = {}
        objnames = dir(self)

        def _loop_on_obj(_names, obj=self, parent='', level='main.'):

            for key in _names:

                val = getattr(obj, "_%s" % key)
                if val is None:
                    # ignore None - when reading if something is missing it
                    # will be considered as None anyways
                    continue

                elif key == 'projects':
                    pars[level + key] = []
                    for k, proj in val.items():
                        _objnames = dir(proj)
                        _loop_on_obj(_objnames, obj=proj, parent=level[:-1],
                                     level=k + '.')
                        pars[level + key].append(k)

                elif key == 'datasets':
                    pars[level + key] = []
                    for k, ds in val.items():
                        if not keepdata:
                            ds.save(filename=tmpfile)
                        else:
                            # we take the saved version
                            if level=='main.':
                                savedproj[k].save(filename=tmpfile)
                            else:
                                savedproj[level[:-1]][k].save(filename=tmpfile)
                        fn = '%s%s.scp' % (level, k)
                        zipf.write(tmpfile, arcname=fn)
                        pars[level + key].append(fn)

                elif key == 'scripts':
                    pars[level + key] = []
                    for k, sc in val.items():
                        _objnames = dir(sc)
                        _loop_on_obj(_objnames, obj=sc, parent=level[:-1],
                                     level=k + '.')
                        pars[level + key].append(k)

                elif isinstance(val, Meta):
                    # we assume that objects in the meta objects
                    # are all json serialisable #TODO: could be imporved.
                    pars[level + key] = val.to_dict()

                elif key == 'parent':
                    pars[level + key] = parent

                else:
                    # probably some string
                    pars[level + key] = val

        # Recursive scan on Project content
        _loop_on_obj(objnames)

        with open(tmpfile, 'w') as f:
            f.write(json.dumps(pars, sort_keys=True, indent=2))

        zipf.write(tmpfile, arcname='pars.json')
        os.remove(tmpfile)
        zipf.close()

        self._filename = filename

    @classmethod
    def load(cls, filename='', directory=None, **kwargs):
        """Load a project file ( extension: ``.pscp``).

        It's a class method, that can be used directly on the class,
        without prior opening of a class instance.

        Parameters
        ----------
        filename : str
            The filename to the file to be read.
        directory : str, optional, default= ``scpdata``
            The directory from where to load the file. If this information is
            not given, the project will be loaded if possible from
            the default location defined in the configuration options.

        See Also
        --------
        save


        """
        # TODO: use pathlib instead of os.path? may simplify the code.

        if not filename:
            raise IOError('no filename provided!')

        directory, filename = os.path.split(filename)

        if not os.path.exists(directory):
            directory = kwargs.get("directory",
                                   projectsoptions.projects_directory)
        elif kwargs.get("directory", None) is not None:
            warnings.warn("got multiple directory information. Use that "
                          "obtained "
                          "from filename!", SpectroChemPyWarning)

        filename = os.path.expanduser(os.path.join(directory, filename))
        if not os.path.exists(filename) and not filename.endswith('.pscp'):
            filename = filename + '.pscp'
            if not os.path.exists(filename):
                raise IOError('no valid project filename provided')

        fid = open(filename, 'rb')

        # open the zip file as a dict-like object
        obj = ScpFile(fid)

        # read json file
        pars = obj['pars.json']

        # make a project (or a subclass of it, so we use cls)
        def _make_project(_cls, pars, obj, pname):

            args = []
            argnames = []

            projects = pars['%s.projects' % pname]
            for item in projects:
                args.append(_make_project(Project, pars, obj, item))
                argnames.append(item)

            datasets = pars['%s.datasets' % pname]
            for item in datasets:
                args.append(obj[item])
                item = item.split('.')
                argnames.append(item[-2])

            scripts = pars['%s.scripts' % pname]
            for item in scripts:
                args.append(Script(item, pars['%s.content'%item]))
                argnames.append(item)

            name = pars['%s.name' % pname]
            meta = pars['%s.meta' % pname]

            new = _cls(*args, argnames=argnames, name=name, **meta)

            return new

        return _make_project(cls, pars, obj, 'main')


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


# =============================================================================
if __name__ == '__main__':
    pass

    import dill
    from spectrochempy.api import *

    options.log_level = ERROR

    scripts = []

    @makescript(priority=100)  # priority 100 means: it will be executed
                               # before any other scripts
    def preprocess_IR(proj):
        """

        Parameters
        ----------
        proj

        Returns
        -------
        datasets, labels : datasets and assotiated labels
        """
        datasets = []
        labels = []
        for p in reversed(proj.projects):  # we put the parent first this way
            p.IR[:, 3998.:1293., INPLACE]  # let's keep only the useful
            # region
            print(proj)
            datasets.append(p.IR)
            labels.append(p.label)
        for ds in datasets:
            ds.y -= ds.y[0]  # remove the timestamp offset from the axe y
            ds.y.ito('hour')  # adapt units to the length of the acquisition
            # for a prettier display
            ds.y.title = 'Time-on-stream'
            # put some explicit title (to replace timestamps)
        return (datasets, labels)


    @makescript(priority=0)  # priority 0, it will be executed after
                             # all other scripts
    def plot_IR(datasets, labels, figsize=(6.8, 3), dpi=100, style='sans'):
        multiplot_stack(sources=datasets, labels=labels, nrow=1, ncol=3,
                        sharex=True, sharez=True, figsize=figsize, dpi=dpi,
                        style=style)
        return None


    # options.log_level = DEBUG
    proj = Project.load('HIZECOKE')
    datasets, labels = preprocess_IR(proj)
    plot_IR(datasets, labels)
    show()
