# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["Project"]

import copy as cpy
import pathlib
import textwrap
import uuid
import warnings

import dill  # noqa: F401
import traitlets as tr

from spectrochempy.core.dataset.nddataset import NDIO
from spectrochempy.core.project.abstractproject import AbstractProject
from spectrochempy.utils.meta import Meta
from spectrochempy.utils.print import DisplayItem
from spectrochempy.utils.print import DisplaySection
from spectrochempy.utils.print import _html_heading
from spectrochempy.utils.print import _render_sections
from spectrochempy.utils.print import colored_output
from spectrochempy.utils.traits import NDDatasetType

# from collections import OrderedDict

# cfg = config_manager
# preferences = preferences


# ======================================================================================
# Project class
# ======================================================================================
@tr.signature_has_traits
class Project(AbstractProject, NDIO):
    """
    A manager for projects and datasets.

    It can handle multiple datasets and subprojects in a main
    project.

    Parameters
    ----------
    *args : Series of objects, optional
        Argument type will be interpreted correctly if they are of type
        `NDDataset` or `Project` .
        This is optional, as they can be added later.
    argnames : list, optional
        If not None, this list gives the names associated to each
        object passed as args. It MUST be the same length that the
        number of args, or an error will be raised.
        If None, the internal name of each object will be used instead.
    name : str, optional
        The name of the project.  If the name is not provided, it will be
        generated automatically.
    **meta : dict
        Any other attributes to describe the project.

    See Also
    --------
    NDDataset : The main object containing arrays.

    Examples
    --------
    >>> import spectrochempy as scp
    >>> myproj = scp.Project(name='project_1')
    >>> ds = scp.NDDataset([1., 2., 3.], name='dataset_1')
    >>> myproj.add_dataset(ds)
    >>> print(myproj)
    Project project_1:
        ⤷ dataset_1 (dataset)

    """

    _id = tr.Unicode()
    _name = tr.Unicode(allow_none=True)
    _explicit_name = tr.Bool(False)

    _parent = tr.This()
    _projects = tr.Dict(tr.This())
    _datasets = tr.Dict(NDDatasetType())
    _meta = tr.Instance(Meta)

    _filename = tr.Instance(pathlib.Path, allow_none=True)
    _directory = tr.Instance(pathlib.Path, allow_none=True)
    _html_output = tr.Bool(False)

    def __init__(self, *args, argnames=None, name=None, **meta):
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

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _set_from_type(self, obj, name=None):
        from spectrochempy.core.dataset.nddataset import NDDataset

        if isinstance(obj, NDDataset):
            # add it to the _datasets dictionary
            self.add_dataset(obj, name)

        elif isinstance(obj, type(self)):
            self.add_project(obj, name)

        else:
            raise TypeError(
                f"Project does not accept objects of type {type(obj).__name__}. "
                "Only NDDataset and Project instances are supported."
            )

    def _get_from_type(self, name):
        pass  # TODO: ???

    def _repr_html_(self):
        sections = self._repr_sections()
        body = _render_sections(sections)
        heading = _html_heading(self)
        return (
            '<div class="scp-output">'
            f"<details><summary>{heading}</summary>\n{body}\n"
            "</details></div>"
        )

    def _repr_sections(self):
        sections: list[DisplaySection] = []

        # ------------------------------------------------------------------
        # SUMMARY
        # ------------------------------------------------------------------
        summary_items: list[DisplayItem] = []
        summary_items.append(DisplayItem("field", self.name, "name"))
        author = self.meta.get("author", None)
        if author:
            summary_items.append(DisplayItem("field", author, "author"))
        description = self.meta.get("description", None)
        if description:
            summary_items.append(
                DisplayItem("field", description.strip(), "description")
            )
        sections.append(DisplaySection("summary", "Summary", summary_items))

        # ------------------------------------------------------------------
        # DATA — project hierarchy
        # ------------------------------------------------------------------
        data_items: list[DisplayItem] = []
        str_output = self.__str__()
        lines = str_output.split("\n")
        hier_lines = lines[1:] if len(lines) > 1 else []
        if not hier_lines:
            hier_lines = ["(empty project)"]
        for line in hier_lines:
            n_spaces = len(line) - len(line.lstrip())
            line = "&nbsp;" * n_spaces + line.lstrip()
            data_items.append(DisplayItem("block", line))
        sections.append(DisplaySection("data", "Data", data_items))

        return sections

    def __repr__(self):
        return f"Project: {self.name}"

    def _cstr(self):
        out = ""
        out += f"         name: {self.name}\n"

        author = self.meta.get("author", None)
        if author:
            out += f"       author: {author}\n"

        description = self.meta.get("description", None)
        if description:
            wrapper = textwrap.TextWrapper(
                initial_indent="",
                subsequent_indent=" " * 15,
                replace_whitespace=True,
                width=80,
            )
            pars = description.strip().splitlines()
            desc = ""
            if pars:
                desc += f"{wrapper.fill(pars[0])}\n"
            for par in pars[1:]:
                desc += "{}\n".format(textwrap.indent(par, " " * 15))
            desc = f"\0\0\0{desc.rstrip()}\0\0\0\n"
            out += "  description: "
            out += desc

        out += "DATA\n"

        str_output = self.__str__()
        lines = str_output.split("\n")
        hier_lines = lines[1:] if len(lines) > 1 else []
        out += "\n".join(hier_lines)

        if not out.endswith("\n"):
            out += "\n"
        out += "\n"

        if not self._html_output:
            return colored_output(out.rstrip())
        return out.rstrip()

    # ----------------------------------------------------------------------------------
    # Special methods
    # ----------------------------------------------------------------------------------
    def __getitem__(self, key):
        if not isinstance(key, str):
            raise KeyError("The key must be a string.")

        if key == "No project":
            return None

        if "/" in key:
            # Case of composed name (we assume not more than one level subproject
            parent, child = key.split("/")[0], key.split("/")[1]
            if (
                parent in self.projects_names
                and child in self._projects[parent].datasets_names
            ):
                return self._projects[parent]._datasets[child]
        if key in self.datasets_names:
            return self._datasets[key]
        if key in self.projects_names:
            return self._projects[key]
        raise KeyError(f"{key}: This object name does not exist in this project.")

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("The key must be a string.")

        if key in self.allnames and not isinstance(value, type(self[key])):
            raise ValueError(
                "the key exists but for a different type "
                f"of object: {type(self[key]).__name__}",
            )

        if key in self.datasets_names:
            value.parent = self
            self._datasets[key] = value
        elif key in self.projects_names:
            value.parent = self
            self._projects[key] = value
        else:
            # the key does not exist
            self._set_from_type(value, name=key)

    def __getattr__(self, item):
        if "_validate" in item or "_changed" in item:
            # this avoids infinite recursion due to the traits management
            return super().__getattribute__(item)

        if item in self.allnames:
            # allows to access project, dataset or script by attribute
            return self[item]

        if item in self.meta:
            # return the attribute
            return self.meta[item]

        raise AttributeError

    def __iter__(self):
        yield from self.allitems

    def __str__(self):
        s = f"Project {self.name}:\n"

        lens = len(s)

        def _listproj(s, project, ns):
            ns += 1
            sep = "   " * ns

            for k, v in project._projects.items():
                s += f"{sep} ⤷ {k} (sub-project)\n"
                s = _listproj(s, v, ns)  # recursive call

            for k, _v in project._datasets.items():
                s += f"{sep} ⤷ {k} (dataset)\n"

            if len(s) == lens:
                # nothing has been found in the project
                s += f"{sep} (empty project)\n"

            return s

        return _listproj(s, self, 0).rstrip("\n")

    def _attributes_(self):
        return [
            "name",
            "meta",
            "parent",
            "datasets",
            "projects",
        ]

    def __copy__(self):
        new = Project()
        for item in self._attributes_():
            item = "_" + item
            data = getattr(self, item)
            setattr(new, item, cpy.copy(data))
        return new

    # ----------------------------------------------------------------------------------
    # properties
    # ----------------------------------------------------------------------------------
    @tr.default("_id")
    def _id_default(self):
        # a unique id
        return f"{type(self).__name__}_{str(uuid.uuid1()).split('-')[0]}"

    @property
    def id(self):
        """Readonly object identifier (str)."""
        return self._id

    @property
    def name(self):
        """
        A user-friendly name for the project.

        The default is automatically generated (str).
        """
        return self._name

    @name.setter
    def name(self, name):
        if name is not None:
            self._name = name
            self._explicit_name = True
        else:
            self._name = "Project-" + self.id.split("-")[0]

    @property
    def has_defined_name(self):
        """True if the name was explicitly provided by the user (bool)."""
        return self._explicit_name

    @property
    def parent(self):
        """Instance of the Project which is the parent (if any) of the current project (project)."""
        return self._parent

    @parent.setter
    def parent(self, value):
        if self._parent is not None:
            # A parent project already exists for this subproject but the
            # entered values gives a different parent. This is not allowed,
            # as it can produce impredictable results. We will first remove it
            # from the current project.
            self._parent.remove_project(self.name)
        self._parent = value

    @tr.default("_parent")
    def _get_parent(self):
        return None

    @tr.default("_meta")
    def _meta_default(self):
        return Meta()

    @property
    def meta(self):
        """
        Metadata for the project (meta).

        meta contains all attribute except the name,
        id and parent of the current project.
        """
        return self._meta

    @property
    def datasets_names(self):
        """
        Names of all dataset included in this project.

        (does not return those located in sub-folders) (list).
        """
        return list(self._datasets.keys())

    @property
    def directory(self):
        return self._directory

    @property
    def datasets(self):
        """
        Datasets included in this project excluding those located in subprojects.

        (list).
        """
        d = []
        for name in self.datasets_names:
            d.append(self._datasets[name])
        return d

    @datasets.setter
    def datasets(self, datasets):
        self.add_datasets(*datasets)

    @property
    def projects_names(self):
        """Names of all subprojects included in this project (list)."""
        return list(self._projects.keys())

    @property
    def projects(self):
        """Subprojects included in this project (list)."""
        p = []
        for name in self.projects_names:
            p.append(self._projects[name])
        return p

    @projects.setter
    def projects(self, projects):
        self.add_projects(*projects)

    @property
    def allnames(self):
        """Names of all objects contained in this project (list)."""
        return self.datasets_names + self.projects_names

    @property
    def allitems(self):
        """All items contained in this project (list)."""
        return list(self._datasets.items()) + list(self._projects.items())

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _implements(name=None):
        """
        Check if the current object implements `Project` .

        Rather than isinstance(obj, Project) use object._implements('Project').
        This is useful to check type without importing the module.
        """
        if name is None:
            return "Project"
        return name == "Project"

    def copy(self):
        """Make an exact copy of the current project."""
        return self.__copy__()

    # ----------------------------------------------------------------------------------
    # dataset items
    # ----------------------------------------------------------------------------------
    def add_datasets(self, *datasets):
        """
        Add several datasets to the current project.

        Parameters
        ----------
        *datasets : series of `NDDataset`
            Datasets to add to the current project.
            The name of the entries in the project will be identical to the
            names of the datasets.

        See Also
        --------
        add_dataset : Add a single dataset to the current project.

        Examples
        --------
        >>> import spectrochempy as scp
        >>> ds1 = scp.NDDataset([1, 2, 3])
        >>> ds2 = scp.NDDataset([4, 5, 6])
        >>> ds3 = scp.NDDataset([7, 8, 9])
        >>> proj = scp.Project()
        >>> proj.add_datasets(ds1, ds2, ds3)

        """
        for ds in datasets:
            self.add_dataset(ds)

    def add_dataset(self, dataset, name=None):
        """
        Add a single dataset to the current project.

        Parameters
        ----------
        dataset : `NDDataset`
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
        >>> import spectrochempy as scp
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

    def clear_datasets(self):
        """
        Remove all datasets from the project.

        See Also
        --------
        remove_dataset : Remove a single dataset.
        add_dataset : Add a dataset.
        """
        for v in self._datasets.values():
            v._parent = None
        self._datasets = {}

    def remove_all_dataset(self):
        """
        Remove all datasets from the project.

        .. deprecated::
            Use :meth:`clear_datasets` instead.
            Will be removed in version 0.11.0.

        See Also
        --------
        clear_datasets : Remove all datasets.
        """
        warnings.warn(
            "remove_all_dataset() is deprecated and will be removed in 0.11.0; "
            "use clear_datasets() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.clear_datasets()

    # ----------------------------------------------------------------------------------
    # project items
    # ----------------------------------------------------------------------------------
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

    def clear_projects(self):
        """
        Remove all subprojects from the current project.

        See Also
        --------
        remove_project : Remove a single subproject.
        add_project : Add a subproject.
        """
        for v in self._projects.values():
            v._parent = None
        self._projects = {}

    def remove_all_project(self):
        """
        Remove all subprojects from the current project.

        .. deprecated::
            Use :meth:`clear_projects` instead.
            Will be removed in version 0.11.0.

        See Also
        --------
        clear_projects : Remove all subprojects.
        """
        warnings.warn(
            "remove_all_project() is deprecated and will be removed in 0.11.0; "
            "use clear_projects() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.clear_projects()
