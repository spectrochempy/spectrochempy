# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

import uuid

from traitlets import (Dict, List, Bool, Instance, Unicode, HasTraits, This,
                       default)
from traitlets.config.configurable import Configurable

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.utils.meta import Meta

__all__ = ['Project', 'ProjectsOptions']

class ProjectsOptions(Configurable):

    default_directory = Unicode(help='location where all projects are '
                                     'strored by defauult').tag(config=True)

class Project(HasTraits):
    """A manager of multiple datasets in a project

    """

    _id = Unicode
    _name = Unicode(allow_none=True)
    _parent = This
    _datasets = Dict(Instance(NDDataset))
    _subprojects = Dict(This)
    _meta = Meta()

    # ........................................................................
    def __init__(self, name=None, **kwargs):

        self.name = name
        if kwargs:
            self.meta.update(kwargs)

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __getitem__(self, key):
        return self._datasets[key]

    def __setitem__(self, key, value):
        self[key] = value

    def __iter__(self):
        for items in sorted(self._datasets.items()):
            yield items

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
            raise ValueError('A parent already exists for this project',
                             'but the entered values gives a different '
                             'parent. This is not allowed' )
        self._parent = value

    # ........................................................................
    @default('_parent')
    def _get_parent(self):
        return None

    # ........................................................................
    @default('_meta')
    def _meta_default(self):
        return Meta()

    @property
    def meta(self):
        return self._meta

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    # ........................................................................
    # dataset items

    def add_datasets(self, *args):

        for ds in args:
            self.add_dataset(ds)

    def add_dataset(self, ds):

        self._datasets[ds.name] = ds

    def remove_dataset(self, name):
        del self._datasets[name]

    def remove_all_dataset(self):
        self._datasets = {}

    def dataset_names(self, sort=True):
        lst = self._datasets.keys()
        if sort:
            lst = sorted(lst)
        return lst

    def datasets(self, sort=True):
        for name in self.dataset_names(sort):
            return self._datasets[name]

    # ........................................................................
    # subproject items

    def add_subprojects(self, *args):
        """
        Add one or a series of projects to the current project.

        Parameters
        ----------
        *args : project instances


        """
        for proj in args:
            self.add_subproject(proj)

    def add_subproject(self, proj):
        """
        Add one project to the current project.

        Parameters
        ----------
        proj : a project instance


        """
        proj.parent = self
        self._subprojects[proj.name] = proj

    def remove_subproject(self, name):
        del self._subprojects[name]

    def remove_all_subproject(self):
        self._subprojects = {}

    def subprojects_names(self, sort=True):
        lst = self._subprojects.keys()
        if sort:
            lst = sorted(lst)
        return lst

    def subprojects(self, sort=True):
        for name in self.subprojects_names(sort):
            return self._subprojects[name]


# =============================================================================
if __name__ == '__main__':

    from spectrochempy.api import *
    from tests.conftest import ds1, ds2, dsm

    myp = Project(name='AGIR processing')

    ds1 = ds1()
    ds1.name = 'toto'
    ds2 = ds2()
    ds2.name = 'tata'
    ds3 = dsm()
    ds3.name = 'titi'

    myp.add_datasets(ds1, ds2, ds3)

    assert myp.dataset_names()[-1]=='toto'

    # iteration

    d=[]
    for item in myp:
        d.append(item)

    assert d[1][0] == 'titi'

    # add sub project

    msp1 = Project(name='AGIR ATG')
    msp1.add_dataset(ds1)

    msp2 = Project(name='AGIR IR')

    myp.add_subprojects(msp1, msp2)

    print(myp)