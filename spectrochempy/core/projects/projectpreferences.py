# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
__all__ = []

import os

from traitlets import (Dict, List, Bool, Instance, Unicode, HasTraits, This,
                       Any, default)
from traitlets.config.configurable import Configurable





# ============================================================================
class ProjectPreferences(Configurable) :

    def __init__(self, **kwargs):
        super(ProjectPreferences, self).__init__(**kwargs)


    project_directory = Unicode(help='Location where projects are '
                        'stored by default').tag(config=True, type='folder')

    @default('project_directory')
    def _get_default_project_directory(self):
        """
        Determines the SpectroChemPy project directory name and
        creates the directory if it doesn't exist.

        This directory is typically ``$HOME/spectrochempy/projects``,
        but if the
        SCP_PROJECTS_HOME environment variable is set and the
        ``$SCP_PROJECTS_HOME`` directory exists, it will be that
        directory.

        If neither exists, the former will be created.

        Returns
        -------
        dir : str
            The absolute path to the projects directory.

        """

        # first look for SCP_PROJECTS_HOME
        scp = os.environ.get('SCP_PROJECTS_HOME')

        if scp is not None and os.path.exists(scp) :
            return os.path.abspath(scp)

        scp = os.path.join(os.path.expanduser('~'), 'spectrochempy',
                                 'projects')

        if not os.path.exists(scp) :
            os.makedirs(scp, exist_ok=True)

        elif not os.path.isdir(scp) :
            raise IOError('Intended Projects directory is actually a file.')

        return os.path.abspath(scp)

# ============================================================================
if __name__ == '__main__' :
    pass
