# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from traitlets.config.configurable import Configurable, Config
from traitlets import All, observe


class MetaConfigurable(Configurable):

    def __init__(self, jsonfile=None, **kwargs):

        super().__init__(**kwargs)

        self.cfg = self.parent.config_manager
        self.jsonfile = self.parent.config_file_name
        if jsonfile is not None:
            self.jsonfile = jsonfile

    def to_dict(self):
        """Return config value in a dict form

        Returns
        -------
        dict
            A regular dictionary

        """

        d = self._trait_values.copy()
        keys = list(d.keys())
        for k in keys:
            if k not in self.traits(config=True):
                del d[k]
        return d

    @observe(All)
    def _anytrait_changed(self, change):
        # update configuration
        if not hasattr(self, 'cfg'):
            # not yet initialized
            return

        if change.name in self.traits(config=True):
            self.cfg.update(self.jsonfile, {
                self.__class__.__name__: {change.name: change.new, }
            })

            self.updated = True


# ======================================================================================================================
if __name__ == '__main__':
    pass
