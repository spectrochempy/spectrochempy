# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
from pathlib import Path
from matplotlib import cycler

from traitlets.config.configurable import Configurable
from traitlets import All, observe

__all__ = ["MetaConfigurable"]


class MetaConfigurable(Configurable):

    def __init__(self, jsonfile=None, **kwargs):  # lgtm [py/missing-call-to-init]

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
        d = {}
        for k, v in self.traits(config=True).items():
            d[k] = v.default_value
        return d

    @observe(All)
    def _anytrait_changed(self, change):
        # update configuration
        if not hasattr(self, 'cfg'):
            # not yet initialized
            return

        if change.name in self.traits(config=True):

            value = change.new
            if isinstance(value, (type(cycler), Path)):
                value = str(value)

            self.cfg.update(self.jsonfile, {self.__class__.__name__: {change.name: value, }})

            self.updated = True


# ======================================================================================================================
if __name__ == '__main__':
    pass
