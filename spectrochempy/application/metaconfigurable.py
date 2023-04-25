# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
from pathlib import Path

import numpy as np
import traitlets as tr
from matplotlib import cycler
from traitlets.config import Config
from traitlets.config.configurable import Configurable
from traitlets.config.loader import LazyConfigValue

from spectrochempy.utils.objects import Adict


class MetaConfigurable(Configurable):
    """
    A subclass of Configurable that stores configuration changes in a json file.

    Saving the configuration changes allows to retrieve them between different
    executions of the main application.
    """

    name = tr.Unicode(allow_none=True, help="Object name")

    def __init__(self, **kwargs):
        # keep only the current config section
        reset = kwargs.pop("reset", False)
        parent = kwargs.get("parent")
        parent_config = parent.config
        config = Config()
        if self.name in parent_config and not reset:
            config = Config({self.name: parent_config[self.name]})
        # call the superclass __init__ is required
        super().__init__(parent=parent, config=config)
        # get the config manager
        self.cfg = self.parent.config_manager

    @tr.default("name")
    def _name_default(self):
        # this ensures a name has been defined for the subclassed model estimators
        return self.__class__.__name__

    def to_dict(self):
        """
        Return config value in a dict form.

        Returns
        -------
        `dict`
            A regular dictionary.
        """
        d = {}
        for k, v in self.traits(config=True).items():
            d[k] = v.default_value
        return d

    def trait_defaults(self, *names, **metadata):
        # override traitlets trait default to take into accound changes in the config file
        defaults = super().trait_defaults(*names, **metadata)
        # modify with the loaded external config
        if not names:  # full dictionary
            config = self.config[self.name]
            if "shape" in config and isinstance(config["shape"], LazyConfigValue):
                del config["shape"]  # remove the lazy configurable object
            defaults.update(config)
        return defaults

    def parameters(self, default=False):
        """
        Current or default configuration values.

        Parameters
        ----------
        default : `bool`, optional, default: `False`
            If `default` is `True`, the default parameters are returned,
            else the current values.

        Returns
        -------
        `dict`
            Current or default configuration values.
        """
        d = Adict()
        if not default:
            d.update(self.trait_values(config=True))
        else:
            d.update(self.trait_defaults(config=True))
        return d

    def reset(self):
        """
        Reset configuration parameters to their default values
        """
        # for this we need to remove the section corresponding
        # to the current configurable (i.e., self.name)
        if self.name in self.config:
            # remove this entry in config
            del self.config[self.name]
            # also delete the current JSON config file
            f = (Path(self.cfg.config_dir) / self.name).with_suffix(".json")
            f.unlink(missing_ok=True)

        # then set the default parameters
        for k, v in self.parameters(default=True).items():
            if getattr(self, k) != v:
                setattr(self, k, v)

    @tr.observe(tr.All)
    def _anytrait_changed(self, change):
        # update configuration after any change

        if not hasattr(self, "cfg"):
            # not yet initialized
            return

        if change.name in self.trait_names(config=True):

            value = change.new

            # Serialization of callable functions
            # (avoid recursive functions, though!)
            # for this we use the dill library
            # (see
            # https://medium.com/@greyboi/serialising-all-the-functions-in-python-cd880a63b591)
            if callable(value):
                import dill

                value = dill.dumps(value)
                # bytes are however not JSON serialisable: make an encoded string
                import base64

                value = base64.b64encode(value).decode()

            # replace other serializable value by an equivalent
            elif isinstance(value, (type(cycler), Path)):
                value = str(value)
            if isinstance(value, np.ndarray):
                # we need to transform it to a list of elements, bUT with python built-in
                # types, which is not the case e.g., for int64
                value = value.tolist()

            self.cfg.update(
                self.name,
                {
                    self.__class__.__name__: {
                        change.name: value,
                    }
                },
            )
